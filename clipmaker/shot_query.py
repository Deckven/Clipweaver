"""
Narration fragment -> visual shot descriptions.

The semantic catalog index matches text about *what is on screen*. A narration
paragraph is about the subject, not the picture: "награды за миссии, LP и
NPC-баунти урезали на 20%" has no visual counterpart at all, and embedding it
directly scored below the rejection threshold on every fragment of the test
project. This module supplies the missing side of the bridge — it rewrites a
paragraph into short English shot descriptions, grounded in the game's visual
vocabulary from game_vocab.

Returning nothing is a valid, useful answer: a paragraph with no filmable
content should produce no queries rather than a forced guess, so the gap shows
up in the storyboard instead of being papered over with a wrong clip.

CLI:
    python -m clipmaker.shot_query "The crew sails into a storm to finish the armada" --game assassins_creed_4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import config
from .game_vocab import vocab_for
from .log import add_verbose_flag, setup_logging

log = logging.getLogger(__name__)

_NO_VISUAL = "NONE"

# Calibrated against 19 real narration fragments. Two earlier versions failed in
# opposite directions and both failures are cheap to reintroduce:
#   - stating "output NONE for opinions, statistics, patch notes" rejected 19/19,
#     because every paragraph carries commentary around its concrete content;
#   - removing the anti-fabrication rule made the model mine the vocabulary for
#     subject matter, answering a paragraph about convention demos with
#     "wormhole entry, Fortizar in a wormhole".
_SYSTEM = """You extract filmable shots from a video narration paragraph.

The footage catalogue is indexed by what is VISIBLE in each clip — \
"the player approaches a space station, examining its exterior", \
"the market overview window is open, showing item prices", \
"the player navigates through the safety settings menu".

Your job: name what the editor should put on screen while this paragraph is \
read aloud. Narration is commentary; look past it for anything a camera could \
show.

Game rules and mechanics ARE filmable — they are shown through the interface \
that displays them. A security level is the system's security status in the \
HUD. A nerfed reward is the reward window. A spawn location is the star map. \
A safety setting is the safety menu. Always prefer such an interface shot over \
{none}.

Rules:
- Output at most {max_queries} lines, one shot per line, English only.
- Each line describes a picture, not a statement. No numbering, quotes or \
explanation.
- Keep proper nouns that name a showable place or thing.
- Name ONLY things the paragraph itself mentions or directly implies. Never \
introduce a location, ship, structure or feature that is absent from the \
paragraph, even if it is common in this game. The vocabulary below is there to \
name things correctly, not to supply subject matter.
- Output exactly {none} only for pure personal opinion or a bare conclusion \
with no object, place or interface anywhere in the paragraph.

Example paragraph: "Выбор расы больше не диктует, где ты заспавнишься — все \
новички примерно на одинаковом расстоянии от Jita."
Example output:
star map showing starter systems and routes to Jita
character creation screen with race selection
newbie ship undocking from a station

Example paragraph: "Мой многолетний опыт говорит, что презентация на конвенте \
— это одно, а живая игра — совсем другое."
Example output: {none}

{vocab}"""


@dataclass(frozen=True)
class ShotQuery:
    """One visual description derived from a narration fragment."""

    text: str
    source: str  # "written" (shots.json) | "llm" (Ollama) | "fragment" (the narration itself)
    game: str | None = None  # search only this game's catalog; None = the episode's games


def load_written_shots(path: Path) -> dict[int, list[ShotQuery]]:
    """Picture queries written for the project before searching (``shots.json``).

    Two shapes are accepted::

        {"game": "my_game", "1": ["query", ...], "2": [...]}        # keys "_..." are notes
        [{"fragment": 1, "game": "my_game", "shots": ["query", ...]}, ...]

    A query is a string or ``{"text": ..., "game": ...}``. ``game`` may be set
    for the whole file, for a fragment or for one query (the closest wins) and
    must be a key of GAMES. An empty list means "nothing filmable here". A
    missing file gives ``{}``; a malformed one raises ValueError naming the spot.
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc

    if isinstance(data, dict):
        file_game = data.get("game") or data.get("_game")
        items = [(key, None, value) for key, value in data.items()
                 if not key.startswith("_") and key != "game"]
    elif isinstance(data, list):
        file_game = None
        items = []
        for n, item in enumerate(data):
            if not isinstance(item, dict) or "fragment" not in item:
                raise ValueError(f"{path}: item {n} needs a \"fragment\" number")
            items.append((item["fragment"], item.get("game"), item.get("shots", [])))
    else:
        raise ValueError(f"{path}: expected an object or a list")

    out: dict[int, list[ShotQuery]] = {}
    for key, frag_game, queries in items:
        try:
            num = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"{path}: fragment key {key!r} is not a number") from None
        if num < 1:
            raise ValueError(f"{path}: fragment numbers start at 1, got {num}")
        if not isinstance(queries, list):
            raise ValueError(f"{path}: fragment {num}: expected a list of queries")
        shots = []
        for q in queries:
            if isinstance(q, str):
                text, game = q, None
            elif isinstance(q, dict) and isinstance(q.get("text"), str):
                text, game = q["text"], q.get("game")
            else:
                raise ValueError(f"{path}: fragment {num}: a query must be a string or "
                                 f"{{\"text\": ..., \"game\": ...}}")
            game = game or frag_game or file_game
            if game is not None and game not in config.GAMES:
                raise ValueError(f"{path}: fragment {num}: unknown game {game!r}")
            if text.strip():
                shots.append(ShotQuery(text=text.strip(), source="written", game=game))
        out[num] = shots
    return out


def _vocab_block(game: str | None) -> str:
    if not game:
        return ""
    vocab = vocab_for(game)
    if not vocab:
        return ""
    parts = [f"What this game looks like: {vocab.get('description', '')}"]
    for category in ("regions", "entities", "characters", "factions"):
        items = vocab.get(category)
        if isinstance(items, list) and items:
            parts.append(f"Known {category}: {', '.join(items[:40])}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------

def _cache_path() -> Path:
    return config.SHOT_QUERY_CACHE


def _cache_key(fragment: str, system: str, model: str) -> str:
    """Cache key over everything that shapes the answer.

    The system prompt carries the game vocab and the query limit, so editing the
    prompt, a game's vocab or SHOT_QUERY_MAX asks Ollama again."""
    raw = f"{model}|{system}|{fragment}".encode()
    return hashlib.sha1(raw, usedforsecurity=False).hexdigest()[:20]


def _load_cache() -> dict[str, list[str]]:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _warn_once(f"shot query cache {path} is unreadable ({exc}); starting an empty one")
        return {}


def _store_cache(cache: dict[str, list[str]]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as exc:
        # A cache miss next run is not worth failing a storyboard over.
        _warn_once(f"shot query cache {path} not written ({exc}); the next run asks Ollama again")


# --------------------------------------------------------------------------
# ollama
# --------------------------------------------------------------------------

def _call_ollama(prompt: str, system: str, model: str) -> str | None:
    """Return the model's reply, or None if Ollama is unreachable or errors.

    Deliberately does not exit the process the way video_transcriber does — a
    storyboard must still finish (on the raw-fragment path) without Ollama.
    """
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode("utf-8")

    if not config.OLLAMA_URL.startswith(("http://", "https://")):
        _warn_once(f"CLIPMAKER_OLLAMA_URL {config.OLLAMA_URL!r} is not an http(s) URL; searching "
                   "with the narration text itself")
        return None
    request = urllib.request.Request(  # noqa: S310 - http(s) only, checked above
        config.OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        # Scheme checked above: no file:// or custom handlers from the environment.
        with urllib.request.urlopen(request, timeout=config.SHOT_QUERY_TIMEOUT) as resp:  # noqa: S310
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        _warn_once(f"Ollama at {config.OLLAMA_URL} did not answer ({exc}); searching with the "
                   f"narration text itself, which finds far less. Start Ollama for picture queries.")
        return None
    if "error" in data:
        _warn_once(f"Ollama model {model!r}: {data['error']}; searching with the narration text "
                   f"itself. Pull the model with `ollama pull {model}`.")
        return None
    return (data.get("message") or {}).get("content", "")


_warned: set[str] = set()


def _warn_once(message: str) -> None:
    """One line per distinct problem per run, not one per fragment."""
    key = message.split(";")[0]
    if key not in _warned:
        _warned.add(key)
        log.warning("[shot_query] %s", message)


def _parse_reply(reply: str, max_queries: int) -> list[str] | None:
    """Shot descriptions from the model's reply; [] when it said nothing is
    filmable, None when the reply held no usable line at all."""
    lines = []
    for raw in reply.splitlines():
        line = raw.strip().strip("-•*").strip()
        if not line:
            continue
        # strip "1." / "1)" numbering the model adds despite instructions
        if line[:1].isdigit():
            for sep in (". ", ") ", "- "):
                head, found, tail = line.partition(sep)
                if found and head.strip().isdigit():
                    line = tail.strip()
                    break
        line = line.strip('"').strip()
        if not line:
            continue
        if line.upper().startswith(_NO_VISUAL):
            return []
        lines.append(line)
    return lines[:max_queries] if lines else None


# --------------------------------------------------------------------------
# public
# --------------------------------------------------------------------------

def build_shot_queries(
    fragment: str,
    game: str | None = None,
    model: str | None = None,
    max_queries: int | None = None,
    use_cache: bool = True,
) -> list[ShotQuery]:
    """Rewrite a narration fragment into visual shot descriptions.

    Returns an empty list when the paragraph has no filmable content. Falls back
    to a single ``source="fragment"`` query when Ollama is unavailable, so the
    caller behaves exactly as it did before this module existed.
    """
    fragment = (fragment or "").strip()
    if not fragment:
        return []

    model = model or config.SHOT_QUERY_MODEL
    max_queries = max_queries or config.SHOT_QUERY_MAX

    system = _SYSTEM.format(
        max_queries=max_queries, none=_NO_VISUAL, vocab=_vocab_block(game)
    )
    key = _cache_key(fragment, system, model)
    cache = _load_cache() if use_cache else {}
    if key in cache:
        return [ShotQuery(text=t, source="llm") for t in cache[key]]

    reply = _call_ollama(fragment, system, model)
    if reply is None or not reply.strip():
        # An empty reply is a failure, not "nothing filmable": do not cache it,
        # or one hiccup would hide this fragment from every later storyboard.
        return [ShotQuery(text=fragment, source="fragment")]

    queries = _parse_reply(reply, max_queries)
    if queries is None:
        # Only bullets, numbers or quotes: a failed answer, not "nothing
        # filmable" - not cached, searched with the fragment itself.
        _warn_once(f"Ollama model {model!r} gave no usable shot description; searching "
                   "with the narration text itself for such fragments")
        return [ShotQuery(text=fragment, source="fragment")]
    if not queries:
        log.info("[shot_query] nothing filmable in: %s", fragment[:60])
    if use_cache:
        cache[key] = queries
        _store_cache(cache)
    return [ShotQuery(text=q, source="llm") for q in queries]


def main(argv: list[str] | None = None) -> None:
    """CLI: print the picture queries for one narration fragment."""
    parser = argparse.ArgumentParser(description="Narration fragment -> shot descriptions")
    parser.add_argument("fragment")
    parser.add_argument("--game", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-cache", action="store_true")
    add_verbose_flag(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    queries = build_shot_queries(
        args.fragment, game=args.game, model=args.model, use_cache=not args.no_cache
    )
    if not queries:
        print("(no filmable content)")
        return
    for q in queries:
        print(f"[{q.source}] {q.text}")


if __name__ == "__main__":
    main()
