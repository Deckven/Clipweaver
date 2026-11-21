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
    python -m clipmaker.shot_query "фрагмент нарации" --game eve_online
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import config
from .game_vocab import vocab_for

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
    source: str  # "llm" | "fragment"


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


def _cache_key(fragment: str, game: str | None, model: str) -> str:
    raw = f"{model}|{game or ''}|{fragment}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:20]


def _load_cache() -> dict[str, list[str]]:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _store_cache(cache: dict[str, list[str]]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass  # a cache miss next run is not worth failing a storyboard over


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

    request = urllib.request.Request(
        config.OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=config.SHOT_QUERY_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    if "error" in data:
        return None
    return (data.get("message") or {}).get("content", "")


def _parse_reply(reply: str, max_queries: int) -> list[str]:
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
    return lines[:max_queries]


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

    key = _cache_key(fragment, game, model)
    cache = _load_cache() if use_cache else {}
    if key in cache:
        return [ShotQuery(text=t, source="llm") for t in cache[key]]

    system = _SYSTEM.format(
        max_queries=max_queries, none=_NO_VISUAL, vocab=_vocab_block(game)
    )
    reply = _call_ollama(fragment, system, model)
    if reply is None:
        return [ShotQuery(text=fragment, source="fragment")]

    queries = _parse_reply(reply, max_queries)
    if use_cache:
        cache[key] = queries
        _store_cache(cache)
    return [ShotQuery(text=q, source="llm") for q in queries]


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Narration fragment -> shot descriptions")
    parser.add_argument("fragment")
    parser.add_argument("--game", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    queries = build_shot_queries(
        args.fragment, game=args.game, model=args.model, use_cache=not args.no_cache
    )
    if not queries:
        print("(no filmable content)")
        return
    for q in queries:
        print(f"[{q.source}] {q.text}")


if __name__ == "__main__":
    _main()
