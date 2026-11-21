"""
Search YouTube videos by text query using yt-dlp.
Supports both single queries and large text blocks (split into fragments).
"""

import importlib.util
import json
import re
import subprocess
import sys


# Russian and English stop words for query extraction
_STOP_WORDS = set(
    "и в на с по к за из от до о об у а но что как это то все его её их"
    " не он она мы вы они был была было были быть бы же ли ни так уже"
    " ещё тоже только когда если где кто чем при для через этот эта эти"
    " свой свои свою своё тот та те мой моя мои твой твоя ваш наш"
    " the a an and or but in on at to for of is are was were be been"
    " it its this that these those with from by as has have had do does"
    " will would can could should may might not no very also just then"
    " than more most some any all each every".split()
)


_missing_warned = False


def _ytdlp_cmd() -> list[str]:
    """yt-dlp from this interpreter if it is installed here (works in a venv that
    is not activated), else the executable on PATH."""
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    return ["yt-dlp"]


def search_youtube(query: str, max_results: int = 10) -> list[dict]:
    """Search YouTube and return list of video info dicts.

    Without yt-dlp the search returns an empty list and says so once, so a
    storyboard or plan is still written from the local catalog."""
    global _missing_warned
    cmd = [
        *_ytdlp_cmd(),
        f"ytsearch{max_results}:{query}",
        "--dump-json",
        "--flat-playlist",
        "--no-warnings",
        "--quiet",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, encoding="utf-8"
        )
    except FileNotFoundError:
        if not _missing_warned:
            print('[yt-dlp] not installed, YouTube search skipped (pip install -e ".[youtube]")')
            _missing_warned = True
        return []
    except subprocess.TimeoutExpired:
        print(f"[yt-dlp] Timeout for query: {ascii(query)}")
        return []

    if result.returncode != 0:
        return []

    videos = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        duration_sec = data.get("duration") or 0
        minutes, seconds = divmod(int(duration_sec), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            duration_str = f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            duration_str = f"{minutes}:{seconds:02d}"

        videos.append({
            "title": data.get("title", "Untitled"),
            "url": data.get("url") or data.get("webpage_url") or f"https://www.youtube.com/watch?v={data.get('id', '')}",
            "thumbnail": data.get("thumbnail") or data.get("thumbnails", [{}])[-1].get("url", ""),
            "duration": duration_str,
            "duration_sec": duration_sec,
            "channel": data.get("channel") or data.get("uploader") or "",
            "view_count": data.get("view_count") or 0,
            "id": data.get("id", ""),
        })

    return videos


def format_view_count(count: int) -> str:
    """Format view count for display."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def split_into_paragraphs(text: str, min_words: int = 40) -> list[str]:
    """Split large text into paragraph-sized chunks for YouTube search.

    Uses natural paragraph breaks (\n\n). Merges short paragraphs
    with the next one to avoid too many tiny fragments.
    """
    raw_paragraphs = re.split(r'\n\s*\n', text.strip())

    chunks = []
    buffer = ""
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        if buffer:
            buffer += " " + para
        else:
            buffer = para

        # Flush if buffer is long enough
        if len(buffer.split()) >= min_words:
            chunks.append(buffer)
            buffer = ""

    # Don't lose the tail
    if buffer:
        if chunks:
            chunks[-1] += " " + buffer
        else:
            chunks.append(buffer)

    return chunks


def extract_search_query(text: str, max_words: int = 8) -> str:
    """Extract a concise YouTube search query from a text fragment.

    Prioritizes proper nouns, English terms, and named entities
    over generic Russian words.
    """
    # Generic words that look like proper nouns but aren't useful for search
    _noise = {
        "кампания", "напомню", "целью", "штурм", "пятого", "формально",
        "параллельно", "одновременно", "примерно", "фактически",
    }

    # 1. Multi-word English names (Pandemic Horde, Brave Collective, etc.)
    english_multi = re.findall(r"[A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+)+", text)
    # Single English words (3+ chars)
    english_single = [w for w in re.findall(r"\b[A-Z][a-zA-Z']{2,}\b", text)
                      if w not in {"The", "War"}]
    # Abbreviations (INIT, PL, ISK, EVE — 2+ uppercase)
    abbreviations = re.findall(r'\b[A-Z]{2,}\b', text)
    # Alphanumeric codes (C-J6MT, B-9C24, R-AG7W)
    codes = re.findall(r'\b[A-Z]-[A-Z0-9]{2,}\b', text)

    # 2. Quoted terms — names and nicknames only. Narration also quotes whole
    # sentences from interviews; those are prose, not search terms.
    quoted = [q for q in re.findall(r'[«"](.*?)[»"]', text)
              if len(q.split()) <= 4]

    # 3. Capitalized Russian words not at sentence start (proper nouns)
    # Find sentence-starting words to exclude
    sentence_starts = set()
    for m in re.finditer(r'(?:^|[.!?…]\s+)([А-ЯЁ][а-яё]+)', text):
        sentence_starts.add(m.group(1))
    russian_proper = []
    for m in re.finditer(r'[А-ЯЁ][а-яё]{3,}', text):
        word = m.group()
        if word.lower() not in _noise and word not in sentence_starts:
            russian_proper.append(word)

    # Build priority list (deduplicated, order preserved)
    seen = set()
    priority = []
    for term in english_multi + codes + abbreviations + quoted + english_single + russian_proper:
        term_clean = term.strip()
        key = term_clean.lower()
        # Skip pure numbers, very short terms, or already seen
        if not term_clean or key in seen or len(term_clean) < 2:
            continue
        if term_clean.isdigit():
            continue
        seen.add(key)
        priority.append(term_clean)

    if priority:
        return " ".join(priority[:max_words])

    # Fallback: remove stop words, take longest remaining words
    words = re.findall(r'[a-zA-Zа-яА-ЯёЁ0-9]+', text.lower())
    meaningful = [w for w in words if w not in _STOP_WORDS and len(w) > 3]
    meaningful.sort(key=len, reverse=True)

    return " ".join(meaningful[:max_words])


def search_youtube_for_text(
    text: str,
    results_per_fragment: int = 3,
    min_words: int = 40,
) -> list[dict]:
    """Split a large text into fragments, search YouTube for each.

    Returns list of {fragment, query, results: [video_info]}.
    """
    paragraphs = split_into_paragraphs(text, min_words=min_words)

    storyboard = []
    for i, fragment in enumerate(paragraphs):
        query = extract_search_query(fragment)
        if not query:
            continue

        results = search_youtube(query, max_results=results_per_fragment)

        storyboard.append({
            "index": i + 1,
            "fragment": fragment,
            "query": query,
            "results": results,
        })

    return storyboard


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Search YouTube by text query (yt-dlp, nothing is downloaded)")
    ap.add_argument("query", nargs="+", help="search words")
    ap.add_argument("-n", "--max-results", type=int, default=10)
    args = ap.parse_args()
    query = " ".join(args.query)
    print(f"Searching YouTube for: {query}\n")

    results = search_youtube(query, max_results=args.max_results)
    for i, v in enumerate(results, 1):
        views = format_view_count(v["view_count"])
        print(f"{i:3d}. [{v['duration']}] {v['title']}")
        print(f"     {v['channel']} | {views} views | {v['url']}")
        print()
