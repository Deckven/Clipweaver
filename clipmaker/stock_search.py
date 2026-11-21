"""
Stock footage search: Pexels + Pixabay video APIs.

Returns normalized results compatible with the production plan pipeline.
Both APIs are free with rate limits (Pexels: 200/hr, Pixabay: 5000/hr).
"""
from __future__ import annotations

import requests

from . import config


def search_pexels(query: str, max_results: int = 5) -> list[dict]:
    if not config.PEXELS_API_KEY:
        return []
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": config.PEXELS_API_KEY}
    params = {"query": query, "per_page": max_results, "orientation": "landscape"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[Pexels] Error: {exc}")
        return []

    results = []
    for video in resp.json().get("videos", []):
        files = video.get("video_files", [])
        hd = next((f for f in files if f.get("quality") == "hd"), files[0] if files else {})
        results.append({
            "title": f"Pexels #{video['id']}",
            "url": video.get("url", ""),
            "preview_url": hd.get("link", ""),
            "thumbnail": video.get("image", ""),
            "duration_sec": video.get("duration", 0),
            "resolution": f"{hd.get('width', '?')}x{hd.get('height', '?')}",
            "tags": [],
            "source": "pexels",
            "license": "Pexels License (free commercial use)",
        })
    return results


def search_pixabay(query: str, max_results: int = 5) -> list[dict]:
    if not config.PIXABAY_API_KEY:
        return []
    url = "https://pixabay.com/api/videos/"
    params = {"key": config.PIXABAY_API_KEY, "q": query, "per_page": max_results}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[Pixabay] Error: {exc}")
        return []

    results = []
    for hit in resp.json().get("hits", []):
        medium = hit.get("videos", {}).get("medium", {})
        results.append({
            "title": hit.get("tags", f"Pixabay #{hit.get('id', '?')}"),
            "url": hit.get("pageURL", ""),
            "preview_url": medium.get("url", ""),
            "thumbnail": hit.get("userImageURL", ""),
            "duration_sec": hit.get("duration", 0),
            "resolution": f"{medium.get('width', '?')}x{medium.get('height', '?')}",
            "tags": [t.strip() for t in hit.get("tags", "").split(",") if t.strip()],
            "source": "pixabay",
            "license": "Pixabay License (free commercial use)",
        })
    return results


def search_stock(query: str, max_results: int = 5) -> list[dict]:
    """Search all configured stock providers, return merged results."""
    results: list[dict] = []
    results.extend(search_pexels(query, max_results=max_results))
    results.extend(search_pixabay(query, max_results=max_results))
    return results[:max_results * 2]


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "dark fantasy battle"
    print(f"Searching stock footage for: {q}\n")
    for r in search_stock(q):
        print(f"  [{r['source']}] {r['title']} ({r['duration_sec']}s)")
        print(f"    {r['url']}")
        print()
