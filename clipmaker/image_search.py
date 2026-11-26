"""
Still-image search for storyboard fragments that have no usable footage.

Two keyless MediaWiki providers:

* **Fandom** — game wikis (Assassin's Creed, Diablo, ...). Publisher artwork
  and renders. Copyrighted by the publisher; usable as editorial/commentary
  b-roll, not as free stock. ``license`` is reported as ``publisher`` so the
  caller never mistakes it for cleared material.
* **Wikimedia Commons** — historical paintings, photographs, codex scans.
  Mostly public domain; the real license string is carried through.

Both APIs are anonymous and rate-limited politely by a descriptive User-Agent.
"""
from __future__ import annotations

import argparse
import logging
import sys

import requests

from .log import add_verbose_flag, setup_logging

log = logging.getLogger(__name__)

USER_AGENT = "clipmaker-storyboard/0.1 (personal video research)"
_HEADERS = {"User-Agent": USER_AGENT}
_TIMEOUT = 20


def _get(url: str, params: dict[str, str | int]) -> requests.Response:
    """GET a MediaWiki API with the descriptive User-Agent and the timeout."""
    return requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)

# File extensions worth putting on a timeline. Wikis are full of icon SVGs.
_RASTER_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")

# Wiki chrome that shows up on almost every Fandom page.
_FANDOM_NOISE = ("icon", "logo", "wiki-wordmark", "favicon", "badge", "sprite")


def _normalize(page: dict, source: str, license_name: str) -> dict | None:
    """Turn a MediaWiki page dict into a storyboard image record."""
    info = (page.get("imageinfo") or [{}])[0]
    file_url = info.get("url")
    if not file_url:
        return None
    return {
        "title": page.get("title", "").removeprefix("File:"),
        "page_url": info.get("descriptionurl", ""),
        "file_url": file_url,
        "thumb_url": info.get("thumburl", file_url),
        "width": info.get("width", 0),
        "height": info.get("height", 0),
        "source": source,
        "license": license_name,
    }


def _is_usable(title: str, min_width: int) -> bool:
    lowered = title.lower()
    if not lowered.endswith(_RASTER_SUFFIXES):
        return False
    return not any(noise in lowered for noise in _FANDOM_NOISE)


def fandom_search_pages(
    query: str,
    wiki: str = "assassinscreed",
    limit: int = 3,
) -> list[str]:
    """Find article titles on a Fandom wiki matching a free-text query.

    ``fandom_page_images`` needs an exact article title, which a storyboard
    query never is ("Assassin's Creed Black Flag Observatory" is not a page).
    Searching first turns the query into real titles.
    """
    try:
        resp = _get(
            f"https://{wiki}.fandom.com/api.php",
            params={
                "action": "query", "list": "search", "srsearch": query,
                "srlimit": limit, "srnamespace": 0, "format": "json",
            },
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("[Fandom] Search error for '%s': %s", query, exc)
        return []
    return [hit["title"] for hit in resp.json().get("query", {}).get("search", [])]


def fandom_images_for_query(
    query: str,
    wiki: str = "assassinscreed",
    limit: int = 4,
    min_width: int = 400,
) -> list[dict]:
    """Search the wiki for a query, then collect images off the best articles.

    Walks down the search results until enough usable images accumulate — the
    top article is often a disambiguation or list page carrying nothing but
    chrome.
    """
    collected: list[dict] = []
    seen: set[str] = set()
    for title in fandom_search_pages(query, wiki=wiki, limit=3):
        for image in fandom_page_images(title, wiki=wiki, limit=limit * 2,
                                        min_width=min_width):
            if image["file_url"] in seen:
                continue
            seen.add(image["file_url"])
            image["article"] = title
            collected.append(image)
            if len(collected) >= limit:
                return collected
    return collected


def fandom_page_images(
    page: str,
    wiki: str = "assassinscreed",
    limit: int = 12,
    min_width: int = 400,
) -> list[dict]:
    """Return images embedded on a Fandom wiki article.

    Two calls: list the file titles on the page, then resolve them to URLs.
    Redirects are followed — AC wiki files artifacts under plural titles
    (``"Shroud of Eden"`` -> ``"Shrouds of Eden"``), and without ``redirects``
    those pages silently come back empty.
    """
    api = f"https://{wiki}.fandom.com/api.php"

    try:
        listing = _get(
            api,
            params={
                "action": "query", "titles": page, "prop": "images",
                "imlimit": limit * 3, "redirects": 1, "format": "json",
            },
        )
        listing.raise_for_status()
    except requests.RequestException as exc:
        log.warning("[Fandom] Error listing '%s': %s", page, exc)
        return []

    titles = [
        image["title"]
        for entry in listing.json().get("query", {}).get("pages", {}).values()
        for image in entry.get("images", [])
        if _is_usable(image["title"], min_width)
    ]
    if not titles:
        return []

    try:
        details = _get(
            api,
            params={
                "action": "query", "titles": "|".join(titles[:limit]),
                "prop": "imageinfo", "iiprop": "url|size",
                "iiurlwidth": 1280, "format": "json",
            },
        )
        details.raise_for_status()
    except requests.RequestException as exc:
        log.warning("[Fandom] Error resolving files for '%s': %s", page, exc)
        return []

    results = []
    for entry in details.json().get("query", {}).get("pages", {}).values():
        record = _normalize(entry, source=f"fandom:{wiki}", license_name="publisher")
        if record and record["width"] >= min_width:
            results.append(record)
    return results


def commons_search(query: str, limit: int = 5, min_width: int = 600) -> list[dict]:
    """Search Wikimedia Commons for still images matching a query."""
    try:
        resp = _get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query", "generator": "search", "gsrsearch": query,
                "gsrnamespace": 6, "gsrlimit": limit * 2,
                "prop": "imageinfo", "iiprop": "url|size|extmetadata",
                "iiurlwidth": 1280, "format": "json",
            },
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("[Commons] Error for '%s': %s", query, exc)
        return []

    results = []
    for entry in resp.json().get("query", {}).get("pages", {}).values():
        if not _is_usable(entry.get("title", ""), min_width):
            continue
        info = (entry.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        license_name = (meta.get("LicenseShortName") or {}).get("value", "unknown")
        record = _normalize(entry, source="commons", license_name=license_name)
        if record and record["width"] >= min_width:
            results.append(record)
    return results[:limit]


def main(argv: list[str] | None = None) -> None:
    """CLI: ``fandom QUERY --wiki eveonline`` | ``commons QUERY``."""
    parser = argparse.ArgumentParser(prog="clipmaker.image_search")
    parser.add_argument("provider", choices=("fandom", "commons"))
    parser.add_argument("query", nargs="+")
    parser.add_argument("--wiki", default="assassinscreed",
                        help="Fandom subdomain: eveonline, wowpedia, diablo, ...")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--min-width", type=int, default=600)
    add_verbose_flag(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    query = " ".join(args.query)
    if args.provider == "fandom":
        found = fandom_images_for_query(query, wiki=args.wiki, limit=args.limit,
                                        min_width=args.min_width)
    else:
        found = commons_search(query, limit=args.limit, min_width=args.min_width)

    if not found:
        print("  nothing found — rephrase as an article title or try the other provider")
    for image in found:
        where = f" — {image['article']}" if image.get("article") else ""
        print(f"  [{image['license']}] {image['title']} "
              f"({image['width']}x{image['height']}){where}")
        print(f"    {image['file_url']}")


if __name__ == "__main__":
    main()
