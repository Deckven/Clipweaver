"""
Text fragment splitter. Used by report.py and the upcoming Stage 2 search.
"""

import re


def split_into_fragments(text: str) -> list[str]:
    """Split text into meaningful fragments (sentences / short paragraphs)."""
    raw = re.split(r'(?<=[.!?…])\s+', text.strip())

    fragments = []
    for chunk in raw:
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk.split()) > 40:
            sub = re.split(r'[;,]\s+', chunk)
            fragments.extend(s.strip() for s in sub if s.strip())
        else:
            fragments.append(chunk)

    return fragments
