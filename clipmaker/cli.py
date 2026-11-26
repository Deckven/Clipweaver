"""The ``clipweaver`` command: one entry point for every tool of the package.

``clipweaver <tool> [args]`` runs ``python -m clipmaker.<tool> [args]``; the
tool's own ``--help`` lists its options. Modules are imported only when their
tool runs, so ``clipweaver --help`` needs none of the optional extras.
"""
from __future__ import annotations

import importlib
import sys

# tool -> (module, one-line description), in pipeline order.
TOOLS: dict[str, tuple[str, str]] = {
    "analyzer": ("analyzer", "index videos into the catalog with the VLM (GPU); rename them"),
    "embed_index": ("embed_index", "build, search or describe the semantic index of the catalog"),
    "ocrgrep": ("ocrgrep", "regex over the OCR and scene text of a game's catalog"),
    "grid": ("grid", "frame grid of a video found by name"),
    "stills": ("stills", "single frames and contact sheets from files or YouTube"),
    "image_search": ("image_search", "still images from Fandom wikis and Wikimedia Commons"),
    "video_transcriber": ("video_transcriber", "transcribe a video and structure it with a local LLM"),
    "transcribe": ("transcribe", "word-timed transcript of a project's narration"),
    "shot_query": ("shot_query", "picture queries for one narration fragment (Ollama)"),
    "report": ("report", "storyboard: catalog and YouTube matches per narration fragment"),
    "production_plan": ("production_plan", "coverage per fragment: catalog, YouTube, stock, to shoot"),
    "shot_classifier": ("shot_classifier", "shot type and stock query of narration fragments"),
    "youtube_search": ("youtube_search", "search YouTube by words (nothing is downloaded)"),
    "stock_search": ("stock_search", "search stock video on Pexels and Pixabay"),
    "align": ("align", "find where each storyboard fragment is spoken"),
    "build_storyboard": ("build_storyboard", "merge curation, moments and audio timings"),
    "clip_extractor": ("clip_extractor", "cut every fragment's clip into clips/"),
    "contact_sheet": ("contact_sheet", "HTML sheet of the frames each fragment will show"),
    "layout": ("layout", "validated layout.txt for the Vegas loader"),
    "resolve_export": ("resolve_export", "layout as OTIO and FCPXML for DaVinci Resolve"),
    "vegas_takes": ("vegas_takes", "copy vegas.json raw-take lags into layout.txt"),
    "take_lags": ("take_lags", "where each raw take sits inside the merged narration"),
    "veg_scan": ("veg_scan", "check take lags against a .veg project"),
    "remap": ("remap", "move Vegas events onto a re-voiced narration"),
    "render_offset": ("render_offset", "shift between the narration and a rendered video"),
    "vegas_export": ("vegas_export", "structured transcript sections as JSON for Vegas"),
    "rename_videos": ("rename_videos", "rename videos to their suggested names (dry run by default)"),
    "retrieval_eval": ("retrieval_cli", "recall@k of the catalog search against a gold set"),
}


def usage() -> str:
    """The tool list shown by ``clipweaver --help``."""
    width = max(map(len, TOOLS))
    rows = "\n".join(f"  {name.ljust(width)}  {desc}" for name, (_, desc) in TOOLS.items())
    return ("usage: clipweaver <tool> [args]   (same as python -m clipmaker.<tool>)\n\n"
            f"tools:\n{rows}\n\n"
            "clipweaver <tool> --help shows the tool's options.")


def main(argv: list[str] | None = None) -> None:
    """Run the tool named by the first argument with the rest of the arguments."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(usage())
        return
    name = args[0].replace("-", "_")
    if name not in TOOLS:
        print(f"clipweaver: unknown tool {args[0]!r}\n\n{usage()}", file=sys.stderr)
        sys.exit(2)
    module = importlib.import_module(f"clipmaker.{TOOLS[name][0]}")
    module.main(args[1:])


if __name__ == "__main__":
    main()
