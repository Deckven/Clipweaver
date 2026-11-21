import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Fill os.environ from the KEY=VALUE lines of a .env file.

    A variable already set in the environment wins over the file. No
    python-dotenv dependency: the file only carries a few machine-specific
    paths and keys (see .env.example). An unquoted value ends at " #"."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return
    except UnicodeDecodeError as exc:
        # PowerShell 5 `echo X=1 > .env` writes UTF-16: say so instead of an opaque traceback.
        raise RuntimeError(f"{path} is not UTF-8 text ({exc.reason}); re-save it as UTF-8") from exc
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            os.environ.setdefault(key, value)


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_path(name: str, default: Path) -> Path:
    """A path from the environment; a relative one is taken from the repo root, not the cwd."""
    value = _env_str(name)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


# CLIPMAKER_DOTENV points at another file (tests use it to switch .env off).
_load_dotenv(Path(os.environ.get("CLIPMAKER_DOTENV") or ROOT_DIR / ".env"))

# Shared VLM catalog (all indexed videos)
CATALOG_DIR = _env_path("CLIPMAKER_CATALOG_DIR", ROOT_DIR / "catalog")

# Per-video projects
PROJECTS_DIR = _env_path("CLIPMAKER_PROJECTS_DIR", ROOT_DIR / "projects")

# Footage roots — where the raw game captures and the video editor's project
# folders live on this machine. GAMES below lists folders under them. Set both
# in .env; the defaults are folders inside the repo, so a fresh clone runs
# without pointing at a drive that does not exist.
RAW_FOOTAGE_DIR = _env_path("CLIPMAKER_RAW_FOOTAGE_DIR", ROOT_DIR / "footage" / "raw")
EDIT_PROJECTS_DIR = _env_path("CLIPMAKER_EDIT_PROJECTS_DIR", ROOT_DIR / "footage" / "edit")

# Stock footage API keys (None = skip provider)
PEXELS_API_KEY: str | None = _env_str("PEXELS_API_KEY")
PIXABAY_API_KEY: str | None = _env_str("PIXABAY_API_KEY")

# Production plan settings
WORDS_PER_MIN_RU = 150
PLAN_LOCAL_RESULTS = 3
PLAN_YT_RESULTS = 3
PLAN_STOCK_RESULTS = 3

# Scratch space for VLM frame extraction (files are deleted after each video)
VLM_WORK_DIR = ROOT_DIR / "_vlm_frames"

# VLM analyzer (Stage 1: video -> JSON description)
VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_FRAME_INTERVAL_SEC = 4
VLM_FRAME_MAX_FRAMES = 50
# Max frames per VLM call. Measured 2026-10-21 (7B NF4, 512x288 frames): peak VRAM
# 9.8 GB at 10 frames, 11.9 at 15, 14.1 at 20 — 25 no longer fits a 16 GB card and the
# driver silently spills into system RAM (10x slower). 12 leaves room for the desktop.
VLM_CHUNK_SIZE = 12
VLM_FRAME_MAX_SIZE = 1024
VLM_FRAME_MAX_PIXELS = 512 * 288
OCR_ENABLED = True
OCR_LANGUAGES = ("en",)
OCR_MIN_CONFIDENCE = 0.5
OCR_MAX_LINES_PER_FRAME = 3
VLM_MAX_NEW_TOKENS_PER_FRAME = 60
# title + summary + tags cost the same regardless of chunk size. Without this term a
# short tail chunk (6-7 frames) spends its whole budget on the envelope and the JSON
# is cut off mid-scene, losing the chunk after all retries.
VLM_MAX_NEW_TOKENS_ENVELOPE = 600
VLM_MAX_NEW_TOKENS_MIN = 400
VLM_MAX_NEW_TOKENS_MAX = 2500
VLM_DTYPE = "float16"
VLM_QUANTIZE_4BIT = True  # NF4 via bitsandbytes — 7B fits in ~5 GB VRAM

# Semantic catalog index (embed_index.py)
# Narration is Russian, catalog descriptions are English -> the model must be
# multilingual or cross-language retrieval collapses to zero.
EMBED_MODEL = "intfloat/multilingual-e5-base"
EMBED_DEVICE: str | None = None  # None -> cuda when available, else cpu
EMBED_BATCH_SIZE = 64
EMBED_INDEX_DIR = CATALOG_DIR / "_index"
# Timeline captions are OCR-heavy and per-frame; merged into windows before
# embedding. Only built for videos that have no scenes (see embed_index).
# Entry granularity across the catalog is uneven — median 8 s but up to 128 s on
# coarsely indexed videos — so windows close on whichever bound hits first.
# Counting entries alone produced 13-minute "moments", useless as an edit point.
EMBED_WINDOW_FRAMES = 6
EMBED_WINDOW_MAX_SEC = 90
EMBED_MIN_CHARS = 12  # skip near-empty documents
# Scene descriptions are clean prose; window text is OCR noise; video-level text
# is accurate but says nothing about *when*. Weight hits accordingly.
EMBED_LEVEL_WEIGHTS = {"scene": 1.0, "window": 0.92, "video": 0.85}
# Rejection is by z-score, not raw cosine. E5 similarities are compressed into
# a narrow band — measured on the author's catalog, a real match scores ~0.82 and the
# control query "рецепт борща со сметаной" still scores 0.766, so any absolute
# threshold either passes nonsense or rejects real hits. The z-score of a hit
# against the same query's own score distribution separates them: worst real
# match 3.16, best nonsense 2.80.
EMBED_MIN_ZSCORE = 3.0
EMBED_MIN_COSINE = 0.70  # sanity floor only; z-score does the real work

# Narration fragment -> visual shot descriptions (shot_query.py)
# Bridges the gap the index cannot: narration says what the video is about,
# catalog descriptions say what is on screen.
OLLAMA_URL = _env_str("CLIPMAKER_OLLAMA_URL", "http://localhost:11434/api/chat")
SHOT_QUERY_MODEL = "mistral-small"
SHOT_QUERY_MAX = 3
SHOT_QUERY_TIMEOUT = 120
SHOT_QUERY_CACHE = ROOT_DIR / "_cache" / "shot_queries.json"

# Per-game catalog folders (relative to CATALOG_DIR). Add a game by copying the
# sample entry: video_dirs are where its captures live, catalog_subdir is its
# folder under CATALOG_DIR. Optional keys: "subdir_overrides" maps a capture
# subfolder to a catalog subfolder name.
SAMPLE_GAMES = {
    "assassins_creed_4": {
        "video_dirs": [RAW_FOOTAGE_DIR / "Assassin's Creed IV Black Flag Resynced"],
        "catalog_subdir": "assassins_creed_4_videos",
        "display_name": "Assassin's Creed IV: Black Flag",
    },
}
SAMPLE_DEFAULT_GAME = "assassins_creed_4"


def _load_games(local=None) -> tuple[dict, str]:
    """GAMES and DEFAULT_GAME: your clipmaker/games_local.py if it exists (optional,
    gitignored; see GUIDE.md section 3), else the sample game."""
    if local is None:
        import importlib.util
        if importlib.util.find_spec("clipmaker.games_local") is None:
            return dict(SAMPLE_GAMES), SAMPLE_DEFAULT_GAME
        from clipmaker import games_local as local
    return local.games(RAW_FOOTAGE_DIR, EDIT_PROJECTS_DIR), local.DEFAULT_GAME


GAMES, DEFAULT_GAME = _load_games()


def get_project_dir(project_name: str) -> Path:
    """Return path to a project folder, creating it if needed.

    A name starting with '-' is a CLI flag that a positional-argument parser
    took for the project (``align --project x`` once created projects/--project,
    ``align --help`` projects/--help) — refuse it instead of creating a folder."""
    if not project_name or project_name.startswith("-"):
        raise ValueError(
            f"project name {project_name!r} looks like a command-line flag, "
            "not a project folder — check the command's usage")
    p = PROJECTS_DIR / project_name
    p.mkdir(parents=True, exist_ok=True)
    return p
