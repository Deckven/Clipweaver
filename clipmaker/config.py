"""Settings: paths from .env (see .env.example), model choices and tuning constants, games.

Machine-specific values (footage roots, catalog and projects folders, API keys)
come from the environment or a .env file next to the repository; everything
else is a constant here. GAMES comes from clipmaker/games_local.py when it
exists, else the sample game.
"""
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def _data_root() -> Path:
    """The folder .env, relative paths and the default catalog/projects hang off.

    CLIPMAKER_HOME when set. Otherwise the repository when the package runs from
    a checkout (pyproject.toml next to it: a clone, or ``pip install -e .``), and
    the current folder for a regular ``pip install .`` - never site-packages,
    where catalog/, _cache/ and _vlm_frames/ would be written into the Python
    installation itself.
    """
    home = os.environ.get("CLIPMAKER_HOME", "").strip()
    if home:
        return Path(home).expanduser().resolve()
    checkout = PACKAGE_DIR.parent
    if (checkout / "pyproject.toml").is_file():
        return checkout
    return Path.cwd().resolve()


ROOT_DIR = _data_root()


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
        closing = value.find(value[0], 1) if value[:1] in ("'", '"') else -1
        if closing != -1:
            value = value[1:closing]  # a quoted value ends at its quote; a comment may follow
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            os.environ.setdefault(key, value)


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_path(name: str, default: Path) -> Path:
    """A path from the environment; a relative one is taken from ROOT_DIR, not the cwd."""
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


def parse_path_map(text: str | None) -> list[tuple[str, str]]:
    """``"OLD => NEW; OLD2 => NEW2"`` as (old, new) pairs, longest OLD first."""
    pairs = []
    for part in (text or "").split(";"):
        if "=>" not in part:
            continue
        old, new = (s.strip().replace("\\", "/").rstrip("/") for s in part.split("=>", 1))
        if old and new:
            pairs.append((old, new))
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


# Footage that moved (an archive drive, a renamed folder): a catalog or layout
# path starting with OLD is also looked up under NEW. Only lookups change — the
# catalog keeps the paths it was indexed with.
PATH_MAP = parse_path_map(_env_str("CLIPMAKER_PATH_MAP"))

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
OLLAMA_URL: str = _env_str("CLIPMAKER_OLLAMA_URL") or "http://localhost:11434/api/chat"
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


def _load_games(local=None, path: Path | None = None) -> tuple[dict, str]:
    """GAMES and DEFAULT_GAME: your clipmaker/games_local.py if it exists (optional,
    gitignored; see GUIDE.md section 3), else games_local.py in ROOT_DIR (where a
    regular, non-editable install looks), else the sample game.

    The file is loaded by its path next to this one, not as ``clipmaker.games_local``:
    other tools (Newsroom's footage gate) load config.py on its own, without the
    clipmaker package on sys.path, and must still see your games."""
    if local is None:
        if path is None:
            candidates = [PACKAGE_DIR / "games_local.py", ROOT_DIR / "games_local.py"]
            path = next((p for p in candidates if p.is_file()), candidates[0])
        if not path.is_file():
            return dict(SAMPLE_GAMES), SAMPLE_DEFAULT_GAME
        import importlib.util
        spec = importlib.util.spec_from_file_location("clipmaker_games_local", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load the games file {path}")
        local = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(local)
    return local.games(RAW_FOOTAGE_DIR, EDIT_PROJECTS_DIR), local.DEFAULT_GAME


GAMES, DEFAULT_GAME = _load_games()


def game_config(game: str) -> dict:
    """GAMES[game], or ValueError naming the known games."""
    cfg = GAMES.get(game)
    if not cfg:
        raise ValueError(f"Unknown game '{game}'. Known: {list(GAMES)}")
    return cfg


def project_path(project_name: str) -> Path:
    """projects/<name>, not created. Refuses a name that looks like a CLI flag (a
    positional-argument parser took ``--help`` for the project) and a name that
    leads outside PROJECTS_DIR (``../x``, an absolute path)."""
    if not project_name or project_name.startswith("-"):
        raise ValueError(
            f"project name {project_name!r} looks like a command-line flag, "
            "not a project folder — check the command's usage")
    p = PROJECTS_DIR / project_name
    if not p.resolve().is_relative_to(PROJECTS_DIR.resolve()):
        raise ValueError(f"project name {project_name!r} points outside {PROJECTS_DIR}")
    return p


def get_project_dir(project_name: str) -> Path:
    """Return path to a project folder, creating it if needed (see project_path)."""
    p = project_path(project_name)
    p.mkdir(parents=True, exist_ok=True)
    return p
