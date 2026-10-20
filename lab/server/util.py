"""Paths and small helpers shared by the lab server modules."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "lab"
WEB = LAB / "web"
BOTS = REPO / "bots"
SPRITES = REPO / "visualizer" / "sprites"
SDK_PY = BOTS / "sdk" / "python"
DEFAULT_DATA = LAB / "data"
POSITION_DIRS = [REPO / "data" / "saved_gamestates", REPO / "data" / "phase_1_training_set", LAB / "maps"]


def now() -> float:
    return time.time()


def read_json(path: Path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path: Path, obj, indent=None) -> None:
    """Atomic write: readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=indent, separators=None if indent else (",", ":"))
    os.replace(tmp, path)


def sha256_files(root: Path, files) -> str:
    """Content hash of `files` (relative to root): paths and bytes, order-independent."""
    h = hashlib.sha256()
    for rel in sorted(str(f) for f in files):
        h.update(rel.encode() + b"\0")
        h.update((root / rel).read_bytes())
        h.update(b"\0")
    return h.hexdigest()


IGNORED_DIRS = {".git", "__pycache__", "build", ".venv", "node_modules", ".lab-build", ".mypy_cache", ".pytest_cache"}


def bot_files(root: Path, max_bytes: int = 200 << 20):
    """Files that make up a bot's source (what gets frozen and hashed)."""
    out, total = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith(".") or name.endswith((".pyc", ".o")):
                continue
            p = Path(dirpath) / name
            if p.is_symlink():
                continue
            total += p.stat().st_size
            if total > max_bytes:
                raise ValueError(f"bot folder {root} is larger than {max_bytes >> 20} MiB")
            out.append(p.relative_to(root))
    return out


def wilson(score: float, n: int, z: float = 1.96):
    """Wilson interval for a score probability over n games."""
    if n <= 0:
        return None, None
    denom = 1 + z * z / n
    centre = (score + z * z / (2 * n)) / denom
    half = z * math.sqrt(score * (1 - score) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def elo(score: float):
    if score is None or score <= 0 or score >= 1:
        return None
    return -400 * math.log10(1 / score - 1)
