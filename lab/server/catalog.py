"""Bots and maps the lab can use, plus freezing and launching bots."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import shutil
import threading
import time
import zipfile
from pathlib import Path

from .native import BuildError, Native, python_executable
from .util import BOTS, POSITION_DIRS, REPO, bot_files, read_json, sha256_files, write_json

GENERATED_SIZES = [9, 11, 13, 16]
BUILTINS = [
    {"id": "builtin:random", "name": "random (builtin)", "description": "Uniformly random playable action. Self-play only."},
    {"id": "builtin:endturn", "name": "endturn (builtin)", "description": "Always ends the turn. Self-play only."},
]


def detect_manifest(d: Path) -> dict | None:
    """bot.json, or a recognisable C++/Python bot without one."""
    m = read_json(d / "bot.json")
    if isinstance(m, dict):
        m.setdefault("language", "cpp")
        return m
    for cpp in sorted(d.glob("*.cpp")):
        try:
            if "POLYSHARK_BOT(" in cpp.read_text(errors="ignore"):
                return {"language": "cpp", "sources": ["*.cpp"]}
        except OSError:
            pass
    if (d / "bot.py").is_file():
        return {"language": "python", "entry": "bot.py"}
    return None


def _scan(root: Path, depth: int = 3):
    """Yields (bot_dir, manifest, folder) under root. folder is the path of the
    sub-folder holding the bot, relative to root (None directly under root)."""
    m = detect_manifest(root)
    if m is not None:
        yield root, m, None
        return
    yield from _scan_folder(root, depth, None)


def _scan_folder(folder: Path, depth: int, label):
    if depth <= 0 or not folder.is_dir():
        return
    for child in sorted(folder.iterdir()):
        if not child.is_dir() or child.name.startswith(".") or child.name in ("sdk", "build", "__pycache__"):
            continue
        m = detect_manifest(child)
        if m is not None:
            yield child, m, label
        else:
            yield from _scan_folder(child, depth - 1, child.name if label is None else f"{label}/{child.name}")


class Catalog:
    def __init__(self, data: Path, native: Native):
        self.data = data
        self.native = native
        self.prefs_path = data / "catalog.json"
        self.uploads = data / "uploads"
        self.lock = threading.RLock()
        self.prefs = read_json(self.prefs_path, {}) or {}
        for k, v in (("stars", {}), ("names", {}), ("removed", []), ("paths", []), ("hidden", [])):
            self.prefs.setdefault(k, v)
        self._bots_cache = None
        self._bots_at = 0.0

    def _save(self):
        write_json(self.prefs_path, self.prefs, indent=2)
        self._bots_cache = None

    # -------------------------------------------------------------- bots --

    def _discover(self):
        found = []
        for d, m, folder in _scan(BOTS):
            found.append(("bots/" + str(d.relative_to(BOTS)), d, m, folder, "repo"))
        for p in self.prefs["paths"]:
            root = Path(p)
            if not root.is_dir():
                continue
            single = detect_manifest(root) is not None
            for d, m, folder in _scan(root):
                label = None if single else (root.name + ("/" + folder if folder else ""))
                found.append(("path:" + str(d), d, m, label, "path"))
        if self.uploads.is_dir():
            for up in sorted(self.uploads.iterdir()):
                meta = read_json(up / "upload.json", {})
                root = up / "files"
                single = detect_manifest(root) is not None
                for d, m, folder in _scan(root):
                    rel = str(d.relative_to(root)) if d != root else "."
                    label = None if single else (meta.get("name", up.name) + ("/" + folder if folder else ""))
                    found.append((f"upload:{up.name}/{rel}", d, m, label, "upload"))
        return found

    def bots(self, refresh: bool = False):
        with self.lock:
            if not refresh and self._bots_cache is not None and time.time() - self._bots_at < 3:
                return self._bots_cache
            out = []
            hidden = set(self.prefs["hidden"])
            removed = set(self.prefs["removed"])
            for bid, d, m, folder, source in self._discover():
                if bid in hidden:
                    continue
                default = m.get("name") or d.name
                try:
                    detail = str(d.relative_to(REPO))
                except ValueError:
                    detail = str(d)
                out.append({
                    "id": bid, "name": self.prefs["names"].get(bid) or default, "default_name": default,
                    "folder": folder, "dir": str(d), "detail": detail, "source": source,
                    "language": m.get("language", "cpp"), "description": m.get("description", ""),
                    "starred": self.prefs["stars"].get(bid, True) and bid not in removed,
                    "removed": bid in removed, "manifest": m,
                })
            self._bots_cache, self._bots_at = out, time.time()
            return out

    def bot(self, bot_id: str) -> dict | None:
        if bot_id.startswith("builtin:"):
            b = next((x for x in BUILTINS if x["id"] == bot_id), None)
            return dict(b, language="builtin", builtin=bot_id.split(":", 1)[1]) if b else None
        return next((b for b in self.bots() if b["id"] == bot_id), None)

    def set_star(self, bot_id, on: bool):
        with self.lock:
            self.prefs["stars"][bot_id] = bool(on)
            self._save()

    def rename(self, bot_id, name: str):
        with self.lock:
            if name.strip():
                self.prefs["names"][bot_id] = name.strip()
            else:
                self.prefs["names"].pop(bot_id, None)
            self._save()

    def set_removed(self, bot_id, removed: bool):
        with self.lock:
            s = set(self.prefs["removed"])
            (s.add if removed else s.discard)(bot_id)
            self.prefs["removed"] = sorted(s)
            self._save()

    def delete(self, bot_id: str):
        """Forget a loaded/uploaded bot (repo bots can only be removed from the picker)."""
        with self.lock:
            if bot_id.startswith("path:"):
                p = bot_id[5:]
                if p in self.prefs["paths"]:
                    self.prefs["paths"].remove(p)
                else:
                    self.prefs["hidden"].append(bot_id)
            elif bot_id.startswith("upload:"):
                uid = bot_id[7:].split("/", 1)[0]
                bots_in_upload = [b for b in self.bots() if b["id"].startswith(f"upload:{uid}/")]
                if len(bots_in_upload) <= 1:
                    shutil.rmtree(self.uploads / uid, ignore_errors=True)
                else:
                    self.prefs["hidden"].append(bot_id)
            else:
                raise ValueError("bots in the repository can be removed from the picker, not deleted")
            self._save()

    def delete_folder(self, folder: str):
        with self.lock:
            for b in self.bots(refresh=True):
                if b["folder"] == folder or (b["folder"] or "").startswith(folder + "/"):
                    if b["source"] == "repo":
                        self.set_removed(b["id"], True)
                    else:
                        self.delete(b["id"])
            self._save()

    def load_path(self, path: str):
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (REPO / p).resolve()
        if not p.is_dir():
            raise ValueError(f"{p} is not a directory")
        if not any(True for _ in _scan(p)):
            raise ValueError(f"no bots found in {p} (expected bot.json, a POLYSHARK_BOT .cpp, or bot.py)")
        with self.lock:
            s = str(p)
            if s not in self.prefs["paths"]:
                self.prefs["paths"].append(s)
            self.prefs["hidden"] = [h for h in self.prefs["hidden"] if not h.startswith("path:" + s)]
            self._save()
        return [b["id"] for b in self.bots(refresh=True) if b["dir"].startswith(str(p))]

    def upload_zip(self, filename: str, payload: bytes):
        uid = hashlib.sha256(payload).hexdigest()[:10]
        dest = self.uploads / uid
        if not dest.exists():
            tmp = self.uploads / (uid + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            files = tmp / "files"
            files.mkdir(parents=True)
            total = 0
            with zipfile.ZipFile(io.BytesIO(payload)) as z:
                for info in z.infolist():
                    name = info.filename
                    parts = Path(name).parts
                    if name.startswith("/") or ".." in parts or parts[:1] == ("__MACOSX",):
                        continue
                    if (info.external_attr >> 16) & 0o170000 == 0o120000:  # symlink
                        continue
                    total += info.file_size
                    if total > 500 << 20:
                        raise ValueError("ZIP expands beyond 500 MiB")
                    target = files / name
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
            # A single top-level folder is the bot (or folder of bots) itself.
            entries = [e for e in files.iterdir() if not e.name.startswith(".")]
            if len(entries) == 1 and entries[0].is_dir() and detect_manifest(files) is None:
                inner = entries[0]
                tmp_inner = tmp / "inner"
                os.replace(inner, tmp_inner)
                shutil.rmtree(files)
                os.replace(tmp_inner, files)
            write_json(tmp / "upload.json", {"name": Path(filename).stem or uid, "file": filename,
                                             "uploaded_at": time.time(), "sha256": hashlib.sha256(payload).hexdigest()})
            os.replace(tmp, dest)
        if not any(True for _ in _scan(dest / "files")):
            shutil.rmtree(dest, ignore_errors=True)
            raise ValueError("the ZIP contains no bot (expected bot.json, a POLYSHARK_BOT .cpp, or bot.py)")
        with self.lock:
            self.prefs["hidden"] = [h for h in self.prefs["hidden"] if not h.startswith(f"upload:{uid}/")]
            self._save()
        return [b["id"] for b in self.bots(refresh=True) if b["id"].startswith(f"upload:{uid}/")]

    # ------------------------------------------------------- freezing --

    def freeze(self, bot: dict, dest: Path) -> dict:
        """Copies the bot's current source into dest and returns its content hash."""
        src = Path(bot["dir"])
        files = bot_files(src)
        if dest.exists():
            shutil.rmtree(dest)
        for rel in files:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, dest / rel)
        return {"hash": sha256_files(dest, files)[:16], "files": len(files),
                "bytes": sum((dest / r).stat().st_size for r in files)}

    def prepare(self, bot: dict, frozen_dir: Path, ehash: str, source_hash: str) -> dict:
        """Builds (if needed) and returns how to launch a frozen bot: cmd, cwd, env, lib."""
        m = bot["manifest"]
        lang = m.get("language", "cpp")
        if lang == "cpp":
            info = self.native.build_cpp_bot(ehash, frozen_dir, m, source_hash)
            return {"cmd": [info["exe"]], "cwd": str(frozen_dir), "env": {}, "lib": info["lib"],
                    "build": info["key"], "build_log": info["log"]}
        sdk = self.native.engine_dir(ehash) / "src" / "bots" / "sdk" / "python"
        if lang == "python":
            entry = m.get("entry", "bot.py")
            env = {"PYTHONPATH": os.pathsep.join([str(sdk), str(frozen_dir)]), "PYTHONUNBUFFERED": "1",
                   "PYTHONDONTWRITEBYTECODE": "1"}
            return {"cmd": [python_executable(m), entry], "cwd": str(frozen_dir), "env": env, "lib": None}
        if lang == "command":
            run = m.get("run")
            cmd = shlex.split(run) if isinstance(run, str) else list(run or [])
            if not cmd:
                raise BuildError("command bots need a \"run\" entry in bot.json")
            if (frozen_dir / cmd[0]).exists():
                cmd[0] = str(frozen_dir / cmd[0])
            return {"cmd": cmd, "cwd": str(frozen_dir), "env": {"PYTHONPATH": str(sdk)}, "lib": None}
        raise BuildError(f"unknown bot language {lang!r}")

    # -------------------------------------------------------------- maps --

    def maps(self):
        out = []
        for n in GENERATED_SIZES:
            out.append({"id": f"gen:drylands:{n}", "name": f"Drylands {n}×{n}", "kind": "generated", "size": n,
                        "dims": f"{n}×{n}", "folder": "Generated", "detail": "Engine map generator · new seed every pair"})
        for d in POSITION_DIRS:
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.json")):
                j = read_json(p)
                if not isinstance(j, dict) or "tiles" not in j:
                    continue
                n = j.get("map_size", 9)
                rel = str(p.relative_to(REPO))
                out.append({"id": "file:" + rel, "name": p.stem, "kind": "state", "size": n, "dims": f"{n}×{n}",
                            "folder": str(d.relative_to(REPO)),
                            "detail": f"Position · P{j.get('current_player', 0)} to move · turn {j.get('turn', 0)}"})
        return out

    def map_start(self, map_id: str) -> dict:
        """The frozen start spec for a map (generated maps get their seed per game)."""
        if map_id.startswith("gen:"):
            _, gen, size = map_id.split(":")
            return {"kind": "generated", "generator": gen, "size": int(size)}
        if map_id.startswith("file:"):
            p = (REPO / map_id[5:]).resolve()
            if not any(str(p).startswith(str(d.resolve())) for d in POSITION_DIRS):
                raise ValueError("map outside the position folders")
            state = read_json(p)
            if not isinstance(state, dict):
                raise ValueError(f"cannot read {p}")
            return {"kind": "state", "state": state}
        raise ValueError(f"unknown map {map_id}")

    def map(self, map_id: str) -> dict | None:
        return next((m for m in self.maps() if m["id"] == map_id), None)

    def public_bots(self):
        return [{k: v for k, v in b.items() if k != "manifest"} for b in self.bots()]

    def public_manifest(self, bot: dict) -> str:
        return json.dumps(bot.get("manifest", {}), indent=2)
