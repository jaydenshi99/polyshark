"""Frozen engine snapshots, native builds (referee, tool, self-play, bots) and the tool pool.

Every native binary is built from a snapshot of the engine + lab runtime keyed by
its content hash, so a match replays with exactly the code it was played with.
"""
from __future__ import annotations

import concurrent.futures
import glob
import hashlib
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
from collections import defaultdict
from pathlib import Path

from .util import REPO, read_json, sha256_files, write_json

# -ffp-contract=off: no FMA contraction, so float results match on every CPU.
FLAGS = ["-std=c++17", "-O2", "-ffp-contract=off", "-fPIC", "-fvisibility=hidden", "-fvisibility-inlines-hidden"]
JSON_CANDIDATES = [
    REPO / "build" / "_deps" / "nlohmann_json-src" / "single_include",
    REPO / "third_party",
    Path("/opt/homebrew/include"),
    Path("/usr/local/include"),
    Path("/usr/include"),
]


class BuildError(RuntimeError):
    def __init__(self, message, log=""):
        super().__init__(message)
        self.log = log


def _find_json_header() -> Path | None:
    for base in JSON_CANDIDATES:
        p = base / "nlohmann" / "json.hpp"
        if p.is_file():
            return p
    return None


def _runtime_sources():
    """(absolute path, snapshot-relative path) of everything that defines game execution."""
    files = []
    pats = ["engine/include/*.h", "engine/src/*.cpp", "engine/src/game_state/*.cpp",
            "lab/native/*.h", "lab/native/*.cpp", "bots/sdk/cpp/*.h", "bots/sdk/python/*.py"]
    for pat in pats:
        for p in sorted(glob.glob(str(REPO / pat))):
            rel = os.path.relpath(p, REPO)
            if rel.endswith("test_lab.cpp"):
                continue
            files.append((Path(p), rel))
    header = _find_json_header()
    if header:
        files.append((header, "third_party/nlohmann/json.hpp"))
    return files


class ToolProcess:
    def __init__(self, path: Path):
        self.proc = subprocess.Popen([str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def call(self, req: dict) -> dict:
        self.proc.stdin.write(json.dumps(req, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("polyshark-tool exited")
        return json.loads(line)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self):
        try:
            self.proc.kill()
        except OSError:
            pass


class ToolPool:
    """A few persistent polyshark-tool processes for one engine hash."""

    def __init__(self, path: Path, size: int = 3):
        self.path, self.size = path, size
        self.free: queue.Queue = queue.Queue()
        self.created = 0
        self.lock = threading.Lock()

    def call(self, req: dict) -> dict:
        proc = None
        with self.lock:
            if self.free.empty() and self.created < self.size:
                self.created += 1
                proc = ToolProcess(self.path)
        if proc is None:
            proc = self.free.get()
        try:
            return proc.call(req)
        except Exception:
            proc.close()
            proc = None
            with self.lock:
                self.created -= 1
            raise
        finally:
            if proc is not None:
                self.free.put(proc)


class Native:
    def __init__(self, data: Path):
        self.data = data
        self.engines = data / "engines"
        self.bot_builds = data / "builds"
        self.cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("c++") or "c++"
        self._locks = defaultdict(threading.Lock)
        self._lock = threading.Lock()
        self._pools: dict[str, ToolPool] = {}
        self._hash_cache = (None, None)
        self.jobs = max(1, (os.cpu_count() or 2))
        try:
            self.compiler = subprocess.run([self.cxx, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
        except (OSError, IndexError):
            self.compiler = "unknown"

    # ------------------------------------------------------------ engine --

    def current_hash(self) -> str:
        files = _runtime_sources()
        if not any(rel.startswith("third_party/") for _, rel in files):
            raise BuildError("nlohmann/json.hpp not found — run `cmake -S . -B build` once to fetch it")
        sig = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p, _ in files)
        cached_sig, cached_hash = self._hash_cache
        if cached_sig == sig:
            return cached_hash
        h = hashlib.sha256()
        h.update(" ".join(FLAGS).encode() + b"\n" + self.compiler.encode() + b"\n")
        for p, rel in files:
            h.update(rel.encode() + b"\0" + p.read_bytes() + b"\0")
        digest = h.hexdigest()[:12]
        self._hash_cache = (sig, digest)
        return digest

    def engine_dir(self, ehash: str) -> Path:
        return self.engines / ehash

    def engine_ready(self, ehash: str) -> bool:
        return (self.engine_dir(ehash) / "bin" / ".ok").is_file()

    def binary(self, ehash: str, name: str) -> Path:
        return self.engine_dir(ehash) / "bin" / name

    def ensure_engine(self, ehash: str | None = None, progress=None) -> str:
        """Snapshot (current sources only) and build the runtime for `ehash`. Returns the hash."""
        ehash = ehash or self.current_hash()
        with self._locks["engine:" + ehash]:
            if self.engine_ready(ehash):
                return ehash
            root = self.engine_dir(ehash)
            src = root / "src"
            if not src.is_dir():
                if ehash != self.current_hash():
                    raise BuildError(f"engine {ehash} has no snapshot and the sources have changed since")
                tmp = root / "src.tmp"
                shutil.rmtree(tmp, ignore_errors=True)
                for p, rel in _runtime_sources():
                    (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, tmp / rel)
                os.replace(tmp, src)
            if progress:
                progress(f"Building engine {ehash}…")
            self._build_engine(ehash, root, src)
            return ehash

    def _compile_many(self, jobs, log_lines):
        """jobs: list of (argv, label). Raises BuildError with the compiler output."""
        def run(job):
            argv, label = job
            r = subprocess.run(argv, capture_output=True, text=True)
            return label, argv, r
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.jobs) as ex:
            results = list(ex.map(run, jobs))
        failed = []
        for label, argv, r in results:
            log_lines.append("$ " + " ".join(argv))
            if r.stdout or r.stderr:
                log_lines.append((r.stdout + r.stderr).rstrip())
            if r.returncode != 0:
                failed.append(label)
        if failed:
            raise BuildError("compile failed: " + ", ".join(failed), "\n".join(log_lines))

    def _includes(self, src: Path):
        return ["-I" + str(src / "engine" / "include"), "-I" + str(src / "bots" / "sdk" / "cpp"),
                "-I" + str(src / "lab" / "native"), "-I" + str(src / "third_party")]

    def _build_engine(self, ehash: str, root: Path, src: Path):
        obj = root / "obj"
        bin_dir = root / "bin"
        obj.mkdir(parents=True, exist_ok=True)
        bin_dir.mkdir(parents=True, exist_ok=True)
        define = f'-DPOLYSHARK_ENGINE_HASH="{ehash}"'
        base = [self.cxx] + FLAGS + [define] + self._includes(src)
        engine_cpp = sorted(glob.glob(str(src / "engine" / "src" / "*.cpp")) +
                            glob.glob(str(src / "engine" / "src" / "game_state" / "*.cpp")))
        engine_cpp.append(str(src / "lab" / "native" / "lab_core.cpp"))
        mains = ["referee", "process", "tool", "selfplay"]
        jobs = []
        engine_objs = []
        for cpp in engine_cpp:
            o = obj / (Path(cpp).stem + ".o")
            engine_objs.append(str(o))
            jobs.append((base + ["-c", cpp, "-o", str(o)], Path(cpp).name))
        for m in mains:
            jobs.append((base + ["-c", str(src / "lab" / "native" / f"{m}.cpp"), "-o", str(obj / f"main_{m}.o")], m))
        log = []
        try:
            self._compile_many(jobs, log)
            lib = obj / "libengine.a"
            if lib.exists():
                lib.unlink()
            links = [
                (["ar", "rcs", str(lib)] + engine_objs, "libengine.a"),
            ]
            self._compile_many(links, log)
            extra = ["-lpthread"] + (["-ldl"] if platform.system() == "Linux" else [])
            links = [
                ([self.cxx, str(obj / "main_referee.o"), str(obj / "main_process.o"), str(lib), "-o",
                  str(bin_dir / "polyshark-referee")], "referee"),
                ([self.cxx, str(obj / "main_tool.o"), str(lib), "-o", str(bin_dir / "polyshark-tool")], "tool"),
                ([self.cxx, str(obj / "main_selfplay.o"), str(lib)] + extra + ["-o", str(bin_dir / "polyshark-selfplay")],
                 "selfplay"),
            ]
            self._compile_many(links, log)
        finally:
            (root / "build.log").write_text("\n".join(log) + "\n")
        write_json(bin_dir / ".ok", {"engine": ehash, "compiler": self.compiler, "flags": FLAGS})

    # -------------------------------------------------------------- bots --

    def bot_build_dir(self, key: str) -> Path:
        return self.bot_builds / key

    def build_cpp_bot(self, ehash: str, bot_dir: Path, manifest: dict, source_hash: str) -> dict:
        """Builds a C++ bot (executable + plugin) against engine `ehash`. Cached by content."""
        self.ensure_engine(ehash)
        extra_flags = [str(f) for f in manifest.get("flags", [])]
        key_src = json.dumps([ehash, source_hash, FLAGS, extra_flags, manifest.get("sources"), manifest.get("include")])
        key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
        out = self.bot_build_dir(key)
        info = {"key": key, "exe": str(out / "bot"), "lib": str(out / "bot.so"), "log": str(out / "build.log")}
        with self._locks["bot:" + key]:
            if (out / ".ok").is_file():
                return info
            src = self.engine_dir(ehash) / "src"
            pats = manifest.get("sources") or ["*.cpp"]
            sources = sorted({p for pat in pats for p in glob.glob(str(bot_dir / pat), recursive=True)})
            if not sources:
                raise BuildError(f"no C++ sources match {pats} in {bot_dir}")
            includes = ["-I" + str(bot_dir)] + ["-I" + str(bot_dir / i) for i in manifest.get("include", [])]
            base = ([self.cxx] + FLAGS + [f'-DPOLYSHARK_ENGINE_HASH="{ehash}"'] + self._includes(src)
                    + includes + extra_flags)
            obj = out / "obj"
            obj.mkdir(parents=True, exist_ok=True)
            objs = []
            jobs = []
            for i, cpp in enumerate(sources):
                o = obj / f"{i}_{Path(cpp).stem}.o"
                objs.append(str(o))
                jobs.append((base + ["-c", cpp, "-o", str(o)], os.path.relpath(cpp, bot_dir)))
            lib = str(self.engine_dir(ehash) / "obj" / "libengine.a")
            log = []
            try:
                self._compile_many(jobs, log)
                self._compile_many([
                    ([self.cxx] + objs + [lib] + extra_flags + ["-o", str(out / "bot")], "executable"),
                    ([self.cxx, "-shared"] + objs + [lib] + extra_flags + ["-o", str(out / "bot.so")], "plugin"),
                ], log)
            finally:
                (out / "build.log").write_text("\n".join(log) + "\n")
            write_json(out / ".ok", {"engine": ehash, "source_hash": source_hash})
        return info

    # -------------------------------------------------------------- tool --

    def tool(self, ehash: str | None = None) -> ToolPool:
        ehash = self.ensure_engine(ehash)
        with self._lock:
            pool = self._pools.get(ehash)
            if pool is None:
                pool = self._pools[ehash] = ToolPool(self.binary(ehash, "polyshark-tool"))
            return pool

    def call(self, req: dict, ehash: str | None = None) -> dict:
        reply = self.tool(ehash).call(req)
        if not reply.get("ok"):
            raise ValueError(reply.get("error", "tool error"))
        return reply

    def known_engines(self):
        out = []
        if self.engines.is_dir():
            for d in sorted(self.engines.iterdir()):
                ok = read_json(d / "bin" / ".ok")
                if ok:
                    out.append({"hash": d.name, "compiler": ok.get("compiler")})
        return out


def python_executable(manifest: dict) -> str:
    p = manifest.get("python")
    if p:
        cand = Path(p)
        if not cand.is_absolute():
            cand = REPO / p
        return str(cand)
    return sys.executable


__all__ = ["Native", "BuildError", "python_executable", "sha256_files"]
