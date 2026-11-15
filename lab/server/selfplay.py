"""Native self-play runs: in-process C++ bots on many threads, one run at a time."""
from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import threading
import time
import traceback
from pathlib import Path

from .catalog import Catalog
from .native import Native
from .util import now, read_json, write_json


class Run:
    def __init__(self, path: Path, meta: dict):
        self.path = path
        self.meta = meta
        self.state = meta.get("state", "done")
        self.progress = meta.get("final") or {}
        self.history = read_json(path / "history.json", []) or []
        self.proc = None
        self.note = ""

    @property
    def id(self):
        return self.meta["id"]

    def save(self):
        self.meta["state"] = self.state
        write_json(self.path / "run.json", self.meta, indent=1)
        write_json(self.path / "history.json", self.history)


class SelfPlay:
    def __init__(self, data: Path, native: Native, catalog: Catalog):
        self.dir = data / "selfplay"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.native = native
        self.catalog = catalog
        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)
        self.runs: dict[str, Run] = {}
        for d in sorted(self.dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
            meta = read_json(d / "run.json")
            if isinstance(meta, dict):
                r = Run(d, meta)
                if r.state in ("building", "queued", "running"):
                    r.state = "stopped"
                    r.save()
                self.runs[r.id] = r
        threading.Thread(target=self._loop, daemon=True, name="selfplay").start()

    # ------------------------------------------------------------- create --

    def _bot(self, bot_id: str, run_dir: Path, slot: str, ehash: str, args: dict) -> dict:
        b = self.catalog.bot(bot_id)
        if b is None:
            raise ValueError(f"unknown bot {bot_id}")
        if b["language"] == "builtin":
            return {"name": b["name"].split(" ")[0], "builtin": b["builtin"], "args": args, "catalog_id": bot_id}
        if b["language"] != "cpp":
            raise ValueError(f"{b['name']} is a {b['language']} bot. Native self-play loads C++ bots in-process; "
                             "use Match Lab's self-play mode for other languages.")
        frozen = run_dir / "frozen" / slot
        fr = self.catalog.freeze(b, frozen)
        launch = self.catalog.prepare(b, frozen, ehash, fr["hash"])
        return {"name": b["name"], "lib": launch["lib"], "args": args, "catalog_id": bot_id, "hash": fr["hash"],
                "build": launch["build"]}

    def create(self, req: dict) -> str:
        bot_a = req.get("bot_a")
        if not bot_a:
            raise ValueError("pick a bot")
        bot_b = req.get("bot_b") or bot_a
        games = int(req.get("games", 1000))
        if not 1 <= games <= 10_000_000:
            raise ValueError("games must be between 1 and 10,000,000")
        threads = max(1, min(int(req.get("threads") or os.cpu_count() or 1), 256))
        turn_limit = int(req.get("turn_limit", 50))
        if not 1 <= turn_limit <= 5000:
            raise ValueError("turn limit must be between 1 and 5000")
        map_id = req.get("map") or "gen:drylands:9"
        mp = self.catalog.map(map_id)
        if mp is None:
            raise ValueError(f"unknown map {map_id}")
        seed = int(req["seed"]) if str(req.get("seed") or "").strip() else random.getrandbits(63)
        budget = {}
        if req.get("budget_nodes"):
            budget["nodes"] = int(req["budget_nodes"])
        with self.lock:
            number = max([int(k) for k in self.runs if k.isdigit()] or [0]) + 1
            path = self.dir / str(number)
            path.mkdir(parents=True)
            meta = {"id": str(number), "number": number, "name": (req.get("name") or "").strip(),
                    "created_at": now(), "state": "building",
                    "request": {"bot_a": bot_a, "bot_b": bot_b, "map": map_id},
                    "settings": {"games": games, "threads": threads, "turn_limit": turn_limit, "seed": seed,
                                 "observation": "full" if req.get("observation") == "full" else "fog",
                                 "alternate_sides": bool(req.get("alternate_sides", True)),
                                 "replays": max(0, int(req.get("replays") or 0)), "budget": budget,
                                 "map": mp["name"], "map_id": map_id}}
            run = Run(path, meta)
            run.note = "Building…"
            run.save()
            self.runs[run.id] = run
        threading.Thread(target=self._prepare, args=(run, req), daemon=True).start()
        return run.id

    def _prepare(self, run: Run, req: dict):
        try:
            ehash = self.native.ensure_engine(progress=lambda s: setattr(run, "note", s))
            s = run.meta["settings"]
            a = self._bot(run.meta["request"]["bot_a"], run.path, "a", ehash, req.get("args_a") or {})
            same = run.meta["request"]["bot_b"] == run.meta["request"]["bot_a"]
            b = dict(a) if same else self._bot(run.meta["request"]["bot_b"], run.path, "b", ehash, req.get("args_b") or {})
            (run.path / "replays").mkdir(exist_ok=True)
            spec = {"bots": [{k: v for k, v in x.items() if k in ("name", "lib", "builtin", "args")} for x in (a, b)],
                    "games": s["games"], "threads": s["threads"], "start": self.catalog.map_start(s["map_id"]),
                    "seed": s["seed"], "turn_limit": s["turn_limit"], "observation": s["observation"],
                    "alternate_sides": s["alternate_sides"], "budget": s["budget"],
                    "replays": {"dir": str(run.path / "replays"), "first": s["replays"]},
                    "results": str(run.path / "games.jsonl"), "progress_ms": 500}
            write_json(run.path / "spec.json", spec, indent=1)
            with self.lock:
                run.meta.update(engine=ehash, bots=[a, b], mirror=same)
                run.state = "queued"
                run.note = ""
                run.save()
                self.cv.notify_all()
        except Exception as e:
            with self.lock:
                run.state = "error"
                run.meta["error"] = str(e)
                run.meta["error_log"] = getattr(e, "log", "")[-20000:] or traceback.format_exc()[-4000:]
                run.note = ""
                run.save()

    # -------------------------------------------------------------- runner --

    def _loop(self):
        while True:
            with self.cv:
                while True:
                    busy = any(r.state == "running" for r in self.runs.values())
                    queued = [r for r in self.runs.values() if r.state == "queued"]
                    if not busy and queued:
                        run = min(queued, key=lambda r: r.meta["number"])
                        run.state = "running"
                        run.meta["started_at"] = now()
                        run.save()
                        break
                    self.cv.wait(timeout=1.0)
            self._execute(run)

    def _execute(self, run: Run):
        binary = self.native.binary(run.meta["engine"], "polyshark-selfplay")
        err_lines = []
        try:
            proc = subprocess.Popen([str(binary), str(run.path / "spec.json")], stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, bufsize=1)
            with self.lock:
                run.proc = proc
            threading.Thread(target=lambda: err_lines.extend(proc.stderr), daemon=True).start()
            for line in proc.stdout:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                with self.lock:
                    kind = msg.get("type")
                    if kind in ("progress", "done"):
                        run.progress = msg
                        run.history.append([round(msg.get("elapsed_s", 0), 3), msg.get("done", 0),
                                            round(msg.get("games_per_s", 0), 1), round(msg.get("decisions_per_s", 0)),
                                            *msg.get("wins", [0, 0]), msg.get("draws", 0)])
                        if len(run.history) > 4000:
                            run.history = run.history[::2]
                    if kind == "done":
                        run.meta["final"] = msg
                    if kind == "error":
                        run.meta["error"] = msg.get("error")
            proc.wait()
            with self.lock:
                run.proc = None
                final = run.meta.get("final") or {}
                if run.meta.get("error") or proc.returncode not in (0, None) and not final:
                    run.state = "error"
                    run.meta.setdefault("error", "".join(err_lines)[-4000:] or f"exited with {proc.returncode}")
                elif final.get("stopped") or run.meta.get("stop_requested"):
                    run.state = "stopped"
                else:
                    run.state = "done"
                run.meta["finished_at"] = now()
                run.save()
        except Exception as e:
            with self.lock:
                run.state = "error"
                run.meta["error"] = f"{type(e).__name__}: {e}"
                run.save()

    def stop(self, rid: str):
        with self.lock:
            run = self._get(rid)
            run.meta["stop_requested"] = True
            if run.state == "queued" or run.state == "building":
                run.state = "stopped"
                run.save()
            elif run.proc is not None and run.proc.poll() is None:
                run.proc.send_signal(signal.SIGTERM)

    def stop_all(self):
        with self.lock:
            for run in self.runs.values():
                if run.proc is not None and run.proc.poll() is None:
                    run.proc.send_signal(signal.SIGTERM)

    def delete(self, rid: str):
        import shutil
        with self.lock:
            run = self._get(rid)
            if run.state in ("running", "building", "queued"):
                raise ValueError("stop the run before deleting it")
            del self.runs[rid]
        shutil.rmtree(run.path, ignore_errors=True)

    def _get(self, rid: str) -> Run:
        r = self.runs.get(str(rid))
        if r is None:
            raise KeyError(f"no self-play run {rid}")
        return r

    # ------------------------------------------------------------- reading --

    def _summary(self, run: Run) -> dict:
        p = run.progress or {}
        s = run.meta["settings"]
        bots = run.meta.get("bots") or []
        names = [b.get("name") for b in bots] or [run.meta["request"]["bot_a"], run.meta["request"]["bot_b"]]
        end = run.meta.get("finished_at")
        start = run.meta.get("started_at")
        return {"id": run.id, "number": run.meta["number"], "name": run.meta.get("name", ""), "state": run.state,
                "note": run.note, "bots": names, "mirror": run.meta.get("mirror"), "settings": s,
                "done": p.get("done", 0), "games": s["games"], "games_per_s": p.get("games_per_s"),
                "decisions_per_s": p.get("decisions_per_s"), "wins": p.get("wins"), "draws": p.get("draws"),
                "created_at": run.meta["created_at"], "started_at": start, "finished_at": end,
                "elapsed_s": ((end or now()) - start) if start else None, "error": run.meta.get("error"),
                "engine": run.meta.get("engine")}

    def list(self):
        with self.lock:
            return sorted((self._summary(r) for r in self.runs.values()), key=lambda s: -s["number"])

    def detail(self, rid: str, offset=0, limit=100, filt=""):
        with self.lock:
            run = self._get(rid)
            out = self._summary(run)
            out["progress"] = run.progress
            out["history"] = run.history
            out["error_log"] = run.meta.get("error_log")
            out["bots_detail"] = run.meta.get("bots")
        games = []
        path = run.path / "games.jsonl"
        if path.is_file():
            with open(path) as f:
                for line in f:
                    try:
                        games.append(json.loads(line))
                    except ValueError:
                        pass
        if filt:
            games = [g for g in games if filt in (g.get("reason") or "") or filt == str(g.get("winner_bot"))]
        games.sort(key=lambda g: g["game"])
        replays = {int(p.stem) for p in (run.path / "replays").glob("*.json") if p.stem.isdigit()} if (run.path / "replays").is_dir() else set()
        page = games[int(offset):int(offset) + int(limit)]
        for g in page:
            g["replay"] = g["game"] in replays
        out.update(games_list=page, game_total=len(games), game_offset=int(offset))
        return out

    def replay_path(self, rid: str, g: int) -> tuple[Path, str]:
        run = self._get(rid)
        return run.path / "replays" / f"{int(g)}.json", run.meta.get("engine")
