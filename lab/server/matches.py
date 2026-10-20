"""Matches: freezing inputs, scheduling referee games across workers, and results."""
from __future__ import annotations

import os
import random
import shutil
import signal
import subprocess
import threading
import time
import traceback
from collections import deque
from pathlib import Path

from .catalog import Catalog
from .native import BuildError, Native
from .util import elo, now, read_json, wilson, write_json

MASK = (1 << 64) - 1


def mix64(x: int) -> int:
    """Same splitmix64 finaliser as lab_core.cpp's mix64."""
    x = (x + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


def derive_seed(base: int, a: int, b: int = 0) -> int:
    s = mix64(mix64(mix64(base) ^ (a & MASK)) ^ ((b * 0x632BE59BD9B4E019) & MASK))
    return s or 1


MODES = {"round_robin": "Everyone vs everyone", "reference": "Against a reference", "self": "Self-play"}
ACTIVE = ("building", "queued", "running")
RESULT_KEYS = ("status", "winner", "reason", "forfeit", "turns", "actions", "wall_ms", "final_hash", "error")


def _record():
    return {"wins": 0, "draws": 0, "losses": 0, "games": 0}


def _add(rec, outcome):
    rec["games"] += 1
    rec[{1: "wins", 0: "draws", -1: "losses"}[outcome]] += 1


def _finish(rec):
    g = rec["games"]
    rec["score"] = (rec["wins"] + 0.5 * rec["draws"]) / g if g else None
    return rec


class Match:
    def __init__(self, path: Path, meta: dict):
        self.path = path
        self.meta = meta
        self.results: dict[int, dict] = {}
        self.queue: deque = deque()
        self.running: dict[int, subprocess.Popen | None] = {}
        self.cancelled: set[int] = set(meta.get("cancelled_games", []))
        self.state = meta.get("state", "paused")
        self.note = ""
        self.revision = 0
        self.last_started = 0.0
        self.stopping = False
        self.stats_cache = (None, None)

    @property
    def id(self) -> str:
        return self.meta["id"]

    def games(self):
        return self.meta.get("games", [])

    def workers(self) -> int:
        return int(self.meta["settings"].get("workers", 1))

    def status_of(self):
        """game -> status for every game, in one pass (queue membership is O(1) here)."""
        queued = set(self.queue)

        def status(n: int) -> str:
            r = self.results.get(n)
            if r is not None:
                return "completed" if r.get("status") == "completed" else "failed"
            if n in self.running:
                return "running"
            if n in self.cancelled:
                return "cancelled"
            return "queued" if n in queued else "pending"
        return status

    def save(self):
        self.meta["state"] = self.state
        self.meta["cancelled_games"] = sorted(self.cancelled)
        write_json(self.path / "match.json", self.meta, indent=1)


class MatchLab:
    def __init__(self, data: Path, native: Native, catalog: Catalog, max_workers: int):
        self.dir = data / "matches"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.native = native
        self.catalog = catalog
        self.max_workers = max_workers
        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)
        self.matches: dict[str, Match] = {}
        self._load()
        threading.Thread(target=self._dispatch_loop, daemon=True, name="dispatcher").start()

    # ---------------------------------------------------------------- load --

    def _load(self):
        for d in sorted(self.dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
            meta = read_json(d / "match.json")
            if not isinstance(meta, dict):
                continue
            m = Match(d, meta)
            for g in m.games():
                r = read_json(d / "games" / f"{g['n']}.json")
                if isinstance(r, dict):
                    m.results[g["n"]] = self._compact(r)
            if m.state in ACTIVE:
                # The server stopped mid-match: keep completed games, wait for Resume.
                m.state = "paused" if len(m.results) < len(m.games()) else self._final_state(m)
                m.save()
            self.matches[m.id] = m

    @staticmethod
    def _compact(r: dict) -> dict:
        out = {k: r.get(k) for k in RESULT_KEYS}
        out["timing"] = [{k: t.get(k) for k in ("decisions", "mean_ms", "max_ms", "total_ms")} for t in r.get("timing") or []]
        out["summary"] = [{k: s.get(k) for k in ("cities", "units", "stars", "income", "hp")} for s in r.get("summary") or []]
        return out

    def _final_state(self, m: Match) -> str:
        if m.meta.get("cancelled"):
            return "cancelled"
        if any(r.get("status") != "completed" for r in m.results.values()):
            return "errors"
        return "complete" if len(m.results) >= len(m.games()) else "paused"

    # -------------------------------------------------------------- create --

    def create(self, req: dict) -> str:
        mode = req.get("mode", "round_robin")
        if mode not in MODES:
            raise ValueError("unknown mode")
        bot_ids = list(dict.fromkeys(req.get("bots") or []))
        reference = req.get("reference") if mode == "reference" else None
        if reference:
            bot_ids = [b for b in bot_ids if b != reference]
        if mode == "round_robin" and len(bot_ids) < 2:
            raise ValueError("pick at least two bots for everyone vs everyone")
        if mode == "reference" and (not reference or not bot_ids):
            raise ValueError("pick a reference and at least one challenger")
        if mode == "self" and not bot_ids:
            raise ValueError("pick at least one bot")
        all_ids = ([reference] if reference else []) + bot_ids
        bots = []
        for bid in all_ids:
            b = self.catalog.bot(bid)
            if b is None or b.get("language") == "builtin":
                raise ValueError(f"unknown bot {bid}")
            bots.append(b)
        map_ids = list(dict.fromkeys(req.get("maps") or []))
        if not map_ids:
            raise ValueError("pick at least one map")
        maps = []
        for mid in map_ids:
            m = self.catalog.map(mid)
            if m is None:
                raise ValueError(f"unknown map {mid}")
            maps.append({"id": mid, "name": m["name"], "dims": m["dims"], "kind": m["kind"],
                         "start": self.catalog.map_start(mid)})
        pairs = int(req.get("pairs", 1))
        if pairs < 1:
            raise ValueError("pairs must be a positive whole number")
        turn_limit = int(req.get("turn_limit", 50))
        if not 1 <= turn_limit <= 5000:
            raise ValueError("turn limit must be between 1 and 5000")
        workers = max(1, min(int(req.get("workers", 4)), self.max_workers))
        seed = int(req["seed"]) if str(req.get("seed") or "").strip() else random.getrandbits(63)
        budget = {}
        if req.get("budget_nodes"):
            budget["nodes"] = int(req["budget_nodes"])
        if req.get("budget_ms"):
            budget["ms"] = int(req["budget_ms"])
        replay_limit = req.get("replay_limit")
        replay_limit = int(replay_limit) if str(replay_limit or "").strip() else None
        settings = {
            "pairs": pairs, "workers": workers, "turn_limit": turn_limit,
            "observation": "full" if req.get("observation") == "full" else "fog",
            "timeouts": {"init_ms": int(req.get("init_ms", 10000)), "move_ms": int(req.get("move_ms", 10000))},
            "budget": budget, "save_replays": bool(req.get("save_replays", True)), "replay_limit": replay_limit,
            "seed": seed, "max_actions_per_turn": 2000,
        }

        keys = [f"b{i}" for i in range(len(bots))]
        ref_key = keys[0] if reference else None
        challengers = keys[1:] if reference else keys
        if mode == "round_robin":
            matchups = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
        elif mode == "reference":
            matchups = [(ref_key, c) for c in challengers]
        else:
            matchups = [(k, k) for k in keys]
        games = []
        for pair in range(pairs):
            for mi, mp in enumerate(maps):
                map_seed = derive_seed(seed, mi, pair) if mp["kind"] == "generated" else None
                for x, y in matchups:
                    for seats in ((x, y), (y, x)):
                        n = len(games)
                        games.append({"n": n, "map": mi, "pair": pair, "matchup": [x, y], "seats": list(seats),
                                      "map_seed": map_seed,
                                      "seeds": [derive_seed(seed, n, 1), derive_seed(seed, n, 2)],
                                      "replay": settings["save_replays"] and (replay_limit is None or n < replay_limit)})

        with self.lock:
            number = max([int(k) for k in self.matches if k.isdigit()] or [0]) + 1
            path = self.dir / str(number)
            path.mkdir(parents=True)
            meta = {
                "id": str(number), "number": number, "name": (req.get("name") or "").strip(),
                "created_at": now(), "mode": mode, "reference": ref_key,
                "bots": {k: {"catalog_id": b["id"], "name": b["name"], "language": b["language"],
                             "folder": b.get("folder"), "detail": b["detail"], "args": (req.get("args") or {}).get(b["id"], {})}
                         for k, b in zip(keys, bots)},
                "maps": maps, "settings": settings, "games": games, "state": "building",
            }
            m = Match(path, meta)
            m.state = "building"
            m.note = "Freezing bots…"
            m.save()
            self.matches[m.id] = m
        threading.Thread(target=self._prepare, args=(m, bots), daemon=True).start()
        return m.id

    def _prepare(self, m: Match, bots):
        try:
            ehash = self.native.ensure_engine(progress=lambda s: setattr(m, "note", s))
            m.meta["engine"] = ehash
            for (key, info), bot in zip(m.meta["bots"].items(), bots):
                m.note = f"Freezing and building {info['name']}…"
                frozen = m.path / "frozen" / key
                fr = self.catalog.freeze(bot, frozen)
                info.update(hash=fr["hash"], files=fr["files"], bytes=fr["bytes"], frozen=str(frozen),
                            manifest=bot["manifest"])
                info["launch"] = self.catalog.prepare(bot, frozen, ehash, fr["hash"])
            with self.lock:
                if m.meta.get("cancelled"):
                    m.state = "cancelled"
                else:
                    m.state = "queued"
                    m.meta["queued_at"] = now()
                    m.queue.extend(g["n"] for g in m.games())
                m.note = ""
                m.save()
                self.cv.notify_all()
        except Exception as e:  # build errors are shown on the match
            with self.lock:
                m.state = "error"
                m.meta["error"] = str(e)
                m.meta["error_log"] = getattr(e, "log", "")[-20000:] or traceback.format_exc()[-4000:]
                m.note = ""
                m.save()

    # ----------------------------------------------------------- scheduler --

    def _dispatch_loop(self):
        with self.cv:
            while True:
                self._start_ready()
                self.cv.wait(timeout=1.0)

    def _start_ready(self):
        running = sum(len(m.running) for m in self.matches.values())
        while running < self.max_workers:
            ready = [m for m in self.matches.values()
                     if m.state in ("queued", "running") and m.queue and len(m.running) < m.workers()]
            if not ready:
                return
            # Fair share: the match with the fewest running games goes next; ties rotate.
            m = min(ready, key=lambda x: (len(x.running), x.last_started))
            n = m.queue.popleft()
            m.running[n] = None
            m.revision += 1
            m.last_started = time.monotonic()
            if m.state == "queued":
                m.state = "running"
                m.meta.setdefault("started_at", now())
            running += 1
            threading.Thread(target=self._run_game, args=(m, n), daemon=True).start()

    def _spec(self, m: Match, n: int) -> dict:
        g = m.games()[n]
        s = m.meta["settings"]
        start = dict(m.meta["maps"][g["map"]]["start"])
        if start.get("kind") == "generated":
            start["seed"] = g["map_seed"]
        players = []
        for seat, key in enumerate(g["seats"]):
            b = m.meta["bots"][key]
            launch = b["launch"]
            players.append({"name": b["name"], "bot_id": key, "cmd": launch["cmd"], "cwd": launch["cwd"],
                            "env": launch.get("env", {}), "seed": g["seeds"][seat], "args": b.get("args", {})})
        p = m.path
        return {
            "game_id": f"m{m.id}-g{n}", "start": start, "turn_limit": s["turn_limit"],
            "observation": s["observation"], "timeouts": s["timeouts"], "budget": s.get("budget", {}),
            "max_actions_per_turn": s.get("max_actions_per_turn", 2000), "players": players,
            "logs": [str(p / "logs" / f"{n}-p0.log"), str(p / "logs" / f"{n}-p1.log")],
            "replay": str(p / "replays" / f"{n}.json") if g.get("replay") else None,
            "result": str(p / "games" / f"{n}.json"),
        }

    def _run_game(self, m: Match, n: int):
        result = None
        try:
            spec = self._spec(m, n)
            for sub in ("logs", "replays", "games", "specs"):
                (m.path / sub).mkdir(exist_ok=True)
            spec_path = m.path / "specs" / f"{n}.json"
            write_json(spec_path, spec, indent=1)
            referee = self.native.binary(m.meta["engine"], "polyshark-referee")
            proc = subprocess.Popen([str(referee), str(spec_path)], stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, text=True)
            with self.lock:
                m.running[n] = proc
            _, err = proc.communicate()
            result = read_json(m.path / "games" / f"{n}.json")
            if not isinstance(result, dict):
                result = {"status": "error", "error": (err or "").strip()[-2000:] or f"referee exited with {proc.returncode}"}
        except Exception as e:
            result = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        with self.lock:
            m.running.pop(n, None)
            if result.get("status") == "cancelled" or (m.stopping and result.get("status") != "completed"):
                m.cancelled.add(n)
                try:
                    os.unlink(m.path / "games" / f"{n}.json")
                except OSError:
                    pass
            else:
                if result.get("status") != "completed":
                    write_json(m.path / "games" / f"{n}.json", result)
                m.results[n] = self._compact(result)
            m.revision += 1
            if not m.running and not m.queue and m.state in ("queued", "running"):
                m.state = "cancelled" if m.meta.get("cancelled") else self._final_state(m)
                m.meta["finished_at"] = now()
                m.save()
            elif m.stopping and not m.running:
                m.state = "cancelled"
                m.stopping = False
                m.meta["finished_at"] = now()
                m.save()
            self.cv.notify_all()

    # ------------------------------------------------------------- control --

    def cancel(self, mid: str):
        with self.lock:
            m = self._get(mid)
            m.meta["cancelled"] = True
            while m.queue:
                m.cancelled.add(m.queue.popleft())
            if m.state == "building":
                m.save()
                return
            if m.running:
                m.stopping = True
                for proc in m.running.values():
                    if proc is not None and proc.poll() is None:
                        proc.send_signal(signal.SIGTERM)
            else:
                m.state = "cancelled"
                m.meta["finished_at"] = now()
            m.revision += 1
            m.save()

    def cancel_game(self, mid: str, n: int):
        with self.lock:
            m = self._get(mid)
            if n in m.queue:
                m.queue.remove(n)
                m.cancelled.add(n)
            proc = m.running.get(n)
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                m.cancelled.add(n)
            m.revision += 1
            m.save()

    def requeue(self, mid: str):
        """Runs every game without a completed result again (failed, cancelled, unfinished)."""
        with self.lock:
            m = self._get(mid)
            if m.state in ("building", "error"):
                raise ValueError("this match never finished building")
            if m.running:
                raise ValueError("wait for running games to stop first")
            if not m.meta.get("engine") or not self.native.engine_ready(m.meta["engine"]):
                raise ValueError("this match's engine build is missing")
            todo = [g["n"] for g in m.games() if m.results.get(g["n"], {}).get("status") != "completed"]
            for n in todo:
                m.results.pop(n, None)
                m.cancelled.discard(n)
                try:
                    os.unlink(m.path / "games" / f"{n}.json")
                except OSError:
                    pass
            m.queue = deque(todo)
            m.meta["cancelled"] = False
            m.meta.pop("finished_at", None)
            m.state = "queued" if todo else self._final_state(m)
            m.revision += 1
            m.save()
            self.cv.notify_all()

    def shutdown(self):
        """Server exit: stop running referees; their matches resume as paused."""
        with self.lock:
            for m in self.matches.values():
                for proc in m.running.values():
                    if proc is not None and proc.poll() is None:
                        proc.send_signal(signal.SIGTERM)

    def delete(self, mid: str):
        with self.lock:
            m = self._get(mid)
            if m.state in ACTIVE or m.running:
                raise ValueError("cancel the match before deleting it")
            del self.matches[mid]
        shutil.rmtree(m.path, ignore_errors=True)

    def rename(self, mid: str, name: str):
        with self.lock:
            m = self._get(mid)
            m.meta["name"] = name.strip()
            m.save()

    def _get(self, mid: str) -> Match:
        m = self.matches.get(str(mid))
        if m is None:
            raise KeyError(f"no match {mid}")
        return m

    # ------------------------------------------------------------- reading --

    def _avg_game_s(self, m: Match | None = None) -> float | None:
        pool = [m] if m else list(self.matches.values())
        walls = [r["wall_ms"] for x in pool for r in x.results.values() if r.get("wall_ms")]
        walls = walls[-500:]
        return sum(walls) / len(walls) / 1000 if walls else None

    def _summary(self, m: Match) -> dict:
        total = len(m.games())
        done = sum(1 for r in m.results.values() if r.get("status") == "completed")
        failed = sum(1 for r in m.results.values() if r.get("status") != "completed")
        remaining = len(m.queue) + len(m.running)
        avg = self._avg_game_s(m) or self._avg_game_s()
        eta = None
        if m.state in ("queued", "running") and avg is not None:
            slots = max(1, min(m.workers(), remaining))
            eta = (remaining / slots) * avg
        created = m.meta.get("created_at", 0)
        end = m.meta.get("finished_at") if m.state not in ACTIVE else None
        names = [b["name"] for b in m.meta["bots"].values()]
        mode = m.meta["mode"]
        if mode == "reference":
            ref = m.meta["bots"][m.meta["reference"]]["name"]
            title = f"{', '.join(names[1:])} vs {ref}"
        else:
            title = " · ".join(names)
        return {
            "id": m.id, "number": m.meta["number"], "name": m.meta.get("name", ""), "title": title,
            "mode": mode, "mode_label": MODES[mode], "state": m.state, "note": m.note, "stopping": m.stopping,
            "total": total, "completed": done, "failed": failed, "queued": len(m.queue), "running": len(m.running),
            "cancelled": len(m.cancelled), "created_at": created, "finished_at": end,
            "elapsed_s": (end or now()) - created, "eta_s": eta, "workers": m.workers(),
            "maps": [mp["name"] for mp in m.meta["maps"]], "bots": names, "error": m.meta.get("error"),
        }

    def overview(self) -> dict:
        with self.lock:
            summaries = [self._summary(m) for m in self.matches.values()]
            active = [s for s in summaries if s["state"] in ACTIVE]
            queued = sum(s["queued"] for s in active)
            running = sum(s["running"] for s in active)
            completed = sum(s["completed"] for s in active)
            failed = sum(s["failed"] for s in active)
            avg = self._avg_game_s()
            clear = (queued + running) / max(1, min(self.max_workers, queued + running)) * avg if avg and (queued + running) else (0 if not (queued + running) else None)
        summaries.sort(key=lambda s: -s["number"])
        return {
            "stats": {"queued": queued, "running": running, "completed": completed, "failed": failed,
                      "clear_s": clear, "workers": self.max_workers, "busy": running},
            "current": [s for s in summaries if s["state"] in ACTIVE or s["state"] == "paused"],
            "historical": [s for s in summaries if s["state"] not in ACTIVE and s["state"] != "paused"],
            "updated_at": now(),
        }

    def detail(self, mid: str, view: str = "results", **q) -> dict:
        with self.lock:
            m = self._get(mid)
            out = self._summary(m)
            out.update({"settings": m.meta["settings"], "engine": m.meta.get("engine"), "view": view,
                        "bots": {k: {"name": b["name"], "catalog_id": b["catalog_id"], "language": b["language"],
                                     "folder": b.get("folder")} for k, b in m.meta["bots"].items()},
                        "reference": m.meta.get("reference"),
                        "maps": [{"name": mp["name"], "dims": mp["dims"], "kind": mp["kind"]} for mp in m.meta["maps"]],
                        "error_log": m.meta.get("error_log"), "revision": m.revision, "updated_at": now()})
            if view == "results":
                out["results"] = self._results(m)
            elif view == "games":
                out.update(self._games_page(m, **q))
            elif view == "inputs":
                out["inputs"] = self._inputs(m)
            return out

    def _inputs(self, m: Match) -> dict:
        bots = []
        for k, b in m.meta["bots"].items():
            launch = b.get("launch") or {}
            bots.append({"key": k, "name": b["name"], "catalog_id": b["catalog_id"], "language": b["language"],
                         "detail": b["detail"], "hash": b.get("hash"), "files": b.get("files"), "bytes": b.get("bytes"),
                         "frozen": b.get("frozen"), "build": launch.get("build"), "cmd": launch.get("cmd"),
                         "args": b.get("args"), "manifest": b.get("manifest")})
        maps = []
        for mp in m.meta["maps"]:
            st = mp["start"]
            maps.append({"name": mp["name"], "dims": mp["dims"], "kind": mp["kind"], "id": mp["id"],
                         "start": st if st.get("kind") == "generated" else {"kind": "state", "turn": st["state"].get("turn"),
                                                                            "current_player": st["state"].get("current_player")}})
        seeds = sorted({g["map_seed"] for g in m.games() if g.get("map_seed")})
        return {"bots": bots, "maps": maps, "engine": m.meta.get("engine"), "settings": m.meta["settings"],
                "map_seeds": seeds[:50], "map_seed_count": len(seeds), "games": len(m.games()),
                "created_at": m.meta.get("created_at")}

    def _games_page(self, m: Match, offset=0, limit=100, search="", map_filter="", bot="", unfinished=False):
        rows = []
        names = {k: b["name"] for k, b in m.meta["bots"].items()}
        search = (search or "").lower().strip()
        game_status = m.status_of()
        for g in m.games():
            n = g["n"]
            status = game_status(n)
            if not unfinished and status in ("queued", "pending", "running"):
                continue
            if map_filter != "" and str(g["map"]) != str(map_filter):
                continue
            if bot and bot not in g["seats"]:
                continue
            r = m.results.get(n, {})
            mp = m.meta["maps"][g["map"]]
            w = r.get("winner")
            row = {"n": n, "pair": g["pair"], "map": mp["name"], "map_index": g["map"], "dims": mp["dims"],
                   "seats": g["seats"], "names": [names[k] for k in g["seats"]], "status": status,
                   "winner_seat": w, "winner": names[g["seats"][w]] if isinstance(w, int) else None,
                   "reason": r.get("reason"), "forfeit": r.get("forfeit"), "turns": r.get("turns"),
                   "actions": r.get("actions"), "wall_ms": r.get("wall_ms"), "error": r.get("error"),
                   "replay": bool(g.get("replay")) and (m.path / "replays" / f"{n}.json").is_file(),
                   "map_seed": g.get("map_seed"), "timing": r.get("timing")}
            hay = f"{n} {row['map']} {' '.join(row['names'])} pair {g['pair'] + 1} {status} {row['reason'] or ''}".lower()
            if search and search not in hay:
                continue
            rows.append(row)
        rows.sort(key=lambda r: -r["n"])
        offset = max(0, int(offset or 0))
        return {"games": rows[offset:offset + int(limit)], "game_total": len(rows), "game_offset": offset}

    # ------------------------------------------------------------- results --

    def _results(self, m: Match) -> dict:
        cached_rev, cached = m.stats_cache
        if cached_rev == m.revision and cached is not None:
            return cached
        res = compute_results(m)
        m.stats_cache = (m.revision, res)
        return res

    def game_paths(self, mid: str, n: int):
        m = self._get(mid)
        return {"replay": m.path / "replays" / f"{n}.json", "result": m.path / "games" / f"{n}.json",
                "logs": [m.path / "logs" / f"{n}-p0.log", m.path / "logs" / f"{n}-p1.log"],
                "engine": m.meta.get("engine"), "game": m.games()[n],
                "names": [m.meta["bots"][k]["name"] for k in m.games()[n]["seats"]],
                "map": m.meta["maps"][m.games()[n]["map"]]["name"]}

    def bot_launch(self, mid: str, key: str) -> dict:
        return self._get(mid).meta["bots"][key]


def compute_results(m: Match) -> dict:
    """Scores (complete side-swapped pairs only), endings, seats, timing and lengths."""
    meta = m.meta
    mode = meta["mode"]
    keys = list(meta["bots"])
    maps = range(len(meta["maps"]))
    rows = {k: {"id": k, **_record(), "maps": {i: _record() for i in maps}, "opponents": {},
                "map_opponents": {i: {} for i in maps}} for k in keys}
    ref_maps = {i: _record() for i in maps}
    pairs: dict = {}
    for g in m.games():
        pairs.setdefault((g["map"], g["pair"], tuple(g["matchup"])), []).append(g)
    rated = 0
    for (mi, _pair, matchup), games in pairs.items():
        rs = [m.results.get(g["n"]) for g in games]
        if len(games) < 2 or any(r is None or r.get("status") != "completed" for r in rs):
            continue
        rated += 1
        for g, r in zip(games, rs):
            w = r.get("winner")
            if mode == "self":
                k = g["seats"][0]
                out = 0 if w is None else (1 if w == 0 else -1)
                _add(rows[k], out)
                _add(rows[k]["maps"][mi], out)
                continue
            for seat, k in enumerate(g["seats"]):
                opp = g["seats"][1 - seat]
                out = 0 if w is None else (1 if w == seat else -1)
                if mode == "reference" and k == meta["reference"]:
                    _add(ref_maps[mi], out)
                    _add(rows[k], out)
                    _add(rows[k]["maps"][mi], out)
                    rows[k]["opponents"].setdefault(opp, _record())
                    _add(rows[k]["opponents"][opp], out)
                    continue
                _add(rows[k], out)
                _add(rows[k]["maps"][mi], out)
                _add(rows[k]["opponents"].setdefault(opp, _record()), out)
                _add(rows[k]["map_opponents"][mi].setdefault(opp, _record()), out)

    out_rows = []
    for k in keys:
        r = rows[k]
        _finish(r)
        for rec in r["maps"].values():
            _finish(rec)
        for rec in r["opponents"].values():
            _finish(rec)
        for d in r["map_opponents"].values():
            for rec in d.values():
                _finish(rec)
        g = r["games"]
        r["win_rate"] = r["wins"] / g if g else None
        lo, hi = wilson(r["score"], g) if g else (None, None)
        r["ci_low"], r["ci_high"] = lo, hi
        r["elo"] = elo(r["score"]) if g else None
        r["elo_low"], r["elo_high"] = (elo(lo), elo(hi)) if g else (None, None)
        r["significant"] = bool(g and lo is not None and (lo > 0.5 or hi < 0.5))
        r["maps"] = {str(i): rec for i, rec in r["maps"].items()}
        r["map_opponents"] = {str(i): d for i, d in r["map_opponents"].items()}
        if mode == "reference" and k == meta["reference"]:
            r["is_reference"] = True
        out_rows.append(r)

    # Every completed game, from each bot's side.
    endings = {k: {"id": k, "games": 0, "capital_won": 0, "capital_lost": 0, "turn_limit": 0, "forfeit_won": 0,
                   "forfeit_lost": 0, "forfeit_kinds": {}, "turns_won": [], "turns_lost": [], "turns": []} for k in keys}
    per_map = [{"games": 0, "wall_ms": 0.0, "turns": 0, "capital": 0, "turn_limit": 0, "forfeit": 0, "failed": 0, "running": 0,
                "queued": 0} for _ in maps]
    seats = {k: {"id": k, "p0": _record(), "p1": _record()} for k in keys}
    timing = {k: {"id": k, "decisions": 0, "total_ms": 0.0, "max_ms": 0.0, "games": 0} for k in keys}
    overall = {"capital": 0, "turn_limit": 0, "forfeit": 0, "failed": 0}
    game_status = m.status_of()
    lengths = []
    for g in m.games():
        r = m.results.get(g["n"])
        pm = per_map[g["map"]]
        if r is None:
            st = game_status(g["n"])
            if st in ("running", "queued"):
                pm[st] += 1
            continue
        if r.get("status") != "completed":
            overall["failed"] += 1
            pm["failed"] += 1
            continue
        reason, w, turns = r.get("reason"), r.get("winner"), r.get("turns") or 0
        overall[reason] = overall.get(reason, 0) + 1
        pm["games"] += 1
        pm["wall_ms"] += r.get("wall_ms") or 0
        pm["turns"] += turns
        pm[reason] = pm.get(reason, 0) + 1
        lengths.append(turns)
        sides = [0] if mode == "self" else [0, 1]
        for seat in sides:
            k = g["seats"][seat]
            e = endings[k]
            e["games"] += 1
            e["turns"].append(turns)
            won = w == seat
            lost = w is not None and w != seat
            if reason == "capital":
                e["capital_won" if won else "capital_lost"] += 1
            elif reason == "turn_limit":
                e["turn_limit"] += 1
            elif reason == "forfeit":
                e["forfeit_won" if won else "forfeit_lost"] += 1
                if lost:
                    kind = (r.get("forfeit") or {}).get("kind", "?")
                    e["forfeit_kinds"][kind] = e["forfeit_kinds"].get(kind, 0) + 1
            if reason == "capital":
                e["turns_won" if won else "turns_lost"].append(turns)
        for seat, k in enumerate(g["seats"]):
            if mode != "self":
                out = 0 if w is None else (1 if w == seat else -1)
                _add(seats[k][f"p{seat}"], out)
            t = (r.get("timing") or [{}, {}])[seat] if r.get("timing") else {}
            tm = timing[k]
            tm["games"] += 1
            tm["decisions"] += t.get("decisions") or 0
            tm["total_ms"] += t.get("total_ms") or 0
            tm["max_ms"] = max(tm["max_ms"], t.get("max_ms") or 0)
        if mode == "self":
            out = 0 if w is None else (1 if w == 0 else -1)
            _add(seats[g["seats"][0]]["p0"], out)

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    end_rows = []
    for k in keys:
        e = endings[k]
        end_rows.append({**{x: e[x] for x in ("id", "games", "capital_won", "capital_lost", "turn_limit",
                                              "forfeit_won", "forfeit_lost", "forfeit_kinds")},
                         "turns_mean": mean(e["turns"]), "turns_won_mean": mean(e["turns_won"]),
                         "turns_lost_mean": mean(e["turns_lost"])})
    for s in seats.values():
        _finish(s["p0"])
        _finish(s["p1"])
    time_rows = [{"id": k, "mean_ms": (t["total_ms"] / t["decisions"]) if t["decisions"] else None,
                  "max_ms": t["max_ms"], "decisions_per_game": (t["decisions"] / t["games"]) if t["games"] else None,
                  "games": t["games"]} for k, t in timing.items()]
    hist = {}
    for t in lengths:
        hist[t] = hist.get(t, 0) + 1
    for rec in ref_maps.values():
        _finish(rec)
    map_rows = [{"games": pm["games"], "failed": pm["failed"], "running": pm["running"], "queued": pm["queued"],
                 "capital": pm["capital"], "turn_limit": pm["turn_limit"], "forfeit": pm["forfeit"],
                 "wall_ms_mean": pm["wall_ms"] / pm["games"] if pm["games"] else None,
                 "turns_mean": pm["turns"] / pm["games"] if pm["games"] else None} for pm in per_map]
    return {"rows": out_rows, "rated_pairs": rated, "maps": map_rows, "total_pairs": len(pairs), "reference_maps": {str(i): r for i, r in ref_maps.items()},
            "endings": end_rows, "seats": list(seats.values()), "timing": time_rows, "overall": overall,
            "lengths": {"mean": mean(lengths), "hist": sorted(hist.items())},
            "finished": sum(1 for r in m.results.values() if r.get("status") == "completed"),
            "total": len(m.games())}
