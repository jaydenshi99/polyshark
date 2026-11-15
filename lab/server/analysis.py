"""Position Analyser backend: asks protocol bots to analyse positions."""
from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path

from .catalog import Catalog
from .native import Native
from .util import bot_files, sha256_files

IDLE_S = 300
MAX_PROCS = 6


class ProtocolBot:
    """One warm bot process in analysis mode."""

    def __init__(self, launch: dict, hello: dict, init_timeout: float):
        env = os.environ.copy()
        env.update(launch.get("env") or {})
        self.proc = subprocess.Popen(launch["cmd"], cwd=launch["cwd"], env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
                                     start_new_session=True)
        self.lines: queue.Queue = queue.Queue()
        self.stderr: deque = deque(maxlen=400)
        self.lock = threading.Lock()
        self.next_id = 0
        self.last_used = time.time()
        threading.Thread(target=self._read_out, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()
        self.ready = self._exchange(hello, init_timeout, lambda m: m.get("type") == "ready")

    def _read_out(self):
        for line in self.proc.stdout:
            s = line.strip()
            if s.startswith("{"):
                self.lines.put(s)
            elif s:
                self.stderr.append("[stdout] " + s)
        self.lines.put(None)

    def _read_err(self):
        for line in self.proc.stderr:
            self.stderr.append(line.rstrip("\n"))

    def _exchange(self, msg: dict, timeout: float, accept) -> dict:
        self.proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        while True:
            left = deadline - time.time()
            if left <= 0:
                raise TimeoutError(f"no reply within {timeout:.0f} s")
            try:
                line = self.lines.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError("bot exited")
            reply = json.loads(line)
            if reply.get("type") == "error":
                raise RuntimeError("bot error: " + str(reply.get("message")))
            if accept(reply):
                return reply

    def request(self, msg: dict, timeout: float) -> dict:
        with self.lock:
            self.next_id += 1
            msg = dict(msg, id=self.next_id)
            rid = self.next_id
            self.last_used = time.time()
            try:
                return self._exchange(msg, timeout, lambda m: m.get("id") == rid)
            finally:
                self.last_used = time.time()

    def alive(self) -> bool:
        return self.proc.poll() is None

    def log_tail(self, n=60) -> str:
        return "\n".join(list(self.stderr)[-n:])

    def close(self):
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except OSError:
            pass


class Analyst:
    def __init__(self, data: Path, native: Native, catalog: Catalog):
        self.dir = data / "analysis"
        self.native = native
        self.catalog = catalog
        self.procs: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.rules = {}
        threading.Thread(target=self._reaper, daemon=True).start()

    def _reaper(self):
        while True:
            time.sleep(30)
            with self.lock:
                for key in [k for k, p in self.procs.items() if time.time() - p.last_used > IDLE_S or not p.alive()]:
                    self.procs.pop(key).close()

    def _rules(self, ehash):
        if ehash not in self.rules:
            self.rules[ehash] = self.native.call({"cmd": "rules"}, ehash)["rules"]
        return self.rules[ehash]

    def _launch(self, bot: dict, ehash: str):
        """Freezes the bot's current source by content hash and builds it."""
        src = Path(bot["dir"])
        files = bot_files(src)
        h = sha256_files(src, files)[:16]
        dest = self.dir / "bots" / h
        if not dest.is_dir():
            tmp = self.dir / "bots" / f"{h}.{threading.get_ident()}.tmp"
            self.catalog.freeze(bot, tmp)
            try:
                os.replace(tmp, dest)
            except OSError:  # another request froze the same source first
                shutil.rmtree(tmp, ignore_errors=True)
        return h, self.catalog.prepare(bot, dest, ehash, h)

    def close_all(self):
        with self.lock:
            for p in self.procs.values():
                p.close()
            self.procs.clear()

    def analyse(self, bot_id: str, state: dict, budget: dict | None = None, args: dict | None = None,
                observation: str = "fog", timeout_s: float = 60.0, ehash: str | None = None) -> dict:
        bot = self.catalog.bot(bot_id)
        if bot is None or bot.get("language") == "builtin":
            raise ValueError(f"unknown bot {bot_id}")
        ehash = self.native.ensure_engine(ehash)
        player = state.get("current_player", 0)
        obs = self.native.call({"cmd": "observe", "state": state, "player": player, "observation": observation}, ehash)
        if not obs["legal"]:
            raise ValueError("no legal actions in this position")
        src_hash, launch = self._launch(bot, ehash)
        budget = budget or {}
        args = args or {}
        key = (bot_id, src_hash, ehash, player, json.dumps(args, sort_keys=True), observation)
        hello = {"type": "hello", "protocol": 1, "player": player, "seed": 1, "mode": "analysis",
                 "match": {"start": {"kind": "state", "size": state.get("map_size")}, "turn_limit": 0,
                           "observation": observation, "timeouts": {"init_ms": 30000, "move_ms": int(timeout_s * 1000)},
                           "budget": budget, "start_turn": state.get("turn", 0), "first_player": player},
                 "args": args, "rules": self._rules(ehash)}
        with self.lock:
            proc = self.procs.get(key)
            if proc is not None and not proc.alive():
                self.procs.pop(key).close()
                proc = None
        if proc is None:
            proc = ProtocolBot(launch, hello, 30.0)
            with self.lock:
                self.procs[key] = proc
                while len(self.procs) > MAX_PROCS:
                    self.procs.popitem(last=False)[1].close()
        msg = {"player": player, "turn": state.get("turn", 0), "step": 0, "obs": obs["obs"], "legal": obs["legal"]}
        t0 = time.time()
        mode = "analysis"
        reply = proc.request(dict(msg, type="analyse", budget=budget), timeout_s)
        if reply.get("unsupported") or reply.get("type") != "analysis":
            mode = "choice"
            reply = proc.request(dict(msg, type="act"), timeout_s)
        elapsed = (time.time() - t0) * 1000
        real, labels = obs["legal_real"], obs["labels"]

        def entry(i):
            return {"index": i, "label": labels[i], "action": real[i]}

        out = {"bot": bot_id, "bot_name": bot["name"], "mode": mode, "player": player, "elapsed_ms": elapsed,
               "source_hash": src_hash, "engine": ehash, "log": proc.log_tail(), "ready": proc.ready}
        if mode == "analysis":
            moves = []
            for mv in reply.get("moves") or []:
                i = mv.get("index")
                if isinstance(i, int) and 0 <= i < len(real):
                    moves.append({**entry(i), **{k: mv.get(k) for k in ("score", "visits", "note")}})
            moves.sort(key=lambda m: (-(m["score"] if isinstance(m.get("score"), (int, float)) else float("-inf")),
                                      -(m.get("visits") or 0)))
            out.update(eval=reply.get("eval"), moves=moves, pv=reply.get("pv"), info=reply.get("info"))
        else:
            i = reply.get("index")
            if not (isinstance(i, int) and 0 <= i < len(real)):
                raise ValueError(f"bot replied with an invalid index {i!r}")
            out.update(choice=entry(i), eval=reply.get("eval"), info=reply.get("info"), moves=[])
        return out
