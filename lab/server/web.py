"""HTTP server for the Match Lab website and its JSON API."""
from __future__ import annotations

import gzip
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import traceback
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .analysis import Analyst
from .catalog import BUILTINS, Catalog
from .matches import MatchLab, derive_seed
from .native import BuildError, Native
from .selfplay import SelfPlay
from .util import LAB, POSITION_DIRS, REPO, SPRITES, WEB, bot_files, now, read_json, sha256_files, write_json

FAVICON = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M8 1l7 7-7 7-7-7z" fill="#9bdeb3"/></svg>'
PAGES = {"/": "index.html", "/lab/": "lab.html", "/position/": "position.html", "/selfplay/": "selfplay.html",
         "/bots/": "bots.html"}


class Lab:
    """Everything the handlers need, created once per server."""

    def __init__(self, data: Path, max_workers: int):
        self.data = data
        self.token = secrets.token_hex(16)
        self.native = Native(data)
        self.catalog = Catalog(data, self.native)
        self.matches = MatchLab(data, self.native, self.catalog, max_workers)
        self.selfplay = SelfPlay(data, self.native, self.catalog)
        self.analyst = Analyst(data, self.native, self.catalog)
        self.starred_path = data / "positions" / "starred.json"
        self.engine_state = {"building": False, "error": None}
        self._timeline_cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        threading.Thread(target=self.build_engine, daemon=True).start()

    # ------------------------------------------------------------- engine --

    def build_engine(self):
        with self._lock:
            if self.engine_state["building"]:
                return
            self.engine_state.update(building=True, error=None)
        try:
            self.native.ensure_engine()
        except Exception as e:
            self.engine_state["error"] = str(e)
            self.engine_state["log"] = getattr(e, "log", "")[-20000:]
        finally:
            self.engine_state["building"] = False

    def engine_info(self):
        try:
            h = self.native.current_hash()
            ready = self.native.engine_ready(h)
        except BuildError as e:
            h, ready = None, False
            self.engine_state["error"] = str(e)
        return {"hash": h, "ready": ready, "building": self.engine_state["building"],
                "error": self.engine_state.get("error"), "log": self.engine_state.get("log"),
                "compiler": self.native.compiler, "known": self.native.known_engines()}

    # ------------------------------------------------------------ replays --

    def replay_source(self, kind: str, rid: str, n: int):
        if kind == "matches":
            p = self.matches.game_paths(rid, n)
            return p["replay"], p["engine"], {"names": p["names"], "map": p["map"], "game": p["game"]}
        path, engine = self.selfplay.replay_path(rid, n)
        return path, engine, {}

    def timeline(self, path: Path, engine: str):
        key = (str(path), path.stat().st_mtime_ns)
        with self._lock:
            if key in self._timeline_cache:
                self._timeline_cache.move_to_end(key)
                return self._timeline_cache[key]
        replay = read_json(path)
        tl = self.native.call({"cmd": "timeline", "replay": replay}, engine)
        out = {"replay": {k: v for k, v in replay.items() if k != "actions"}, "timeline": tl}
        with self._lock:
            self._timeline_cache[key] = out
            while len(self._timeline_cache) > 16:
                self._timeline_cache.popitem(last=False)
        return out

    def frames(self, path: Path, engine: str, start: int, count: int):
        replay = read_json(path)
        return self.native.call({"cmd": "replay", "replay": replay, "from": start, "count": min(count, 256)}, engine)

    # ---------------------------------------------------------- positions --

    def positions(self):
        maps = [m for m in self.catalog.maps() if m["kind"] == "state"]
        starred = read_json(self.starred_path, []) or []
        return {"files": maps, "starred": [{k: v for k, v in s.items() if k != "state"} for s in starred]}

    def starred(self, sid: str):
        for s in read_json(self.starred_path, []) or []:
            if s["id"] == sid:
                return s
        raise KeyError(sid)

    def star_position(self, body):
        starred = read_json(self.starred_path, []) or []
        entry = {"id": secrets.token_hex(5), "name": (body.get("name") or "").strip() or "Untitled position",
                 "note": body.get("note", ""), "source": body.get("source"), "state": body["state"],
                 "turn": body["state"].get("turn"), "player": body["state"].get("current_player"),
                 "map_size": body["state"].get("map_size"), "created_at": now()}
        starred.insert(0, entry)
        write_json(self.starred_path, starred)
        return {"id": entry["id"]}

    def unstar_position(self, sid):
        starred = [s for s in read_json(self.starred_path, []) or [] if s["id"] != sid]
        write_json(self.starred_path, starred)

    def save_position(self, body):
        name = re.sub(r"[^A-Za-z0-9_.-]+", "-", (body.get("name") or "").strip()).strip("-.")
        if not name:
            raise ValueError("give the position a file name")
        folder = REPO / "data" / "saved_gamestates"
        path = folder / (name if name.endswith(".json") else name + ".json")
        if path.exists():
            raise ValueError(f"{path.relative_to(REPO)} already exists")
        folder.mkdir(parents=True, exist_ok=True)
        # Same pretty format as the visualizer's SAVE button, so it loads there too.
        path.write_text(json.dumps(body["state"], indent=2, sort_keys=True))
        return {"path": str(path.relative_to(REPO)), "map_id": "file:" + str(path.relative_to(REPO))}

    def load_position(self, body):
        src = body.get("source") or {}
        kind = src.get("kind")
        if kind == "map":
            start = self.catalog.map_start(src["map"])
            if start["kind"] == "generated":
                start["seed"] = int(src.get("seed") or 1)
            return self.native.call({"cmd": "new", "start": start})
        if kind == "starred":
            return self.native.call({"cmd": "inspect", "state": self.starred(src["id"])["state"]})
        if kind in ("matches", "selfplay"):
            path, engine, meta = self.replay_source(kind, src["id"], int(src["game"]))
            replay = read_json(path)
            step = max(0, min(int(src.get("step", 0)), len(replay["actions"])))
            states = self.native.call({"cmd": "replay", "replay": replay, "from": step, "count": 1}, engine)["states"]
            out = self.native.call({"cmd": "inspect", "state": states[0], "turn_limit": replay.get("turn_limit", 0),
                                    "start_turn": 0}, engine)
            out["context"] = {"kind": kind, "id": src["id"], "game": int(src["game"]), "step": step,
                              "steps": len(replay["actions"]), "players": replay.get("players"),
                              "map": meta.get("map"), "turn_limit": replay.get("turn_limit")}
            return out
        if kind == "json":
            return self.native.call({"cmd": "inspect", "state": src["state"]})
        raise ValueError("unknown position source")

    # ---------------------------------------------------------- validate --

    def validate_bot(self, bot_id: str):
        """A three-turn game of the bot against itself: does it speak the protocol?"""
        bot = self.catalog.bot(bot_id)
        if bot is None or bot["language"] == "builtin":
            raise ValueError(f"unknown bot {bot_id}")
        ehash = self.native.ensure_engine()
        src = Path(bot["dir"])
        h = sha256_files(src, bot_files(src))[:16]
        root = self.data / "validate" / h
        frozen = root / "bot"
        if not frozen.is_dir():
            self.catalog.freeze(bot, frozen)
        t0 = time.time()
        try:
            launch = self.catalog.prepare(bot, frozen, ehash, h)
        except BuildError as e:
            return {"ok": False, "stage": "build", "error": str(e), "log": e.log[-20000:]}
        build_s = time.time() - t0
        spec = {"game_id": "validate", "start": {"kind": "generated", "generator": "drylands", "size": 9, "seed": 1},
                "turn_limit": 3, "observation": "fog", "timeouts": {"init_ms": 30000, "move_ms": 10000},
                "players": [{"name": bot["name"], "cmd": launch["cmd"], "cwd": launch["cwd"], "env": launch.get("env", {}),
                             "seed": derive_seed(1, 0, s + 1)} for s in range(2)],
                "logs": [str(root / "p0.log"), str(root / "p1.log")], "replay": str(root / "replay.json"),
                "result": str(root / "result.json")}
        write_json(root / "spec.json", spec, indent=1)
        r = subprocess.run([str(self.native.binary(ehash, "polyshark-referee")), str(root / "spec.json")],
                           capture_output=True, text=True, timeout=600)
        result = read_json(root / "result.json") or {"status": "error", "error": r.stderr[-2000:]}
        logs = [(root / f"p{p}.log").read_text(errors="replace")[-6000:] if (root / f"p{p}.log").exists() else ""
                for p in range(2)]
        ok = result.get("status") == "completed" and result.get("reason") != "forfeit"
        return {"ok": ok, "stage": "game", "result": result, "logs": logs, "build_s": build_s,
                "build": launch.get("build"), "build_log": Path(launch["build_log"]).read_text()[-20000:]
                if launch.get("build_log") and Path(launch["build_log"]).exists() else ""}


def make_handler(lab: Lab):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PolysharkLab/1"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet
            pass

        # ------------------------------------------------------ helpers --

        def _send(self, code, body: bytes, ctype: str, extra=None):
            if len(body) > 1024 and "gzip" in (self.headers.get("Accept-Encoding") or "") and \
                    (ctype.startswith("application/json") or ctype.startswith("text/")):
                body = gzip.compress(body, 5)
                extra = dict(extra or {}, **{"Content-Encoding": "gzip"})
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code=200, extra=None, exact=False):
            body = obj if exact else _js_safe(obj)
            self._send(code, json.dumps(body, separators=(",", ":"), default=str).encode(), "application/json", extra)

        def _error(self, code, msg, log=None):
            self._json({"error": msg, **({"log": log} if log else {})}, code)

        def _file(self, path: Path, download: str | None = None):
            if not path.is_file():
                return self._error(404, "not found")
            ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
                ctype += "; charset=utf-8"
            extra = {"Content-Disposition": f'attachment; filename="{download}"'} if download else None
            data = path.read_bytes()
            if download is None and path.suffix in (".png", ".svg"):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(data)
                return
            self._send(200, data, ctype, extra)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        # ---------------------------------------------------------- GET --

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            url = urlparse(self.path)
            path, q = url.path, {k: v[-1] for k, v in parse_qs(url.query).items()}
            try:
                if path in PAGES:
                    html = (WEB / PAGES[path]).read_text().replace("__LAB_TOKEN__", lab.token)
                    return self._send(200, html.encode(), "text/html; charset=utf-8")
                if path == "/favicon.ico":
                    return self._send(200, FAVICON, "image/svg+xml")
                if path in ("/lab", "/position", "/selfplay", "/bots"):
                    self.send_response(301)
                    self.send_header("Location", path + "/")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if path.startswith("/assets/"):
                    return self._file(_safe(WEB / "assets", path[len("/assets/"):]))
                if path.startswith("/sprites/"):
                    return self._file(_safe(SPRITES, unquote(path[len("/sprites/"):])))
                if path.startswith("/api/"):
                    return self._api_get(path[5:], q)
                return self._error(404, "not found")
            except KeyError as e:
                self._error(404, str(e))
            except (ValueError, BuildError) as e:
                self._error(400, str(e), getattr(e, "log", None))
            except Exception as e:
                traceback.print_exc()
                self._error(500, f"{type(e).__name__}: {e}")

        def _api_get(self, p: str, q: dict):
            parts = [unquote(x) for x in p.strip("/").split("/")]
            if parts == ["catalog"]:
                return self._json({"bots": lab.catalog.public_bots(), "builtins": BUILTINS, "maps": lab.catalog.maps(),
                                   "engine": lab.engine_info(), "max_workers": lab.matches.max_workers,
                                   "cpus": os.cpu_count()})
            if parts == ["engine"]:
                return self._json(lab.engine_info())
            if parts == ["rules"]:
                return self._json(lab.native.call({"cmd": "rules"})["rules"])
            if parts == ["protocol"]:
                return self._send(200, (LAB / "PROTOCOL.md").read_bytes(), "text/markdown; charset=utf-8")
            if parts == ["overview"]:
                return self._json(lab.matches.overview())
            if parts == ["positions"]:
                return self._json(lab.positions())
            if parts == ["selfplay"]:
                return self._json({"runs": lab.selfplay.list(), "cpus": os.cpu_count()})
            if len(parts) == 2 and parts[0] == "matches":
                view = q.get("view", "results")
                kw = {}
                if view == "games":
                    kw = {"offset": int(q.get("offset", 0)), "search": q.get("search", ""), "map_filter": q.get("map", ""),
                          "bot": q.get("bot", ""), "unfinished": q.get("unfinished") == "1"}
                return self._json(lab.matches.detail(parts[1], view, **kw))
            if len(parts) == 2 and parts[0] == "selfplay":
                return self._json(lab.selfplay.detail(parts[1], q.get("offset", 0), q.get("limit", 100), q.get("filter", "")))
            if len(parts) >= 5 and parts[0] in ("matches", "selfplay") and parts[2] == "games":
                kind, rid, n, what = parts[0], parts[1], int(parts[3]), parts[4]
                if what == "log" and kind == "matches":
                    logs = lab.matches.game_paths(rid, n)["logs"]
                    p = logs[int(parts[5]) if len(parts) > 5 else 0]
                    text = p.read_text(errors="replace") if p.exists() else "(no output)"
                    return self._send(200, text.encode(), "text/plain; charset=utf-8")
                if what == "result" and kind == "matches":
                    return self._json(read_json(lab.matches.game_paths(rid, n)["result"]) or {})
                path, engine, meta = lab.replay_source(kind, rid, n)
                if not path.is_file():
                    return self._error(404, "this game has no saved replay")
                if what == "replay":
                    return self._file(path, f"polyshark-{kind}-{rid}-game{n}.json" if q.get("download") else None)
                if what == "timeline":
                    out = lab.timeline(path, engine)
                    return self._json({**out, "meta": meta})
                if what == "frames":
                    return self._json(lab.frames(path, engine, int(q.get("from", 0)), int(q.get("count", 64))), exact=True)
                if what == "visualizer":
                    replay = read_json(path)
                    start = replay["start"]
                    if start.get("kind") != "generated":
                        return self._error(400, "the raylib visualizer only replays generated maps")
                    w = replay["result"].get("winner")
                    lines = [f"seed {start['seed']} {start['size']}",
                             f"outcome {1 if w == 0 else -1 if w == 1 else 0} {1 if w == 1 else -1 if w == 0 else 0}"]
                    lines += [" ".join(str(x) for x in a) for a in replay["actions"]]
                    return self._send(200, ("\n".join(lines) + "\n").encode(), "text/plain; charset=utf-8",
                                      {"Content-Disposition": f'attachment; filename="{kind}-{rid}-game{n}.replay"'})
            if len(parts) == 3 and parts[0] == "positions" and parts[1] == "starred":
                return self._json(lab.starred(parts[2]))
            return self._error(404, "unknown endpoint")

        # --------------------------------------------------------- POST --

        def do_POST(self):
            if self.headers.get("X-Lab-Token") != lab.token:
                return self._error(403, "missing or stale lab token — reload the page")
            url = urlparse(self.path)
            parts = [unquote(x) for x in url.path[5:].strip("/").split("/")] if url.path.startswith("/api/") else []
            try:
                if parts == ["bots", "upload"]:
                    name = self.headers.get("X-Filename") or "upload.zip"
                    if int(self.headers.get("Content-Length") or 0) > 200 << 20:
                        raise ValueError("ZIP larger than 200 MiB")
                    raw = self._body()
                    return self._json({"ids": lab.catalog.upload_zip(name, raw)})
                body = json.loads(self._body() or b"{}")
                return self._json(self._api_post(parts, body) or {"ok": True})
            except KeyError as e:
                self._error(404, str(e))
            except (ValueError, BuildError, TimeoutError, RuntimeError) as e:
                self._error(400, str(e), getattr(e, "log", None))
            except Exception as e:
                traceback.print_exc()
                self._error(500, f"{type(e).__name__}: {e}")

        def _api_post(self, parts, body):
            c, m, sp = lab.catalog, lab.matches, lab.selfplay
            if parts == ["bots", "star"]:
                return c.set_star(body["id"], body.get("starred", True))
            if parts == ["bots", "rename"]:
                return c.rename(body["id"], body.get("name", ""))
            if parts == ["bots", "remove"]:
                return c.set_removed(body["id"], body.get("removed", True))
            if parts == ["bots", "delete"]:
                return c.delete(body["id"])
            if parts == ["bots", "delete-folder"]:
                return c.delete_folder(body["folder"])
            if parts == ["bots", "load-path"]:
                return {"ids": c.load_path(body["path"])}
            if parts == ["bots", "validate"]:
                return lab.validate_bot(body["id"])
            if parts == ["engine", "build"]:
                threading.Thread(target=lab.build_engine, daemon=True).start()
                return {"ok": True}
            if parts == ["matches"]:
                return {"id": m.create(body)}
            if len(parts) == 3 and parts[0] == "matches":
                mid, action = parts[1], parts[2]
                if action == "cancel":
                    return m.cancel(mid)
                if action == "requeue":
                    return m.requeue(mid)
                if action == "delete":
                    return m.delete(mid)
                if action == "rename":
                    return m.rename(mid, body.get("name", ""))
            if len(parts) == 5 and parts[0] == "matches" and parts[2] == "games" and parts[4] == "cancel":
                return m.cancel_game(parts[1], int(parts[3]))
            if parts == ["selfplay"]:
                return {"id": sp.create(body)}
            if len(parts) == 3 and parts[0] == "selfplay":
                if parts[2] == "stop":
                    return sp.stop(parts[1])
                if parts[2] == "delete":
                    return sp.delete(parts[1])
            if parts == ["position", "load"]:
                return lab.load_position(body)
            if parts == ["position", "inspect"]:
                req = {"cmd": "inspect", "state": body["state"]}
                req.update({k: body[k] for k in ("turn_limit", "start_turn") if k in body})
                return lab.native.call(req)
            if parts == ["position", "apply"]:
                return lab.native.call({"cmd": "apply", "state": body["state"], "action": body["action"]})
            if parts == ["position", "analyse"]:
                return lab.analyst.analyse(body["bot"], body["state"], body.get("budget"), body.get("args"),
                                           body.get("observation", "fog"), float(body.get("timeout_s", 60)))
            if parts == ["position", "star"]:
                return lab.star_position(body)
            if parts == ["position", "unstar"]:
                return lab.unstar_position(body["id"])
            if parts == ["position", "save"]:
                return lab.save_position(body)
            raise KeyError("unknown endpoint")

    return Handler


JS_INT = 2 ** 53


def _js_safe(obj):
    """Integers JavaScript cannot hold exactly (64-bit seeds) become strings."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return str(obj) if abs(obj) >= JS_INT else obj
    if isinstance(obj, dict):
        return {k: _js_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_js_safe(v) for v in obj]
    return obj


def _safe(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if not str(p).startswith(str(root.resolve()) + os.sep):
        raise KeyError("not found")
    return p


def serve(data: Path, host: str, port: int, max_workers: int):
    import fcntl
    data.mkdir(parents=True, exist_ok=True)
    lock = open(data / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"another lab server is using {data}")
    lab = Lab(data, max_workers)
    import signal as _signal

    def _stop(*_):
        raise KeyboardInterrupt
    _signal.signal(_signal.SIGTERM, _stop)
    httpd = ThreadingHTTPServer((host, port), make_handler(lab))
    httpd.daemon_threads = True
    print(f"Polyshark Match Lab on http://{host}:{port}  (data: {data}, workers: {max_workers})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        lab.matches.shutdown()
        lab.selfplay.stop_all()
        lab.analyst.close_all()
        httpd.server_close()
