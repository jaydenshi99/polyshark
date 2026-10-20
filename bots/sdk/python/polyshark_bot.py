"""Polyshark bot SDK for Python (protocol v1) — see lab/PROTOCOL.md.

Subclass Bot, implement act(), and call run(MyBot()). Anything the bot prints
goes to stderr (the lab's game log); stdout carries protocol messages only.
"""
import json
import sys

PROTOCOL_VERSION = 1


class Observation:
    """One decision: the player's view of the game and the playable actions."""

    def __init__(self, msg):
        self.raw = msg
        self.state = msg["obs"]          # serialised GameState with fog applied
        self.legal = msg["legal"]        # authoritative; answer with an index into it
        self.player = msg.get("player", self.state.get("player", 0))
        self.turn = msg.get("turn", self.state.get("turn", 0))
        self.step = msg.get("step", 0)
        self.map_size = self.state["map_size"]

    def find(self, action_type):
        """Index of the first legal action of this type ("EndTurn", "Move", ...), or None."""
        for i, a in enumerate(self.legal):
            if a["type"] == action_type:
                return i
        return None

    def xy(self, tile):
        return tile % self.map_size, tile // self.map_size

    def tile(self, index):
        return self.state["tiles"][index]

    def explored(self, index, player=None):
        words = self.state["players"][self.player if player is None else player]["visible"]
        return bool((words[index // 16] >> (index % 16)) & 1)

    def units(self, owner=None):
        return [u for u in self.state["units"] if owner is None or u["owner"] == owner]

    def cities(self, owner=None):
        return [c for c in self.state["cities"] if owner is None or c["owner"] == owner]


class Decision:
    def __init__(self, index, eval=None, info=None):
        self.index = index
        self.eval = eval    # optional: own evaluation in [-1, 1]
        self.info = info    # optional small JSON-able diagnostics


class Bot:
    name = "bot"
    version = "0"

    def init(self, hello):
        """hello: {"player", "seed", "mode", "match", "args", "rules"}."""

    def act(self, obs):
        """Return an index into obs.legal, or a Decision."""
        raise NotImplementedError

    def analyse(self, obs, budget):
        """Optional. Return {"eval", "moves": [{"index", "score", "visits", "note"}], "pv", "info"} or None."""
        return None

    def end(self, result):
        """Called once with {"winner", "reason", "turns"} before the process exits."""


def run(bot):
    out = sys.stdout
    sys.stdout = sys.stderr  # stray prints must not corrupt the protocol stream

    def send(msg):
        out.write(json.dumps(msg, separators=(",", ":")) + "\n")
        out.flush()

    can_analyse = type(bot).analyse is not Bot.analyse
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        kind = msg.get("type")
        try:
            if kind == "hello":
                bot.init(msg)
                send({"type": "ready", "name": bot.name, "version": bot.version,
                      "protocol": PROTOCOL_VERSION, "capabilities": ["analyse"] if can_analyse else []})
            elif kind == "act":
                d = bot.act(Observation(msg))
                if not isinstance(d, Decision):
                    d = Decision(int(d))
                reply = {"type": "action", "id": msg.get("id"), "index": d.index}
                if d.eval is not None:
                    reply["eval"] = d.eval
                if d.info is not None:
                    reply["info"] = d.info
                send(reply)
            elif kind == "analyse":
                result = bot.analyse(Observation(msg), msg.get("budget", {}))
                reply = {"type": "analysis", "id": msg.get("id")}
                if result is None:
                    reply["unsupported"] = True
                else:
                    reply.update(result)
                send(reply)
            elif kind == "end":
                bot.end(msg)
                return
            elif kind == "quit":
                return
        except Exception as e:  # report and keep the stream alive; the lab decides what it means
            print(f"[psb] {kind} failed: {e!r}", file=sys.stderr)
            send({"type": "error", "id": msg.get("id"), "message": repr(e)})
