"""
Agents, pluggable strategies, and an arena that runs them against each other to produce
self-play training data (and evaluation win rates).

Design:
  - Strategy   : the decision policy. `act(state) -> Decision`. Pluggable — RandomStrategy,
                 MCTSStrategy (wraps mcts.MCTS), and easy to add more (greedy, human, ...).
  - Agent      : a named strategy. Thin wrapper so the arena can talk about "who won".
  - Arena      : seats agents at player 0 / player 1, plays a game action-at-a-time driven
                 by `state.current_player()`, and (optionally) records a training Sample per
                 decision. Outcomes are filled in at game end.

The arena loop is action-at-a-time, not turn-at-a-time: the engine exposes one action per
step and `current_player()` says whose it is, which matches MCTS's turn-local re-search
(one fresh search per real decision state, until end_turn hands over).

Run a demo:
    source .venv/bin/activate
    python ai/mctsnn-v2/src/arena.py
"""

import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../build/bindings"))
import polyshark  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from mcts import MCTS, HeuristicEvaluator, HEURISTIC_VALUE_SCALE  # noqa: E402

_AT = polyshark.ActionType
# Engine emits these but they're not policy actions (debug / out-of-scope building).
_EXCLUDED = {_AT.DebugAddPop, _AT.ConstructBuilding}


def legal_actions(state):
    """Affordable, in-scope legal actions — the shared action space all strategies pick in."""
    acts = [a for a in state.legal_actions() if a.affordable and a.type not in _EXCLUDED]
    if not acts:  # degenerate safety: fall back to the raw affordable set
        acts = [a for a in state.legal_actions() if a.affordable]
    return acts


def _forced_end_turn(state):
    """The end_turn Action if it is the *only* legal action, else None. Such states carry
    no decision: searching them is wasted compute and recording them poisons training —
    the policy target is a forced 100%-end_turn marginal and the value label is a
    duplicate of the turn's other states (see docs/endturn_collapse.md)."""
    acts = legal_actions(state)
    if len(acts) == 1 and acts[0].type == _AT.EndTurn:
        return acts[0]
    return None


# --------------------------------------------------------------------------- pieces

@dataclass
class Decision:
    """What a strategy returns for one decision."""
    action: object                       # concrete engine Action to apply
    targets: Optional[list] = None       # per-stage visit dists (policy targets) or None


class Strategy:
    """A decision policy. Subclass and implement `act`."""

    def act(self, state) -> Decision:
        raise NotImplementedError

    def reset(self):
        """Called at the start of each game (for stateful strategies)."""


class RandomStrategy(Strategy):
    """Uniform over the legal action space. No training targets (value-only samples)."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, state):
        return Decision(self.rng.choice(legal_actions(state)), targets=None)


class MCTSStrategy(Strategy):
    """PUCT search (mcts.MCTS) with a pluggable evaluator. Produces per-stage visit-count
    targets for policy training. Temperature is scheduled: exploratory for the opening,
    greedy afterwards (self-play). Set `add_noise=True` for self-play root exploration."""

    def __init__(self, evaluator, n_sims=100, c_puct=1.5,
                 temperature=1.0, temp_turns=6, add_noise=True):
        self.mcts = MCTS(evaluator, c_puct=c_puct, add_noise=add_noise)
        self.n_sims = n_sims
        self.temperature = temperature
        self.temp_turns = temp_turns

    def act(self, state):
        temp = self.temperature if state.get_turn() < self.temp_turns else 0.0
        action, _root, targets = self.mcts.search(state, self.n_sims, temperature=temp)
        return Decision(action, targets)


@dataclass
class Agent:
    """A named strategy. Plug in any Strategy — this is the battle participant."""
    name: str
    strategy: Strategy

    def act(self, state):
        return self.strategy.act(state)

    def reset(self):
        self.strategy.reset()


# --------------------------------------------------------------------------- data

@dataclass
class Sample:
    """One recorded decision. `outcome` (value target) is filled at game end, from the
    acting player's perspective."""
    state: object                        # GameState at the decision (encode later, per player)
    player: int                          # who acted
    action: object                       # the chosen engine Action (also the replay stream)
    targets: Optional[list]              # policy targets (None for non-search strategies)
    outcome: Optional[float] = None      # +1 win / -1 loss / heuristic on turn-cap
    train_value: bool = True             # value-loss eligible (trainer subsamples per game —
                                         # within-game states share one label; see training.md)


@dataclass
class GameResult:
    winner: int                          # engine player id, or -1 if no capital captured
    turns: int
    steps: int
    reason: str                          # "capital" | "turn_cap" | "step_cap"
    seed: int                            # map seed (for replay reproduction)
    sz: int                              # map size
    history: list                        # ordered applied Actions (the replay stream)
    v_finals: tuple                      # (outcome_p0, outcome_p1) from each player's frame
    samples: list = field(default_factory=list)


@dataclass
class MatchStats:
    games: int
    wins: dict                           # agent name -> win count
    draws: int
    avg_turns: float
    samples: list                        # pooled, outcome-labelled

    def win_rate(self, name):
        return self.wins.get(name, 0) / self.games if self.games else 0.0

    def __str__(self):
        w = "  ".join(f"{k}={v}" for k, v in self.wins.items())
        return (f"{self.games} games | {w} draws={self.draws} | "
                f"avg_turns={self.avg_turns:.1f} | samples={len(self.samples)}")


def make_heuristic_terminal_value(scale=HEURISTIC_VALUE_SCALE):
    """Build a turn-cap value labeller at a given squash scale. Keep this scale equal to the
    search leaf value's scale so labels and search agree (see mcts.py)."""
    def terminal_value(state, player):
        opp = 1 - player
        diff = polyshark.heuristic_score(state, player) - polyshark.heuristic_score(state, opp)
        return math.tanh(scale * diff)
    return terminal_value


def make_winner_terminal_value(dead_zone=1.0):
    """Turn-cap labeller that declares a WINNER: ±1 by the sign of the heuristic margin
    (0 only inside the dead zone). The value head then estimates win probability — which
    moves sharply with position between equal opponents — instead of an expected-margin
    squash that is nearly flat within a turn. Crucially, mutual passing stops being
    label-neutral: any activity edge at the cap banks a full ±1
    (docs/endturn_collapse.md, actionable #1). `dead_zone` is in heuristic points
    (a village capture is ~4; ~1.0 means tech/unit dust doesn't decide a game)."""
    def terminal_value(state, player):
        opp = 1 - player
        diff = polyshark.heuristic_score(state, player) - polyshark.heuristic_score(state, opp)
        if diff > dead_zone:
            return 1.0
        if diff < -dead_zone:
            return -1.0
        return 0.0
    return terminal_value


# Default-scale instance (used by Arena unless a custom fn is passed).
heuristic_terminal_value = make_heuristic_terminal_value()


def write_replay(path, result):
    """Write a GameResult to the engine replay format read by the visualiser (see
    ai/tdl/selfplay.py and visualizer/visualizer.cpp --replay):

        seed <seed> <sz>
        outcome <v_p0> <v_p1>
        <type> <src> <dst> <param> <path_bits> <path_steps>   (one line per action)

    The map is reproduced from `seed`/`sz` via make_random_game, then the action stream is
    replayed. Only valid for games started from a seed (not a custom start_state)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"seed {result.seed} {result.sz}\n")
        f.write(f"outcome {result.v_finals[0]:.6f} {result.v_finals[1]:.6f}\n")
        for a in result.history:
            f.write(f"{int(a.type)} {a.src} {a.dst} {a.param} {a.path_bits} {a.path_steps}\n")
    return path


# --------------------------------------------------------------------------- arena

class Arena:
    """Seats agents at players 0/1 and runs games. `agents[i]` plays engine player `i`."""

    def __init__(self, agents, terminal_value_fn=heuristic_terminal_value):
        assert len(agents) == 2, "1v1 only (Phase 1)"
        self.agents = list(agents)
        self.terminal_value_fn = terminal_value_fn

    def play_game(self, seed=0, start_state=None, max_turns=30, max_steps=4000, collect=True):
        """Play one game to a capital capture, turn cap, or step cap. Returns a GameResult
        with outcome-labelled samples (if `collect`)."""
        state = start_state if start_state is not None else polyshark.make_random_game(seed)
        sz = state.map_size()
        for ag in self.agents:
            ag.reset()

        samples, history, steps, reason = [], [], 0, "capital"
        while True:
            if state.is_terminal():
                reason = "capital"; break
            if state.get_turn() >= max_turns:
                reason = "turn_cap"; break
            if steps >= max_steps:
                reason = "step_cap"; break

            forced = _forced_end_turn(state)
            if forced is not None:
                # No decision to make or learn from: play it agent-free (no search) and
                # record no sample — neither policy nor value trains on forced states.
                history.append(forced)
                state = state.apply_action(forced)
                steps += 1
                continue

            p = state.current_player()
            dec = self.agents[p].act(state)
            if collect:
                samples.append(Sample(state, p, dec.action, dec.targets))
            history.append(dec.action)          # always: the replay stream
            state = state.apply_action(dec.action)
            steps += 1

        winner = state.winner() if reason == "capital" else -1
        v_finals = self._outcomes(state, winner)
        self._label(samples, v_finals)
        return GameResult(winner, state.get_turn(), steps, reason,
                          seed, sz, history, v_finals, samples)

    def _outcomes(self, final_state, winner):
        """(outcome_p0, outcome_p1) — ±1 on a decisive game, heuristic margin on turn-cap."""
        if winner >= 0:
            return (1.0 if winner == 0 else -1.0, 1.0 if winner == 1 else -1.0)
        return (self.terminal_value_fn(final_state, 0), self.terminal_value_fn(final_state, 1))

    def _label(self, samples, v_finals):
        for s in samples:
            s.outcome = v_finals[s.player]

    def play_matches(self, n_games, base_seed=0, swap_sides=True, collect=True, **kw):
        """Play `n_games`, alternating sides (fairness against first-player advantage), and
        aggregate win rates + pooled samples."""
        wins = {a.name: 0 for a in self.agents}
        draws, turns, pooled = 0, [], []

        for i in range(n_games):
            order = self.agents[::-1] if (swap_sides and i % 2) else self.agents
            res = Arena(order, self.terminal_value_fn).play_game(
                seed=base_seed + i, collect=collect, **kw)
            if res.winner >= 0:
                wins[order[res.winner].name] += 1
            else:
                draws += 1
            turns.append(res.turns)
            if collect:
                pooled.extend(res.samples)

        avg = sum(turns) / len(turns) if turns else 0.0
        return MatchStats(n_games, wins, draws, avg, pooled)


# ----------------------------------------------------------------------- demo

def _demo():
    print("=== random vs random (1 game) ===")
    a0 = Agent("rand-A", RandomStrategy(seed=1))
    a1 = Agent("rand-B", RandomStrategy(seed=2))
    res = Arena([a0, a1]).play_game(seed=7, max_turns=40)
    print(f"winner=player{res.winner} reason={res.reason} turns={res.turns} "
          f"steps={res.steps} samples={len(res.samples)}")
    labelled = res.samples[0]
    print(f"sample[0]: player={labelled.player} outcome={labelled.outcome:+.2f} "
          f"targets={'yes' if labelled.targets else 'none'}\n")

    print("=== random-vs-random match (10 games, swapped sides) ===")
    stats = Arena([a0, a1]).play_matches(10, base_seed=100, max_turns=40)
    print(stats, "\n")

    print("=== MCTS(heuristic) vs random (4 games) ===")
    mcts_agent = Agent("mcts", MCTSStrategy(HeuristicEvaluator(), n_sims=40, add_noise=False))
    rand_agent = Agent("rand", RandomStrategy(seed=3))
    stats = Arena([mcts_agent, rand_agent]).play_matches(4, base_seed=200, max_turns=30)
    print(stats)
    print(f"mcts win rate: {stats.win_rate('mcts'):.0%}")
    n_policy = sum(1 for s in stats.samples if s.targets)
    print(f"pooled samples: {len(stats.samples)} ({n_policy} with policy targets)")


if __name__ == "__main__":
    _demo()
