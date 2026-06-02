"""
Heuristics-based Monte Carlo Tree Search for Polytopia (phase-1 bot).

No neural net. Selection uses UCB1 (or PUCT if priors are provided), expansion
adds one child per simulation, and the value of a leaf is the heuristic value
of the rollout's final state. Sign is flipped at every EndTurn during
back-propagation so each node stores the value from the perspective of the
player whose turn it is at that node.

The skeleton is small on purpose — the interesting work is in `heuristics.py`
and (later) a smarter rollout policy. Replace those without touching this
file.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import heuristics


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

UCB_C            = 1.4              # exploration constant (sqrt(2) is the textbook default)
PUCT_C           = 1.5              # used only when `use_puct=True`
MAX_ROLLOUT_LEN  = 200              # cap on actions during a rollout
DEFAULT_SIMS     = 400              # simulations per `search()` call


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    # Parent edge — None at the root. Stored so backprop can walk upward.
    parent: Optional["Node"] = None
    # Action that produced this node from its parent (None at root).
    action: object           = None
    # Player whose turn it is in the *state this node represents*.
    to_move: int             = 0

    # Statistics in the perspective of `to_move`.
    n: int   = 0
    w: float = 0.0

    # Children keyed by Action. `untried` holds actions not yet expanded.
    children: dict           = field(default_factory=dict)
    untried:  list           = field(default_factory=list)

    # Optional hand-rolled prior per action (filled in `_populate_children`
    # when `use_puct` is on).
    priors:   dict           = field(default_factory=dict)

    @property
    def q(self) -> float:
        return self.w / self.n if self.n > 0 else 0.0

    @property
    def is_fully_expanded(self) -> bool:
        return not self.untried


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTS:
    def __init__(
        self,
        *,
        n_simulations: int = DEFAULT_SIMS,
        use_puct:      bool = False,
        seed:          Optional[int] = None,
    ):
        self.n_simulations = n_simulations
        self.use_puct      = use_puct
        self.rng           = random.Random(seed)

    # ---- public ----------------------------------------------------------

    def search(self, root_state) -> object:
        """Run simulations from `root_state` and return the most-visited action."""
        root = Node(to_move=root_state.current_player())
        self._populate_children(root, root_state)

        for _ in range(self.n_simulations):
            node, state = self._select(root, root_state)
            if not state.is_terminal() and not node.is_fully_expanded:
                node, state = self._expand(node, state)
            value = self._simulate(state, perspective=node.to_move)
            self._backprop(node, value)

        return self._best_action(root)

    # ---- internals -------------------------------------------------------

    def _select(self, node: Node, state):
        """Walk down by UCB/PUCT until a leaf or a node with unexpanded actions."""
        while node.is_fully_expanded and node.children:
            action = self._best_child_action(node)
            state  = state.apply_action(action)
            node   = node.children[action]
        return node, state

    def _expand(self, node: Node, state):
        """Pop one untried action, create child, advance state."""
        action = node.untried.pop()
        child_state = state.apply_action(action)
        child = Node(
            parent=node,
            action=action,
            to_move=child_state.current_player(),
        )
        self._populate_children(child, child_state)
        node.children[action] = child
        return child, child_state

    def _simulate(self, state, *, perspective: int) -> float:
        """Heuristic rollout from `state`. Value is in `perspective`'s frame."""
        steps = 0
        while not state.is_terminal() and steps < MAX_ROLLOUT_LEN:
            action = heuristics.rollout_policy(state, self.rng)
            if action is None:
                break
            state = state.apply_action(action)
            steps += 1
        return heuristics.state_value(state, perspective)

    def _backprop(self, node: Node, value: float):
        """
        Walk upward to the root, flipping the value when the player-to-move
        changes between parent and child. Polytopia turns are multi-action,
        so the flip happens exactly across EndTurn edges.
        """
        while node is not None:
            node.n += 1
            node.w += value
            if node.parent is not None and node.parent.to_move != node.to_move:
                value = -value
            node = node.parent

    # ---- helpers ---------------------------------------------------------

    def _populate_children(self, node: Node, state) -> None:
        """Seed `untried` (and priors if PUCT) with the legal actions here."""
        if state.is_terminal():
            return
        actions = state.legal_actions()
        # Drop non-affordable actions if any affordable exist — same heuristic
        # as random_bot.py; keeps the branching factor manageable.
        affordable = [a for a in actions if a.affordable]
        node.untried = list(affordable) if affordable else list(actions)
        self.rng.shuffle(node.untried)

        if self.use_puct:
            raw = {a: heuristics.action_prior(state, a) for a in node.untried}
            total = sum(raw.values()) or 1.0
            node.priors = {a: p / total for a, p in raw.items()}

    def _best_child_action(self, node: Node) -> object:
        return self._puct_pick(node) if self.use_puct else self._ucb1_pick(node)

    def _child_q_in_parent_frame(self, parent: Node, child: Node) -> float:
        """`child.q` is stored from `child.to_move`'s perspective; flip it when
        the edge crosses an EndTurn so the parent compares apples to apples."""
        return child.q if parent.to_move == child.to_move else -child.q

    def _ucb1_pick(self, node: Node) -> object:
        log_N = math.log(max(node.n, 1))
        best_score, best_a = -math.inf, None
        for a, child in node.children.items():
            if child.n == 0:
                return a  # always sample unvisited children first
            q = self._child_q_in_parent_frame(node, child)
            score = q + UCB_C * math.sqrt(log_N / child.n)
            if score > best_score:
                best_score, best_a = score, a
        return best_a

    def _puct_pick(self, node: Node) -> object:
        sqrt_N = math.sqrt(max(node.n, 1))
        best_score, best_a = -math.inf, None
        for a, child in node.children.items():
            p = node.priors.get(a, 1.0)
            q = self._child_q_in_parent_frame(node, child)
            u = PUCT_C * p * sqrt_N / (1 + child.n)
            score = q + u
            if score > best_score:
                best_score, best_a = score, a
        return best_a

    def _best_action(self, root: Node) -> object:
        if not root.children:
            return None
        return max(root.children.items(), key=lambda kv: kv[1].n)[0]


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def search_with_timing(mcts: MCTS, state) -> tuple:
    """Run `search` and return (action, elapsed_seconds)."""
    t0 = time.perf_counter()
    a  = mcts.search(state)
    return a, time.perf_counter() - t0
