"""
Basic MCTS prototype — factored, NN-guided, turn-local (see docs/mcts.md).

PUCT search over the factored action tree (type -> entity -> target). It is:
  - turn-local & single-perspective: only the root player's turn is expanded; `end_turn`
    is a forced leaf evaluated at the *pre-end_turn* state. No negamax negation anywhere.
  - value at completed-action states only: routers (partial actions) are policy-only.
  - frozen-root-fog: every node is encoded with the root player's visibility snapshot, so
    search never cheats on tiles the root couldn't see. Spatial targets are masked to that
    snapshot at the FactoredActions source.

This is the correctness-first Python prototype the doc calls for — sequential (no batched
waves / virtual loss yet; those are the C++ port's job) but structurally faithful.

Evaluator is pluggable:
  - NetworkEvaluator  : value head + per-stage policy-head priors (docs-faithful).
  - HeuristicEvaluator: engine heuristic_score value + uniform priors (fast, no torch
    weights needed — good for exercising the tree itself).

Run a demo:
    source .venv/bin/activate
    python ai/mctsnn-v2/src/mcts.py            # heuristic evaluator, plays one turn
    python ai/mctsnn-v2/src/mcts.py --net      # one search with the (random-init) network
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../build/bindings"))
import polyshark  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from factored import FactoredActions  # noqa: E402
from features import visible_snapshot  # noqa: E402
from policy import (  # noqa: E402
    N_TYPES, T_MOVE, T_ATTACK, T_HARVEST, T_CAPTURE, T_TRAIN, T_RESEARCH, T_RECOVER, T_END,
)

# Stages that follow each action type (the type choice itself is stage 0, implicit).
# Mirrors factored.py's path schema: a path key is (type, *these choices).
SCHEMA = {
    T_MOVE:     ("entity", "tile"),
    T_ATTACK:   ("entity", "tile"),
    T_RECOVER:  ("entity",),
    T_HARVEST:  ("tile",),
    T_CAPTURE:  ("tile",),
    T_TRAIN:    ("entity", "train_unit"),
    T_RESEARCH: ("research",),
    T_END:      (),
}
_UNIT_STAGE_TYPES = (T_MOVE, T_ATTACK, T_RECOVER)  # entity stage points at units (else cities)


# --------------------------------------------------------------------------- tree

class Node:
    """A decision point in the factored tree.

    Two flavours share this class:
      - real-state node (`path == ()`): owns a concrete GameState and, once evaluated, a
        scalar `value` (root-player frame). Its stage is `type` (Idle) or `upgrade`
        (UpgradingCity). It is a *leaf* until `expanded` — evaluated once, then expanded.
      - router node (`path != ()`): a partial action within some real state. Policy-only:
        shares that state's evaluation context (`ctx`) and legal helper (`fa`); holds edge
        priors/visits but no value of its own. Expanded on creation.

    A terminal node (`terminal=True`) is a forced leaf: game over, or a chosen `end_turn`
    (its `value` is the pre-end_turn state's value).
    """
    __slots__ = ("state", "ctx", "fa", "path", "stage", "value", "terminal",
                 "expanded", "choices", "P", "N", "W", "children")

    def __init__(self, state=None, path=()):
        self.state = state
        self.ctx = None          # evaluator context (net cache / None), set on expand
        self.fa = None           # FactoredActions for this real state
        self.path = path         # factored choices made so far, from the owning real state
        self.stage = None        # "type" | "entity" | "tile" | "train_unit" | "research" | "upgrade"
        self.value = None        # root-player-frame scalar (real / terminal nodes only)
        self.terminal = False
        self.expanded = False
        self.choices = ()        # legal choice indices at this stage
        self.P = {}              # choice -> prior
        self.N = {}              # choice -> visit count
        self.W = {}              # choice -> value sum
        self.children = {}       # choice -> Node

    def is_real(self):
        return self.path == ()

    def total_N(self):
        return sum(self.N.values())


class MCTS:
    """One configured searcher. Call `search(state, n_sims)` per real decision state."""

    def __init__(self, evaluator, c_puct=1.5, add_noise=False,
                 dirichlet_alpha=0.3, dirichlet_eps=0.25):
        self.ev = evaluator
        self.c_puct = c_puct
        self.add_noise = add_noise
        self.alpha = dirichlet_alpha
        self.eps = dirichlet_eps
        # Frozen root frame, set per search.
        self.root_player = None
        self.root_visible = None
        # The root real state; every stage router sharing it is part of the "root action"
        # and gets Dirichlet noise (identity compare — see _child / _inject_noise).
        self._root_state = None

    # -- public ------------------------------------------------------------

    def search(self, state, n_sims, temperature=0.0):
        """Run `n_sims` simulations rooted at `state`, then pick one action to play.

        Returns (action, root, targets):
          action  : concrete engine Action to apply (may be end_turn).
          root    : the root Node (for inspection / debugging).
          targets : list of (stage_name, {choice: visit_count}) for the chosen action's
                    fired stages — the per-stage policy training targets.
        """
        assert not state.is_terminal(), "search called on a terminal state"
        self.root_player = state.current_player()
        self.root_visible = visible_snapshot(state, self.root_player)
        self.ev.begin_search(self.root_player, self.root_visible)

        root = Node(state=state, path=())
        self._root_state = state
        self._expand_real(root)
        if self.add_noise:
            self._inject_noise(root)           # root's type/upgrade stage

        for _ in range(n_sims):
            self._simulate(root)

        action, targets = self._select_action(root, temperature)
        return action, root, targets

    # -- one simulation ----------------------------------------------------

    def _simulate(self, root):
        node = root
        edges = []  # (node, choice) traversed this sim, for backup
        while True:
            if node.terminal:
                v = node.value
                break
            if not node.expanded:            # real-state leaf: evaluate once
                self._expand_real(node)
                v = node.value
                break
            choice = self._puct(node)
            edges.append((node, choice))
            node = self._child(node, choice)

        # Backup — single perspective, no negation.
        for n, c in edges:
            n.N[c] += 1
            n.W[c] += v

    def _puct(self, node):
        total = node.total_N()
        # +1 under the sqrt so the very first visit follows the priors (cold start).
        sqrt_total = math.sqrt(total + 1)
        best, best_score = None, -math.inf
        for c in node.choices:
            n = node.N[c]
            q = node.W[c] / n if n > 0 else 0.0
            u = self.c_puct * node.P[c] * sqrt_total / (1 + n)
            score = q + u
            if score > best_score:
                best_score, best = score, c
        return best

    def _child(self, node, choice):
        """Realise (and cache) the child reached by taking `choice` at `node`."""
        if choice in node.children:
            return node.children[choice]

        if node.stage == "upgrade":                       # modal: completes immediately
            child = self._apply_child(node, node.fa.upgrade_options[choice])
        else:
            new_path = node.path + (choice,)
            t = new_path[0]
            made = len(new_path) - 1                       # choices after the type
            if made >= len(SCHEMA[t]):                     # action fully assembled
                if t == T_END:
                    child = self._endturn_leaf(node)       # evaluate pre-end_turn state
                else:
                    child = self._apply_child(node, node.fa.action_for(new_path))
            else:                                          # another router stage
                child = Node(state=node.state, path=new_path)
                child.ctx, child.fa = node.ctx, node.fa
                self._init_stage(child, SCHEMA[t][made])
                child.expanded = True                      # routers need no value/eval
                # Root-action stages (entity/target of the first action, still on the root
                # real state) are noised too — not just the type stage. A router belongs to
                # the root action iff it shares the root's GameState (no apply happened yet).
                if self.add_noise and child.state is self._root_state:
                    self._inject_noise(child)

        node.children[choice] = child
        return child

    def _apply_child(self, node, action):
        new_state = node.state.apply_action(action)
        child = Node(state=new_state, path=())
        if new_state.is_terminal():
            child.terminal = True
            child.expanded = True
            w = new_state.winner()
            child.value = 1.0 if w == self.root_player else -1.0
        # else: unexpanded real leaf, evaluated on first visit.
        return child

    def _endturn_leaf(self, node):
        """`end_turn` is turn-local: never applied into the opponent's turn. Its value is
        the current (pre-end_turn) real state's value — already computed on `node`."""
        leaf = Node(state=node.state, path=())
        leaf.terminal = True
        leaf.expanded = True
        leaf.value = node.value
        return leaf

    # -- expansion / staging ----------------------------------------------

    def _expand_real(self, node):
        """Evaluate a real-state leaf (value + eval ctx), build its FactoredActions, and
        open its first decision stage."""
        value, ctx = self.ev.evaluate(node.state)
        node.value = value
        node.ctx = ctx
        node.fa = FactoredActions(
            node.state, encoded=getattr(ctx, "enc", None), root_visible=self.root_visible)
        stage = "upgrade" if node.state.phase() == polyshark.GameStateType.UpgradingCity else "type"
        self._init_stage(node, stage)
        node.expanded = True

    def _init_stage(self, node, stage):
        node.stage = stage
        node.choices = tuple(_legal_choices(stage, node.path, node.fa))
        priors = self.ev.priors(node.ctx, stage, node.path, node.fa, node.choices)
        node.P = {c: float(p) for c, p in zip(node.choices, priors)}
        node.N = {c: 0 for c in node.choices}
        node.W = {c: 0.0 for c in node.choices}

    def _inject_noise(self, node):
        """Mix Dirichlet noise into one stage's priors: P = (1-eps)*P + eps*Dir(alpha).
        Applied to every stage of the root action (type/upgrade, then entity, then target)
        the first time each router is expanded — self-play exploration across the whole
        first decision, not just its type. Deeper actions / later turns are never noised."""
        if not node.choices:
            return
        noise = np.random.dirichlet([self.alpha] * len(node.choices))
        for i, c in enumerate(node.choices):
            node.P[c] = (1 - self.eps) * node.P[c] + self.eps * float(noise[i])

    # -- committing an action ---------------------------------------------

    def _select_action(self, root, temperature):
        """Walk the tree by visit counts to assemble one full action to play, collecting
        the per-stage visit distributions (training targets) along the way."""
        node = root
        path = []
        targets = []
        while True:
            targets.append((node.stage, dict(node.N)))
            choice = _sample_by_visits(node.N, temperature)
            path.append(choice)

            if node.stage == "upgrade":
                return node.fa.upgrade_options[choice], targets
            t = path[0]
            if len(path) - 1 >= len(SCHEMA[t]):            # action complete
                action = node.fa.action_for(tuple(path))
                return action, targets
            node = self._child(node, choice)               # descend into the next router


def _legal_choices(stage, path, fa):
    """Legal choice indices at `stage`, from the FactoredActions masks (single source of
    truth; the evaluator's priors align to this list)."""
    if stage == "type":
        return [i for i, m in enumerate(fa.type_mask()) if m]
    if stage == "upgrade":
        return list(range(len(fa.upgrade_options)))
    if stage == "entity":
        return [i for i, m in enumerate(fa.entity_mask(path[0])) if m]
    if stage == "tile":
        slot = path[1] if path[0] in (T_MOVE, T_ATTACK) else None
        return [i for i, m in enumerate(fa.tile_mask(path[0], slot)) if m]
    if stage == "train_unit":
        return [i for i, m in enumerate(fa.train_unit_mask(path[1])) if m]
    if stage == "research":
        return [i for i, m in enumerate(fa.research_mask()) if m]
    raise ValueError(f"unknown stage {stage}")


def _sample_by_visits(visits, temperature):
    choices = list(visits.keys())
    counts = np.array([visits[c] for c in choices], dtype=np.float64)
    if counts.sum() == 0:                                  # nothing explored -> first legal
        return choices[0]
    if temperature <= 1e-6:
        return choices[int(counts.argmax())]
    p = counts ** (1.0 / temperature)
    p /= p.sum()
    return choices[int(np.random.choice(len(choices), p=p))]


# ------------------------------------------------------------------ evaluators

class HeuristicEvaluator:
    """Value from the engine's gen-0 heuristic (squashed to (-1,1)), uniform priors.
    No torch weights involved — for exercising the search cheaply and deterministically."""

    def __init__(self, scale=0.1):
        self.scale = scale
        self.root_player = None

    def begin_search(self, root_player, root_visible):
        self.root_player = root_player

    def evaluate(self, state):
        me = self.root_player
        opp = 1 - me
        diff = polyshark.heuristic_score(state, me) - polyshark.heuristic_score(state, opp)
        return math.tanh(self.scale * diff), None      # ctx=None -> fa uses default encode

    def priors(self, ctx, stage, path, fa, choices):
        n = len(choices)
        return np.full(n, 1.0 / n, dtype=np.float32) if n else np.zeros(0, dtype=np.float32)


class NetworkEvaluator:
    """Docs-faithful evaluator: PolysharkNet value head + PolicyHead per-stage priors, all
    encoded with the frozen root fog. Slot ordering is kept aligned by feeding the *same*
    `encode_entities` result to both the trunk and FactoredActions (via ctx.enc)."""

    def __init__(self, net=None, policy=None):
        import torch
        from model import PolysharkNet
        from policy import PolicyHead
        self.torch = torch
        self.net = net or PolysharkNet()
        self.policy = policy or PolicyHead()
        self.net.eval()
        self.policy.eval()
        self.root_player = None
        self.root_visible = None

    def begin_search(self, root_player, root_visible):
        self.root_player = root_player
        self.root_visible = root_visible

    class _Ctx:
        __slots__ = ("cache", "enc")

    def _encode(self, state):
        import features
        me, vis = self.root_player, self.root_visible
        enc = features.encode_entities(state, me=me, visible=vis)
        board = features.encode_board(state, me=me, visible=vis)
        glob = features.encode_globals(state, me=me)
        return enc, board, glob

    def evaluate(self, state):
        torch = self.torch
        enc, board, glob = self._encode(state)
        nu = int(enc["unit_types"].shape[0])
        nc = int(enc["city_feats"].shape[0])
        Lu, Lc = max(nu, 1), max(nc, 1)

        def pad(arr, L, width=None):
            shape = (1, L) if width is None else (1, L, width)
            out = np.zeros(shape, dtype=arr.dtype if arr.size else np.float32)
            if arr.shape[0]:
                out[0, :arr.shape[0]] = arr
            return out

        unit_types = np.zeros((1, Lu), dtype=np.int64); unit_types[0, :nu] = enc["unit_types"]
        unit_feats = pad(enc["unit_feats"], Lu, enc["unit_feats"].shape[1])
        unit_tiles = np.zeros((1, Lu), dtype=np.int64); unit_tiles[0, :nu] = enc["unit_tiles"]
        unit_mask = np.zeros((1, Lu), dtype=bool); unit_mask[0, :nu] = True
        city_feats = pad(enc["city_feats"], Lc, enc["city_feats"].shape[1])
        city_tiles = np.zeros((1, Lc), dtype=np.int64); city_tiles[0, :nc] = enc["city_tiles"]
        city_mask = np.zeros((1, Lc), dtype=bool); city_mask[0, :nc] = True

        t = torch.from_numpy
        with torch.no_grad():
            cache = self.net.trunk(
                t(unit_types), t(unit_feats), t(unit_mask), t(unit_tiles),
                t(city_feats), t(city_mask), t(city_tiles),
                t(board[None]), t(glob[None]),
            )
            value = float(self.net.value(cache.core).item())

        ctx = self._Ctx()
        ctx.cache = cache
        ctx.enc = enc
        return value, ctx

    def priors(self, ctx, stage, path, fa, choices):
        torch = self.torch
        if not choices:
            return np.zeros(0, dtype=np.float32)
        core = ctx.cache.core
        with torch.no_grad():
            if stage == "type":
                mask = torch.tensor([fa.type_mask()], dtype=torch.bool)
                logits = self.policy.type_logits(core, mask)
            elif stage == "entity":
                t = path[0]
                toks = ctx.cache.unit_tok if t in _UNIT_STAGE_TYPES else ctx.cache.city_tok
                m = fa.entity_mask(t)
                mask = torch.tensor([m], dtype=torch.bool)
                logits = self.policy.entity_logits(
                    core, torch.tensor([t]), toks[:, :len(m)], mask)
            elif stage == "tile":
                t = path[0]
                if t in (T_MOVE, T_ATTACK):
                    slot = path[1]
                    entity_emb = ctx.cache.unit_tok[:, slot]
                    mask = torch.tensor([fa.tile_mask(t, slot)], dtype=torch.bool)
                else:
                    entity_emb = None
                    mask = torch.tensor([fa.tile_mask(t)], dtype=torch.bool)
                logits = self.policy.tile_logits(
                    core, torch.tensor([t]), entity_emb, ctx.cache.feature_map, mask)
            elif stage == "train_unit":
                slot = path[1]
                city_emb = ctx.cache.city_tok[:, slot]
                mask = torch.tensor([fa.train_unit_mask(slot)], dtype=torch.bool)
                logits = self.policy.train_unit_logits(core, torch.tensor([T_TRAIN]), city_emb, mask)
            elif stage == "research":
                mask = torch.tensor([fa.research_mask()], dtype=torch.bool)
                logits = self.policy.research_logits(core, torch.tensor([T_RESEARCH]), mask)
            elif stage == "upgrade":
                # Modal head needs the pending-city token + level bracket; the basic
                # prototype uses uniform priors here (both options always legal).
                n = len(choices)
                return np.full(n, 1.0 / n, dtype=np.float32)
            else:
                raise ValueError(stage)

            probs = torch.softmax(logits, dim=-1)[0].numpy()
        return np.array([probs[c] for c in choices], dtype=np.float32)


# ----------------------------------------------------------------------- driver

def play_turn(state, mcts, n_sims, temperature=0.0, verbose=False):
    """Turn-local self-play driver: re-search from scratch at each real decision state,
    commit one action, until `end_turn` ends the turn (or the game ends)."""
    samples = []
    while True:
        action, root, targets = mcts.search(state, n_sims, temperature)
        samples.append((state, targets, action))
        if verbose:
            _print_action(state, action, root)
        state = state.apply_action(action)
        if action.type == polyshark.ActionType.EndTurn or state.is_terminal():
            break
    return state, samples


def _print_action(state, action, root):
    at = polyshark.ActionType(action.type)
    root_v = root.value
    top = sorted(root.N.items(), key=lambda kv: -kv[1])[:4]
    print(f"  play {at!s:<28} src={action.src:>3} dst={action.dst:>3} param={action.param:>2}"
          f"   root_v={root_v:+.3f}  top_type_visits={top}")


def _demo(use_net):
    evaluator = NetworkEvaluator() if use_net else HeuristicEvaluator()
    mcts = MCTS(evaluator, c_puct=1.5)
    state = polyshark.make_random_game(7)
    n_sims = 30 if use_net else 100
    kind = "network (random init)" if use_net else "heuristic"

    print(f"MCTS prototype demo — evaluator: {kind}, n_sims={n_sims}")
    print(f"root player = {state.current_player()}, turn = {state.get_turn()}\n")

    if use_net:
        action, root, targets = mcts.search(state, n_sims)
        _print_action(state, action, root)
        print(f"\n  fired stages: {[s for s, _ in targets]}")
    else:
        print("Playing one full turn:")
        _, samples = play_turn(state, mcts, n_sims, verbose=True)
        print(f"\n  {len(samples)} actions committed this turn.")


if __name__ == "__main__":
    _demo(use_net="--net" in sys.argv)
