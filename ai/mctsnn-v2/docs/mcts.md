# MCTS — Factored, NN-Guided, Turn-Local

The search that turns the network into a player and produces self-play training targets.
It is PUCT-style MCTS over the **factored** action tree (type → entity → target), guided by
the policy head's per-stage priors and the value head's leaf estimates.

Ties together: [policy_head.md](policy_head.md) (priors), [value_head.md](value_head.md)
(leaf value), [core.md](core.md) (the cached trunk), and `FactoredActions` (legal masks).

## Core design decisions

- **Factored tree.** A move is not one edge; it is a path of sub-decisions (type, then
  entity, then target), each a tree level with the policy head supplying priors.
- **Turn-local, single perspective.** The search expands only the current (root) player's
  turn — a chain of actions. It never steps into the opponent's turn: selecting `end_turn`
  evaluates the *pre-`end_turn`* state rather than applying it. So every node is a
  root-player decision and every value is "good for the root player" — **no negamax
  negation anywhere.** Opponent strength is captured by the value net, not by look-ahead.
  *(Tradeoff: no tactical read of the opponent's reply; extendable to multi-turn later.)*
- **Value at completed-action states only.** The value head can fire only at a real state
  (an action fully assembled and applied) — never at a router (partial action). Like any
  MCTS node, a real state is a **leaf only on its first visit**: evaluated once, then
  expanded; later simulations descend *through* it to the next action in the turn. Routers
  are policy-only and never evaluated. "Completed action" marks *which* nodes are evaluable,
  not a permanent stop.
- **Frozen root fog (imperfect information).** Every node is encoded with the **root
  player's visibility**, so tiles unexplored at the root stay unexplored in the encoding no
  matter what simulated moves reveal (see below).
- **One trunk pass per real state.** Its cache serves the value head and all sub-action
  stages — never recompute per sub-decision (amortization).

## Imperfect information — frozen root fog

Applying actions during search reveals fog (a unit moving explores new tiles). The agent
must **not** gain knowledge it wouldn't have when committing to the move, so the NN's view
is frozen to what the root player knew at search start. This **decouples**:

- **engine truth** — the real `GameState` evolves normally (reveals tiles); `legal_actions`
  / `FactoredActions` read it, so the agent can still legally act on reveals as it would in
  the real game.
- **agent knowledge** — the NN encoding (value *and* policy) is gated on the **root
  visibility snapshot**, so evaluations never cheat by seeing hidden tiles.

Concretely, `encode_*` take a `visible` override (root snapshot) and a `me` override (root
player), both fixed for the whole search. Because the search is single-perspective, `me` =
root player at every node — even the pre-`end_turn` state is scored "how good for the root
player."

### Actions are restricted to root-visible tiles

Frozen encoding stops the *net* from seeing hidden tiles, but the engine's `legal_actions`
and dynamics still run on the true (revealed) state. To keep the agent honest, **every
spatial target is masked to the root visibility set**: `move` / `attack` / `capture` /
`harvest` targets must be visible at the root. Applied at the **`FactoredActions` source**
(drop out-of-sight-target actions before bucketing, so the type/entity/tile masks stay
consistent) — equivalent to AND-ing a root-visibility mask into the tile-pointer grid.

This eliminates all three leaks — attacks on mid-search-revealed enemies, move reachability
betraying fogged terrain, dynamics resolving on hidden contents — and keeps the net's input
coherent with its masks (never prior/attack a tile it can't see). Entity/categorical stages
are unaffected (own units/cities/techs are always self-visible).

*Cost:* no blind leaps into fully-fogged tiles — a unit can move to the edge of its vision
but not into an unseen tile (a Rider can't spend full movement leaping into fog).
Exploration still advances the frontier one visible step at a time (per-action re-search
reveals the next ring). Recovering blind leaps would require modeling fog contents (ISMCTS)
— deferred.

## Tree structure

Two node kinds alternate:

```
[real state S] --type--> [type router] --entity--> [entity router] --target--> ... --apply-->
[real state S'] --type--> ...  (next action, same turn) ... --end_turn--> [leaf]
```

- **Real-state node** — a concrete `GameState`. Holds the trunk cache + value. Its outgoing
  levels are the factored decision at S. In **UpgradingCity** phase it instead has one
  level: the 2 upgrade options (modal head); the type stage is skipped.
- **Router node** — a partial action (type chosen, or type+entity chosen). Policy-only:
  holds edge priors, visit counts N, and value sums W — but **no value of its own**. All Q
  it accumulates originates from a completed-action leaf backed up through it.

Edges at a level = the legal choices from `FactoredActions` at S (the stage mask). Only the
schema's stages exist per type (e.g. `harvest` = type→target; `research` = type→categorical;
`end_turn` = terminal at the type level). See the schema table in [policy_head.md](policy_head.md).

## The network interface

Per real-state leaf, in one batched wave (all encodes use the root player's `me` +
frozen root `visible`):

```
cache      = net.trunk(encode(S, me=root_player, visible=root_visible))   # once per real state
value      = net.value(cache.core)         # leaf estimate, root player's frame
type_prior = softmax(net.type_logits(cache.core, fa.type_mask()))
```

Deeper stage priors are computed **lazily**, the first time a router is reached (cheap reads
off the same cache), using the matching `FactoredActions` mask:

```
entity_prior = softmax(net.entity_logits(cache.core, type, cache.unit_tok|city_tok, fa.entity_mask(type)))
tile_prior   = softmax(net.tile_logits(cache.core, type, chosen_entity_emb, cache.feature_map, fa.tile_mask(type, slot)))
# train_unit / research / upgrade: the categorical / modal heads, likewise masked
```

Masks guarantee priors are nonzero only on legal choices, and (by the round-trip invariant)
no legal choice is ever forbidden.

## One simulation

1. **Select** — from the root real state, walk down using PUCT at every router level,
   applying virtual loss to each traversed edge. Completing a type→…→target path yields a
   full action; **apply** it to reach the next real state. Continue selecting/applying
   through already-expanded real states (chaining actions within the turn) until reaching an
   **unexpanded real state** — the leaf.
2. **Evaluate the leaf** (root player's perspective, frozen root fog — three cases):
   - *terminal* (capital captured) → exact `v = +1` (root won) / `-1` (root lost), no trunk.
   - *selected action is `end_turn`* → evaluate the **pre-`end_turn`** state (do not apply it
     into the opponent's turn); that value is the edge's value.
   - *normal new real state* → `v = net.value`; expand it (cache + type priors).
3. **Back up** `v` along the whole path — every sub-action edge and every earlier action
   edge of the turn — incrementing N and adding to W. **No negation** — the search is
   single-perspective, so every node is the root player and `v` keeps its sign throughout.

## Selection — PUCT

At a router, pick the child maximizing:

```
score(a) = Q(a) + c_puct · P(a) · √(ΣN) / (1 + N(a))
Q(a) = W(a) / N(a)   (0 if unvisited)
```

`P` = the stage's policy prior (masked). Virtual loss temporarily subtracts from W (and
bumps N) on selected edges so a batched wave explores diverse paths.

## Batched evaluation (waves)

Collect a wave of `BATCH_SIZE` leaf real-states via repeated selection-with-virtual-loss,
run `net.trunk` on all of them in **one** forward pass (see the latency numbers — batching
is a ~4× per-state win), then expand + back up the whole wave and undo virtual loss.

## Root exploration

- **Dirichlet noise** mixed into the priors of the **root action's stages — all of them,
  not just `type`.** In the factored tree the root decision spans type → entity → target,
  and each stage produces its own training target, so each needs injected exploration:
  `P = (1−ε)P + ε·Dir(α)`, ε≈0.25, applied to a stage router the first time it is expanded
  *while still part of the root action* (before the first `apply`). Deeper actions / later
  turns are **not** noised (same reason flat AZ noises only the root). Scale `α` to each
  stage's legal count (larger for the 8-way `type`, smaller for the up-to-121 `tile`
  stage — AZ scales α inversely with branching). Self-play only.
- **Temperature** on the final visit counts when choosing the played action: τ=1 for the
  first few plies (sample ∝ N), then τ→0 (argmax N).

## Playing a turn (self-play driver)

Per real decision state, **re-search from scratch** (turn-local root), then commit one
action:

```
while not turn_over:
    root = mcts_search(S, n_sims)          # fresh search rooted at S
    record_training_sample(S, root)        # per-stage visit distributions (below)
    action = sample_by_visits(root, temperature)
    S = S.apply(action)                    # may be end_turn -> turn ends
```

Each real state is one training sample; per-action re-search keeps targets fresh and gives
clean per-decision visit distributions.

## Training targets

Per recorded real state:
- **Policy** — for each stage that fired, the **normalized visit counts** of that level's
  edges become the target distribution; loss = sum of the fired stages' cross-entropies
  against the head's softmax. (Type always; entity/target/categorical per schema.)
- **Value** — the eventual **game outcome** `z ∈ {+1, −1}` from that state's current-player
  perspective (MSE against the value head). Turn-cap games without a winner use a
  heuristic/bootstrapped value (gen-0 bootstrap — TBD in selfplay.md).

Sample tuple: `(encoded S, {stage → visit dist}, chosen action, current_player)`, outcome
filled at game end.

## Hyperparameters (starting points)

| Name | Start | Note |
|---|---|---|
| `c_puct` | 1.5–2.5 | exploration constant |
| `n_sims` | 100–400 self-play, more for eval | scale with turn complexity |
| `BATCH_SIZE` (wave) | 8 → larger on GPU | leaf eval batch |
| virtual loss | 1–3 | wave diversity |
| Dirichlet α / ε | 0.3 / 0.25 | root type priors |
| temperature | 1 → 0 after N plies | self-play exploration |

## Implementation note

Correctness first: prototype the tree in **Python** (uses `FactoredActions` + `PolicyHead`
directly, easy to debug, is the oracle), validate the full loop on tiny/turn-capped games,
then port the hot path to **C++** for self-play throughput — per-node Python overhead would
otherwise dominate the ~14 µs engine cost across hundreds of decisions × thousands of sims.
