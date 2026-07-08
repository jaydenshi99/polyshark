# Policy Head — Autoregressive, Factored

A move is factored into an ordered sequence of conditional sub-decisions. The head is a
small library of **decision primitives** composed per action type via a **schema**; each
stage's query is conditioned on the choices made before it (autoregressive). The stages
map onto the branch levels of the factored MCTS tree and supply the priors at each level.

**Phase-gated** (from `state.phase()`, see [board.md](board.md)):
- **Idle** — stage-1 type decision over 8 types, then the chosen type's schema.
- **UpgradingCity** — a true modal: skip the type stage, route straight to the upgrade
  head (engine blocks all other actions while a city is pending).

**Value fires only at completed-action states** (all sub-decisions chosen → action
applied → new real state). Intermediate sub-action nodes are policy-only routers.

## Shared trunk cache (computed once per real state, then reused)

Refactor `PolysharkNet` to expose these instead of only `value`; the stages are cheap
read-outs off the cache — **do not re-run attention/conv per sub-action**.

| Name | Shape | Source |
|---|---|---|
| `core` | `[B, 256]` | shared core rep ([core.md](core.md)) |
| `feature_map` | `[B, 64, 11, 11]` | ResBlock body output, **pre-pool** (64 ch, not 128) |
| `unit_tok` | `[B, Lu, 128]` | post-attention unit tokens |
| `city_tok` | `[B, Lc, 128]` | post-attention city tokens |
| `unit_mask` / `city_mask` | `[B, Lu]` / `[B, Lc]` | True = real entity |

## Decision primitives

`prior` = concat of prior-choice embeddings this schema (type via `Embedding(8, 16)`;
chosen entity via its token row `[128]`). Every primitive masks illegal choices to `-inf`
before softmax.

**entity-pointer** — over `unit_tok` *or* `city_tok` (type-dependent candidate set):
```
q      = MLP_q(concat(core, prior))           # [B, 128]
keys   = Linear_k(tokens)                      # [B, N, 128]
scores = (keys @ q[...,None]).squeeze / √128   # [B, N]
scores = scores.masked_fill(~legal & mask, -inf); softmax   # combine with entity_mask
```

**tile-pointer** — over the 121 tiles via `feature_map` (`feature_map` already carries
entity occupancy, since scatter is fused before the conv):
```
q      = MLP_q(concat(core, prior))            # [B, 64]  (project to conv channels)
scores = einsum('bc,bchw->bhw', q, feature_map).reshape(B, 121)
scores = scores.masked_fill(~legal_tiles, -inf); softmax    # [B, 121]
```

**categorical** — over a fixed enum of size `K`:
```
logits = MLP(concat(core, prior))              # [B, K]
logits = logits.masked_fill(~legal, -inf); softmax
```

## Stage-1 type (Idle phase)

```
type_logits = MLP_type(core)                   # [B, 8]  (Linear(256→128)→GELU→Linear(128→8))
type_logits = masked_fill(~legal_type_mask, -inf); softmax
```

Types & schemas (fields confirmed against [legal_actions.cpp](../../../engine/src/game_state/legal_actions.cpp)):

| Type | Schema | Engine `Action` fields |
|---|---|---|
| move | entity-ptr(units) → tile-ptr(dst) | `from=src tile, to=dst tile` |
| attack | entity-ptr(units) → tile-ptr(target) | `from=atk tile, to=def tile` |
| harvest | tile-ptr(resource tiles) | `from=city_id, to=res tile` — city implied by `tile.border_city_id` |
| capture | tile-ptr(capturable tiles) | `to=tile`; capturing unit implied |
| train | entity-ptr(cities) → categorical(unit type, K=5) | `from=city tile, param=unit type` |
| research | categorical(tech, K=8) | `from=to=-1, param=tech` |
| recover | entity-ptr(units) | `from=unit tile`, no target |
| end_turn | — (terminal at stage 1) | — |

Conditioning: `train`'s unit-type stage conditions on the chosen **city** token; move/
attack's tile stage conditions on `type_emb + chosen-unit token`.

## Modal upgrade head (UpgradingCity phase)

Only one city pends at a time (engine serializes level-ups — verified; guard with an
`assert` of exactly one pending city). Options are always a pair per level bracket.

```
bracket = onehot4(level==2, ==3, ==4, >=5)     # which option pair; raw level is uncapped
logits  = MLP(concat(core, pending_city_token, bracket)) → 2   # masked softmax (both legal)
```

Output index 0/1 = `opt_a`/`opt_b` for that bracket; codec maps back to `CityUpgradeType`
(`L1_WORKSHOP`/`L1_EXPLORER`, `L2_RESOURCES`/`L2_WALLS`, `L3_BORDER_GROWTH`/
`L3_POPULATION_GROWTH`, `L4_PARK`/`L4_SUPERUNIT`).

## Masking (highest-risk — every stage)

All masks derive from bucketing `state.legal_actions()` into the factored tree:
`legal_type` = types present · `legal_unit|type` = `from`s of that type · `legal_target|
type,unit` = `to`s matching `(type, from)`.

- **Field mapping per type** (the codec/mask helper must respect): `from` is a **city tile**
  for train, a **city_id** for harvest/capture/upgrade, and **tile indices** for move/
  attack; `param` carries unit-type (train), tech (research), upgrade (upgrade).
- **Filter `DebugAddPop`** — emitted by the engine but not a policy action. (ConstructBuilding out of scope.)
- **Consistency invariant:** a type is legal iff ≥1 concrete action exists → no stage-2/3
  row is ever fully `-inf` (which would NaN the softmax).
- **Round-trip test:** every completed schema path ⇔ exactly one legal `Action`, both
  directions. This is the correctness gate for the whole head.

## MCTS interface

1. At real state `s`: one trunk pass → cache `(core, feature_map, unit_tok, city_tok, masks)`.
2. Descend type branch with `P_type` priors (or straight to upgrade head if UpgradingCity).
3. Descend entity branch with `P_entity | type`, then tile/categorical branch with
   `P_target | type, entity` — each read cheaply off the cache.
4. Action complete → apply → new real state `s'`. If `s'` is a new leaf, run its trunk,
   fire the value head, back the value up the whole path (all sub-action edges + earlier
   action edges this turn). `end_turn` → `s'` is a forced leaf (turn-local search).

## Training targets

- Each stage: cross-entropy between its softmax and the **normalized MCTS visit counts**
  of that branch level.
- Factored policy loss = sum of the fired stages' cross-entropies (only stages the chosen
  action used).
- Value head trains separately toward the game outcome ([value_head.md](value_head.md)).
