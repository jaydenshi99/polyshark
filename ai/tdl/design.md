# TDL — Temporal Difference Learning Agent

## Overview

Train a value network via iterative self-improvement. Each generation produces a better leaf evaluator for MCTS, bootstrapping from a hand-crafted heuristic and improving through self-play.

---

## Core Idea

The NN learns to predict what MCTS *would* evaluate a position as, one full round later (after the opponent has responded). This is TD(0): the target is not a terminal outcome but a bootstrapped estimate from the next generation's MCTS.

---

## Generation Loop

```
Gen 0:  MCTS + heuristic evaluator
          │
          ▼
       self-play games (random seeds, capped at N turns)
          │
          ▼
       collect (state s, MCTS value of s'') pairs
          │
          ▼
       train NN_1 on these pairs
          │
          ▼
Gen 1:  MCTS + NN_1 as leaf evaluator
          │
          ▼
       ... repeat
```

---

## Self-Play Data Collection

### What a training sample is

For each player, collect a training sample at **every gamestate where it is that player's turn**. The target is the MCTS evaluation of the **first gamestate where it is that player's turn again** (i.e. after the opponent has completed their turn).

```
input  : encode(s_i)   for every s_i where current_player == P1
target : V_mcts(s_0')  where s_0' is the next state where current_player == P1
```

All states `s_0 … s_n` during P1's turn share the same target `V_mcts(s_0')`. Since they are all on the path to the same outcome, sharing a target is correct.

`s_0'` is already encoded from P1's perspective (P1's fog, P1's pieces as "mine"), so `V_mcts(s_0')` is used directly as the target — **no negation needed**. This is simpler than the equivalent `-V_mcts(s_m)` formulation using the opponent's last state.

**Fog of war is consistent across all samples**: the encoder always uses `current_player()`, so every `s_i` during P1's turn carries P1's fog perspective regardless of which action step it is.

Since both players act each round, samples are collected symmetrically from both players' perspectives.

### Game length cap

Self-play games are cut short after a fixed turn limit (e.g. 10 turns) in early generations. This keeps data collection tractable. The cap can increase as the NN improves and games become more meaningful.

### Randomness

Games are seeded randomly so the agent is exposed to diverse map layouts and starting conditions.

---

## MCTS Setup

The MCTS runs entirely within the current turn (no multi-turn lookahead). The value it returns is the mean `W/N` across all explored action paths from the root.

**Gen 0 evaluator:** hand-crafted heuristic function at leaf nodes.  
**Gen N evaluator (N ≥ 1):** NN from generation N−1.

No policy head is used — MCTS explores with a uniform prior.

### Implementation: C++ tree, Python eval

The MCTS tree runs entirely in C++. Python is only called once per batch of leaf states to run the NN forward pass.

```
C++ MCTSEngine::search(state, n_sims, eval_fn):
    loop:
        collect batch of BATCH_SIZE leaf states via UCT selection
        call eval_fn(batch) → values          ← single Python crossing
        backpropagate values through paths

Python eval_fn(states: list[GameState]) → list[float]:
    tensors = encode_batch(states)
    with torch.no_grad():
        return model(tensors).squeeze(-1).tolist()
```

`eval_fn` is a Python callable passed to C++ via pybind11. The boundary crossing happens ~100 times per move (800 sims / batch 8), not 800 times.

The C++ MCTS lives in `engine/src/mcts.cpp` + `engine/include/mcts.h`, exposed through `bindings/bindings.cpp`.

### END_TURN evaluation rule

When UCT selection would choose END_TURN as the next action, the simulation stops **before** applying it. The current state (pre-END_TURN) is sent to the evaluator.

This matches training exactly:

| | State evaluated | Perspective |
|---|---|---|
| Training input | `s_n` — right before END_TURN | current player |
| MCTS leaf eval | `s_n` — right before END_TURN | current player |

Backpropagation through END_TURN edges still negates the value (player switches), consistent with standard negamax MCTS.

Contrast with the existing `MCTSNN` approach, which steps into the post-END_TURN state and negates — wrong for imperfect information games because fog masks flip.

---

## Input Encoding

A new encoder is written for TDL — the existing `MCTSNN/encoder.py` is not reused. The game has changed (new units, new techs) and the fog-of-war handling was incorrect.

### Visibility rules

Polytopia has no traditional fog of war. Once a tile is explored it is permanently and fully revealed — terrain, units, buildings, and city state are all always visible. The only hidden information is unexplored tiles.

| Tile state | What is encoded |
|---|---|
| Not explored | All zeros |
| Explored | Everything |

The encoder gates on `is_explored`, not `is_visible`. They are equivalent for any tile that has been seen.

### Spatial channels (`C_in = 60`, per tile, shape `[60, 11, 11]`)

**Every channel is 0 if the tile is unexplored** — terrain, resources, buildings, borders, units, cities, all of it. The only information about an unexplored tile is that ch 0 is 0.

```
 Ch    Feature                        Notes
───────────────────────────────────────────────────────
  0    is_explored                    1 if explored; all other channels also 0 if this is 0

  1    terrain: Field                 one-hot — exactly one set for explored tiles
  2    terrain: Forest
  3    terrain: Mountain
  4    terrain: Water
  5    terrain: Village               cleared to Field when captured

  6    resource: Fruit
  7    resource: Crop
  8    resource: Animal
  9    resource: Metal                0 once harvested

 10    my_Mine                        ownership = border_city_id → city.owner(); transfers on capture
 11    opp_Mine
 12    my_Farm
 13    opp_Farm
 14    my_Road
 15    opp_Road

 16    my_border
 17    opp_border

 18    has_my_unit                    gate: 0 = no friendly unit here
 19    my_unit: Warrior               one-hot unit type
 20    my_unit: Archer
 21    my_unit: Rider
 22    my_unit: Defender
 23    my_unit: Giant
 24    my_unit: hp / max_hp
 25    my_unit: has_attacked
 26    my_unit: move_pts / base_move
 27    my_unit: kills (normalised)
 28    my_unit: promotion_ready

 29    has_opp_unit                   gate: 0 = no enemy unit here
 30    opp_unit: Warrior              one-hot unit type
 31    opp_unit: Archer
 32    opp_unit: Rider
 33    opp_unit: Defender
 34    opp_unit: Giant
 35    opp_unit: hp / max_hp
 36    opp_unit: has_attacked
 37    opp_unit: move_pts / base_move
 38    opp_unit: kills (normalised)
 39    opp_unit: promotion_ready

 40    has_my_city                    gate: 0 = no friendly city here
 41    my_city: is_capital
 42    my_city: level / 10
 43    my_city: pop / (level+1)       progress to next level-up
 44    my_city: units_owned / (level+1)  trained units alive / capacity
 45    my_city: is_sieged
 46    my_city: has_walls
 47    my_city: has_workshop
 48    my_city: pending_upgrade
 49    my_city: capture_ready         enemy unit started turn here — capture is legal this turn

 50    has_opp_city                   gate: 0 = no enemy city here
 51    opp_city: is_capital
 52    opp_city: level / 10
 53    opp_city: pop / (level+1)
 54    opp_city: units_owned / (level+1)
 55    opp_city: is_sieged
 56    opp_city: has_walls
 57    opp_city: has_workshop
 58    opp_city: pending_upgrade
 59    opp_city: capture_ready        my unit started turn here — I can capture this turn
```

### Global vector (`G = 11`)

```
  0    turn / 100
  1    my_stars / 30
  2    my_income / 20    (sum of stars_per_turn across all my cities)
  3    tech: Hunting
  4    tech: Organisation
  5    tech: Farming
  6    tech: Riding
  7    tech: Climbing
  8    tech: Archery
  9    tech: Mining
 10    tech: Strategy
```

`Origin` is excluded — every player always has it, so it carries no information. Opponent stars, income, and tech are not included — none are directly observable (imperfect information).

### Changes from `MCTSNN/encoder.py`

| | Old | New |
|---|---|---|
| Spatial channels | 50 | 60 |
| Global features | 11 | 11 |
| Unit types | Warrior, Archer, Rider (3) | + Defender, Giant (5) |
| Tech bits | 8 (including Origin) | 8 (Origin dropped — always 1) |
| Fog gate | `is_visible` | `is_explored` (Polytopia has permanent reveal) |
| Buildings | 3 channels (type only) | 6 channels (my/opp per type) |
| City channels | level, pop, sieged, walls, workshop, upgrade | + units_owned, capture_ready (×2 cities) |
| opp city workshop | not encoded | added |
| opp_visible_income | included in global | removed (not observable) |

---

## Neural Network

Value-only network. Takes the encoded spatial + global tensors as input.

### Architecture

```
spatial [B, C_in, 11, 11]           global_vec [B, G]
        │                                    │
  Stem Conv(C_in→64, 3×3, pad=1)            │
  BN → ReLU                [B,  64, 11, 11] │
        │                                    │
  ResBlock(64→64)          [B,  64, 11, 11] │
  ResBlock(64→64)          [B,  64, 11, 11] │
        │                                    │
  Conv(64→128, 3×3, pad=0) [B, 128,  9,  9] │   ← 11→9
  BN → ReLU                                  │
  Conv(128→128, 3×3, pad=0)[B, 128,  7,  7] │   ← 9→7
  BN → ReLU                                  │
  Conv(128→128, 3×3, pad=0)[B, 128,  5,  5] │   ← 7→5
  BN → ReLU                                  │
        │                                    │
  Conv(128→64, 1×1)        [B,  64,  5,  5] │   ← channel compress
  BN → ReLU                                  │
        │                                    │
  flatten                  [B, 1600]         │
        └──────────── cat ──────────────────┘
                      [B, 1600+G]
                         │
                  Linear(1600+G→256) → ReLU
                  Linear(256→1)      → Tanh
                         │
                   scalar ∈ [-1, 1]
```

`C_in` and `G` are determined by the input encoding (TBD — see Input Encoding section).

### Residual block

```
ResBlock(64→64):
    Conv(64→64, 3×3, pad=1) → BN → ReLU
    Conv(64→64, 3×3, pad=1) → BN
    + identity shortcut
    → ReLU
```

### Why this shape

- **Map size is 11×11** — the engine default is 9×9 (`DEFAULT_SIZE = 9`) but `make_random_game` must be updated to pass `sz=11`. 9×9 only produces ~1 neutral village, which is too limited for meaningful training. 11×11 gives ~3-4 villages and a real multi-city economy.
- **Gradual reduction 11→9→7→5** preserves spatial structure at each scale. Stopping at 5×5 (25 cells) gives enough resolution to represent capital distance and army positioning.
- **128 channels through the reduction** — richer features per cell as spatial resolution decreases.
- **1×1 compress (128→64) before flatten** — eliminates the large linear layer that would result from flattening 128 channels. Parameter-cheap with no spatial information loss.
- **Full receptive field before reduction** — the two ResBlocks (4 padded conv layers) give RF = 11, covering the whole board before any spatial reduction begins.

### Parameter count

| Layer | Params |
|---|---|
| Stem Conv(C_in→64, 3×3) | depends on C_in |
| ResBlock(64→64) × 2 | ~148K |
| Conv(64→128, 3×3) — 11→9 | ~74K |
| Conv(128→128, 3×3) — 9→7 | ~147K |
| Conv(128→128, 3×3) — 7→5 | ~147K |
| Conv(128→64, 1×1) — compress | ~8K |
| Linear(1600+G→256) | depends on G |
| Linear(256→1) | ~256 |
| **Fixed total (excl. input/global)** | **~524K + stem + linear** |

Final count depends on `C_in` and `G` from the input encoding design.

### Engine change required

`make_random_game` in [bindings/bindings.cpp](../../bindings/bindings.cpp) must be updated to expose a size parameter:

```cpp
m.def("make_random_game", [](uint64_t seed, int sz) {
    MapGenParams p = MapGenParams::for_biome(BiomeType::Drylands, sz);
    p.seed = seed;
    return MapGen(p).generate().state;
}, py::arg("seed") = 0, py::arg("sz") = 11);
```

---

## Training

**Loss:** MSE between `NN(encode(s))` and `V_mcts(s'')`

```python
loss = F.mse_loss(model(encode(s)), target_value)
```

**Data retention:** training data is kept across generations but capped at a maximum buffer size (e.g. last K games). Older data is dropped when the buffer is full, preventing the NN from overfitting to early, low-quality generations.

---

## Convergence Signal

After training Gen N, run an evaluation bracket against Gen N−1 before promoting it.

### Procedure

1. Play **E evaluation games** between Gen N (MCTS + NN_N) and Gen N−1 (MCTS + NN_{N-1}).
2. **Alternate starting player** across games to eliminate first-move advantage — half the games Gen N goes first, half it goes second.
3. Use the same MCTS simulation budget and turn cap as training (no special eval settings).
4. Compute **win rate** for Gen N across all E games.

### Threshold

| Win rate | Decision |
|---|---|
| > 52% | Accept Gen N — proceed to data collection for Gen N+1 |
| 40–52% | Ambiguous — generate more self-play data and retrain Gen N |
| < 40% | Regression — discard Gen N, follow regression protocol below |

The 52% bar accounts for the high game noise in Polytopia (map seed variance, first-move advantage) while still confirming Gen N is not actively regressing.

### ELO Tracking

Maintain a running ELO rating across all generations (including Gen 0 heuristic as the baseline). This gives a long-term view of progress that a single win-rate bracket can miss — e.g. a generation that barely beats the previous one but represents a large ELO jump is still healthy.

### Regression Protocol

Triggered when Gen N win rate < 40%.

1. **Discard Gen N weights** — roll back to Gen N−1 as the active evaluator.
2. **Diagnose before retraining** — compare train vs. validation loss on the Gen N training run:
   - High train loss → data quality problem (bad targets, too few games).
   - Low train loss, high val loss → overfitting.
   - Both low → something else, likely a bug in the TD target pipeline.
3. **Regenerate self-play data** using Gen N−1 with fresh random seeds — the original batch may have been unrepresentative.
4. **Retrain with adjusted hyperparameters** based on the diagnosis (e.g. lower learning rate, more dropout, more games).
5. Re-run the evaluation bracket.

**Repeated regression (two generations in a row):** stop and revisit fundamentals — the heuristic may be too weak to produce meaningful Gen 0 signal, or the TD target formulation has a bug. Do not keep iterating blind.

### On Ambiguous Result

If Gen N falls in the 40–52% ambiguous band:
- Generate an additional batch of self-play games using Gen N−1's evaluator.
- Retrain Gen N on the combined buffer.
- Re-run the bracket. If it still fails to clear 52%, accept it and proceed — stagnation for one generation is acceptable; repeated stagnation is a signal to revisit the heuristic or training hyperparameters.

---

## Gen 0 Heuristic

A simple linear scoring function used as the leaf evaluator before any NN is trained.

```python
def score(state, player):
    cities = [state.get_city(i) for i in range(...) if city.owner() == player]
    units  = [state.get_unit(i)  for i in range(...) if unit.owner() == player]
    techs  = state.techs_mask(player)  # bitmask; exclude Origin (bit 0)

    return (3.0 * len(cities)
          + sum(c.level() for c in cities)
          + 0.3 * bin(techs >> 1).count('1')   # shift out Origin bit
          + 0.2 * len(units))

def heuristic(state, player):
    diff = score(state, player) - score(state, 1 - player)
    return math.tanh(diff / C)   # C ≈ 15–20, tunable
```

**What it captures:** city economy (count + development), tech breadth, army size.  
**What it ignores:** unit HP, unit type quality, capital proximity, star balance, territory control.  
**Why that's fine:** Gen 0 just needs to provide a weak-but-consistent ordering of positions. The NN will learn the nuance.

`C` (scale constant) is a hyperparameter. Start with `C = 15`.

---

## Open Questions

- **MCTS simulations per move:** more sims = better targets but slower data collection. Start low (100–200) and increase in later generations.
- **Self-play games per generation:** how many games to play before training. Start with ~200.
- **Game length cap:** how many turns before cutting off a self-play game. Start with 30.
- **Buffer size K:** how many games to retain before dropping old data. Start with last 1000 games.
- **Evaluation game count E:** E=50 is a reasonable starting point.
- **Training hyperparameters:** optimizer (Adam), learning rate (1e-3), batch size (256), epochs per generation (10–20).
