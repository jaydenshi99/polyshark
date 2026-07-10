# Value Head

Reads the 256-d shared representation (see [core.md](core.md)) and outputs a single
scalar: the current player's expected outcome.

## Structure

```
shared rep [256] ─ Linear(256 → 64) → GELU
                 ─ Linear(64 → 1)   → (2/π)·arctan
                 └─> value ∈ (-1, 1)
```

Small on purpose — the shared core already did the heavy lifting; the head is just a
readout with a touch of nonlinearity. A Dropout(0.2) on its input (kept outside the
Sequential so checkpoint keys are stable) regularizes it during training.

## Output activation: scaled arctan

`(2/π)·arctan(x)` maps to `(-1, 1)`:
- `+1` = current player winning, `-1` = losing, `0` = even.
- Bounded range matches the MCTS backup and the win/loss training target.
- **Arctan over tanh:** softer saturation — its tails approach ±1 more slowly than
  tanh's, so gradients don't vanish as fast for confident (near-terminal) positions.
  The `2/π` factor rescales arctan's native `(-π/2, π/2)` range to `(-1, 1)`.

## Training target

MSE against the **mixed target** `(1−w)·z + w·v̂` from the acting player's perspective:
`z` = game outcome (±1 on capital capture; ±1 winner-by-margin at the turn cap, with a
small negative tie-contempt label inside the dead zone), `v̂` = the search root value
recorded at the decision. `w` (`search_value_weight` ≈ 0.3) is annealed from 0 over the
first gens. See [training.md](training.md) for why pure per-game outcomes collapse
(no within-game credit) and for the anti-memorization guards around this head
(value subsampling, dropout, D8 symmetry augmentation, held-out validation).
