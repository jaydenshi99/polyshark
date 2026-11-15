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
readout with a touch of nonlinearity.

## Output activation: scaled arctan

`(2/π)·arctan(x)` maps to `(-1, 1)`:
- `+1` = current player winning, `-1` = losing, `0` = even.
- Bounded range matches the MCTS backup and the win/loss training target.
- **Arctan over tanh:** softer saturation — its tails approach ±1 more slowly than
  tanh's, so gradients don't vanish as fast for confident (near-terminal) positions.
  The `2/π` factor rescales arctan's native `(-π/2, π/2)` range to `(-1, 1)`.

## Training target

MSE against the game outcome from the current player's perspective (`+1` win / `-1`
loss), or a bootstrapped MCTS value — set when the self-play loop is built.
