# Polyshark

A game engine and AI bot for [The Battle of Polytopia](https://polytopia.io/), a turn-based square-grid strategy game.

## Project Goal

Build a high-performance Polytopia simulator in C++ and use it as the backend for training and running an AI agent. The AI will combine reinforcement learning, tree search (MCTS/minimax), and potentially imitation learning from human replays.

## Directory Structure

```
polyshark/
├── engine/        # C++ game simulator
│   ├── src/
│   ├── include/
│   ├── tests/
│   └── docs/      # design and rules documentation
├── ai/            # Python ML / agent code
├── bindings/      # C++↔Python bridge (pybind11 or ctypes)
└── scripts/       # build and utility scripts
```

## Architecture

| Layer | Language | Purpose |
|---|---|---|
| Game engine | C++ | Fast, deterministic simulation of all game rules |
| ML / AI | Python | Model training, inference, agent orchestration |
| Bindings | TBD (pybind11 / ctypes) | Bridge between C++ engine and Python ML code |

## Current Status

Phase 1 MVP in progress. Implementing a stripped-down version of Polytopia to get core mechanics working before expanding scope.

## Scope

### Phase 1 MVP
- 3 unit types: Warrior, Archer, Rider
- 3 tech unlocks: Mining, Archery, Riding
- Terrain: Field, Forest, Mountain, Water + Villages
- Cities (no level cap): 1★/turn per level, +1★ capital bonus, siege mechanic
- City borders: 3×3 (levels 1–3), auto-expands to 5×5 at level 4 (Border Growth, no choice)
- Border clipping: borders between two cities don't overlap, clipped at midpoint
- Resources: Fruit/field food (no tech), Game/forest food (requires Hunting), Metal (requires Mining)
- Tech prereqs: Hunting required before Archery can be researched
- Fog of war
- Zone of control, healing, veteran promotion
- Win condition: capture enemy capital (Domination)
- 1v1 only

### Later phases
- More unit types, full tech tree
- Multiple tribes with starting bonuses
- Naval units
- Full building set
- Multi-player beyond 1v1
- Configurable map generation

## Game Rules

Full MVP ruleset is documented in [engine/docs/rules.md](engine/docs/rules.md).

**Fog of war (confirmed):**
- Standard vision: 1-tile radius (full 3×3 square, all 8 directions) for units and cities
- Mountain bonus: extended vision (exact radius TBD)
- Explored tiles are permanently revealed; enemy units hide when out of vision
- The AI plays with imperfect information — fog of war is in scope from the start

## AI Design

No fixed approach yet. The plan is to mix:
- **Reinforcement learning** (e.g. PPO or AlphaZero-style self-play)
- **MCTS / tree search** for planning
- **Imitation learning** from human replays (if data is available)

The C++ engine must support fast rollouts (critical for MCTS and RL self-play).

## Self-Updating Rules

When the user corrects a mistake, clarifies a game rule, or provides domain knowledge Claude didn't have, Claude must:
1. Acknowledge the correction
2. Immediately append the new knowledge to the relevant section of CLAUDE.md
3. Confirm it was written

This keeps CLAUDE.md as the living source of truth for Polyshark.

| Correction type | Where |
|---|---|
| Game rule clarification / domain knowledge | CLAUDE.md + rules.md |
| Behavioural correction ("stop doing X", "do Y instead") | feedback memory |
| Corrections that touch both | Write to both |

## What Claude Should Not Do

- **Don't invent game rules** — if a rule or mechanic is unclear, ask rather than guess.
- **Don't refactor engine core without being asked** — stability and correctness come first.
- **Prefer performance over readability in hot paths** — the engine exists to run fast rollouts; don't sacrifice speed for cleaner code in tight loops.

## Conventions

- The C++ engine should be deterministic and serializable (needed for MCTS tree expansion and replay).
- Python is for orchestration and ML only — no game logic in Python.
