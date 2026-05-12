# Polyshark

A game engine and AI bot for [The Battle of Polytopia](https://polytopia.io/), a turn-based hex-grid strategy game.

## Project Goal

Build a high-performance Polytopia simulator in C++ and use it as the backend for training and running an AI agent. The AI will combine reinforcement learning, tree search (MCTS/minimax), and potentially imitation learning from human replays.

## Directory Structure

```
polyshark/
├── engine/        # C++ game simulator
│   ├── src/       # implementation files
│   ├── include/   # header files
│   └── tests/     # C++ unit tests
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

Just started. No code exists yet.

## Scope

### Phase 1 (initial)
- Single tribe (TBD)
- Fixed map layout (single map, single size — TBD)
- Full rule coverage for that tribe: map/terrain, units & combat, cities & economy, tech tree

### Later phases
- Multiple tribes
- Configurable / random map generation
- Multi-player (multiple AI agents)

## Game Systems to Implement

- **Map / terrain** — hex grid, terrain types (forest, mountain, water, field, etc.), resources
- **Units & combat** — unit stats, attack/defense resolution, movement, promotions
- **Cities & economy** — star income, city upgrades, population growth, capturing
- **Tech tree** — researching technologies, unlocking units and buildings

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

Where to write corrections:

| Correction type | Where |
|---|---|
| Game rule clarification / domain knowledge | CLAUDE.md |
| Behavioral correction ("stop doing X", "do Y instead") | feedback memory |
| Corrections that touch both | Write to both |

## What Claude Should Not Do

- **Don't invent game rules** — if a rule or mechanic is unclear, ask rather than guess.
- **Don't refactor engine core without being asked** — stability and correctness come first.
- **Prefer performance over readability in hot paths** — the engine exists to run fast rollouts; don't sacrifice speed for cleaner code in tight loops.

## Conventions

- The C++ engine should be deterministic and serializable (needed for MCTS tree expansion and replay).
- Keep the engine stateless / functional where possible to make tree search easier.
- Python is for orchestration and ML only — no game logic in Python.
