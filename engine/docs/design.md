# Engine Design

> **Note:** This document is a rough draft and starting point. All design decisions, interfaces, and data structures are tentative and will change as the engine is built. Do not treat anything here as final.

## Overview

The engine is a fast, deterministic simulator of Polytopia game rules written in C++.
It has no AI logic — it only knows the rules and how to advance game state.
The AI (in `ai/`) calls into the engine to simulate moves and read game state.

## Core Principle: Separate State from Logic

```
GameState   — data + simple query methods. Represents a snapshot of the game.
GameRules   — stateless free functions that contain actual game logic (legal_actions, apply_action).
```

Simple queries (`is_terminal`, `winner`, `current_player`) live as methods on `GameState`.
Complex rule logic stays as free functions so it can evolve independently of the state layout.

`GameState` must remain trivially copyable — no heap allocation, no virtual methods, no
user-defined copy constructor. This lets MCTS clone states with a plain copy.

---

## GameState

Everything needed to fully describe the game at a single point in time.

```cpp
class GameState {
public:
    bool is_terminal()    const;
    int  winner()         const;
    int  current_player() const;
    void print_map()      const;  // debug only

    friend void      legal_actions(const GameState&, Action[], int&);
    friend GameState apply_action(GameState, Action);

private:
    Tile map[MAP_TILES];          // 11x11 flat array
    Unit units[MAX_UNITS];
    City cities[MAX_CITIES];

    int unit_count  = 0;
    int city_count  = 0;
    int stars       = 0;
    int turn        = 0;
    int cur_player  = 0;
    // TechTree — not yet implemented
};
```

### Why a flat array for the map?

The map is 11x11 = 121 tiles. Storing it as `Tile map[121]` instead of a
2D vector means:
- All tile data is contiguous in memory — the CPU can prefetch it efficiently
- No heap allocation — copying `GameState` is just a `memcpy`
- Simple index math: `map[row * 11 + col]`

### Tile

```cpp
struct Tile {
    TerrainType terrain;   // forest, mountain, water, field, etc.
    ResourceType resource; // fruit, game, fish, metal, etc. (or NONE)
    int unit_id;           // index into units[], or -1 if empty
    int city_id;           // index into cities[], or -1 if none
};
```

### Unit

Units are split into two things: a **static definition** (stats that never change)
and a **runtime instance** (state that changes during the game).

The definition lives in a global lookup table — never copied, never heap-allocated:

```cpp
struct UnitDef {
    int      hp;
    int      attack;
    int      defense;
    int      movement;
    int      range;
    uint32_t abilities;  // bitmask of ability flags
};

// Ability flags
constexpr uint32_t ABILITY_FLOAT   = 1 << 0;  // can move on water
constexpr uint32_t ABILITY_DASH    = 1 << 1;  // can move after attacking
constexpr uint32_t ABILITY_PERSIST = 1 << 2;  // survives killing a unit
// ... etc.

static const UnitDef UNIT_DEFS[NUM_UNIT_TYPES] = {
    // [WARRIOR]  = { .hp=10, .attack=2, .defense=2, .movement=1, .range=1, .abilities=0 },
    // [ARCHER]   = { .hp=10, .attack=2, .defense=1, .movement=1, .range=2, .abilities=0 },
    // ...
};
```

The runtime instance is a small trivially-copyable struct stored in `GameState`:

```cpp
struct Unit {
    UnitType type      = UnitType::None;
    int  owner        = -1;
    int  tile_index   = -1;  // must stay in sync with Tile::unit_id
    int  hp           = 0;
    int  move_points  = 0;
    bool has_attacked = false;
};

// To read a unit's base stats:
const UnitDef& def = UNIT_DEFS[unit.type];
if (def.abilities & ABILITY_FLOAT) { /* allow water movement */ }
```

**Why not virtual methods / inheritance?**
Virtual dispatch requires vtable pointer lookups on every call and prevents trivial
copying — both are problems in hot paths. The differences between unit types are
almost entirely data (different stats, different ability flags), not behavior.
The rules engine handles all behavior by checking flags on the `UnitDef`.

### City

```cpp
struct City {
    int  owner      = -1;
    int  population = 0;
    int  tile_index = -1;
    bool is_capital = false;  // Domination win condition: capture all enemy capitals
    // upgrades, walls, etc. added later
};
```

---

## Grid Coordinates

Polytopia uses a **square grid**. Units move in 4 directions (N, S, E, W).
Coordinates are `(x, y)` where `x` is column and `y` is row.

```
Neighbors of (x, y):
  (x, y-1)  — north
  (x, y+1)  — south
  (x-1, y)  — west
  (x+1, y)  — east

Distance between (x1,y1) and (x2,y2) (Manhattan distance):
  |x2-x1| + |y2-y1|
```

Conversion to flat array index:
```cpp
int to_index(int x, int y) { return y * 11 + x; }
```

---

## Public API

These are the only functions external code (AI, Python bindings, tests) is allowed
to call. Everything else is internal to the engine.

Methods on `GameState` (simple queries):
```cpp
s.is_terminal();    // is the game over?
s.winner();         // winning player index, or -1
s.current_player(); // whose turn it is
```

Free functions (game logic):
```cpp
// Actions are written into a caller-provided array — no heap allocation
void legal_actions(const GameState& s, Action out[], int& out_count);

// Takes GameState by value so the caller controls whether to copy or move
GameState apply_action(GameState s, Action a);

// Serialize / deserialize to raw bytes (for MCTS, saving, Python bindings)
void serialize(const GameState& s, uint8_t* out_buf, int& out_size);
GameState deserialize(const uint8_t* buf, int size);
```

### Why `apply_action` takes `GameState` by value?

It means the caller controls whether to clone:
```cpp
GameState next = apply_action(current, a);              // current is untouched
GameState next = apply_action(std::move(current), a);   // current is consumed (faster)
```
MCTS uses the first form to branch; the main game loop uses the second.

---

## Actions

An action is anything the current player can do on their turn.

```cpp
enum class ActionType {
    Move,
    Attack,
    TrainUnit,
    BuildImprovement,
    ResearchTech,
    CaptureCity,
    EndTurn,
};

struct Action {
    ActionType type;
    int from;    // tile index (where relevant)
    int to;      // tile index (where relevant)
    int param;   // unit type, tech id, improvement type, etc.
};
```

Encoding actions as small plain structs (no pointers, no heap) means they can
be stored in arrays and passed to Python as integers for RL training.

---

## Design Rules

- No dynamic allocation (`new`, `vector`, `string`) in hot paths (`apply_action`, `legal_actions`)
- No virtual dispatch in hot paths
- `GameState` must be trivially copyable
- All randomness goes through a seeded RNG passed as a parameter — never global state
