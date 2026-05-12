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

    // Fog of war — see "Fog of War" section
    bool is_visible(int player, int tile)  const;
    bool is_explored(int player, int tile) const;
    void set_visible(int player, int tile);
    void set_explored(int player, int tile);  // also marks visible
    void clear_visible(int player);

    // Tech unlocks — see "Tech Bitmask" section
    bool has_tech(int player, TechType t) const;
    void research_tech(int player, TechType t);

    friend void      legal_actions(const GameState&, Action[], int&);
    friend GameState apply_action(GameState, Action);

private:
    Tile map[MAP_TILES];
    Unit units[MAX_UNITS];
    City cities[MAX_CITIES];

    uint16_t explored[MAX_PLAYERS][FOG_WORDS] = {};
    uint16_t visible[MAX_PLAYERS][FOG_WORDS]  = {};
    uint32_t techs[MAX_PLAYERS]               = {};
    int      stars[MAX_PLAYERS]               = {};

    int unit_count  = 0;
    int city_count  = 0;
    int turn        = 0;
    int cur_player  = 0;
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
    ResourceType resource; // Fruit, Game, Metal, or None
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
    int      cost;
    uint32_t abilities;
};

// Ability flags (defined in types.h)
constexpr uint32_t ABILITY_FORTIFY = 1 << 0;  // ×1.5 defence on forest/mountain/city
constexpr uint32_t ABILITY_DASH    = 1 << 1;  // can move after non-attack action
constexpr uint32_t ABILITY_ESCAPE  = 1 << 2;  // can move again after attacking (Rider)
constexpr uint32_t ABILITY_RANGED  = 1 << 3;  // attacks at 2-tile range, doesn't advance

// Indexed by UnitType — defined in unit_def.h
static const UnitDef UNIT_DEFS[] = {
    // None
    {},
    // Warrior:   HP  ATK DEF MOV RNG COST  ABILITIES
    {             10,  2,  2,  1,  1,  2,   ABILITY_FORTIFY | ABILITY_DASH    },
    // Archer
    {             10,  2,  1,  1,  2,  3,   ABILITY_RANGED  | ABILITY_FORTIFY },
    // Rider
    {             10,  2,  1,  2,  1,  3,   ABILITY_ESCAPE  | ABILITY_FORTIFY | ABILITY_DASH },
};

// Helper — preferred over indexing UNIT_DEFS directly
inline const UnitDef& unit_def(UnitType t) {
    return UNIT_DEFS[static_cast<int>(t)];
}
```

The runtime instance is a class stored in `GameState`. Fields are private; access via getters:

```cpp
class Unit {
public:
    UnitType type()            const;
    int      owner()           const;
    int      tile_index()      const;  // must stay in sync with Tile::unit_id
    int      hp()              const;
    int      max_hp()          const;  // base HP + 5 per promotion
    int      move_points()     const;
    bool     has_attacked()    const;
    int      kills()           const;
    bool     promotion_ready() const;  // true when kills >= 3 and not yet accepted
    bool     is_alive()        const;  // hp > 0

    void take_damage(int amount);
    void spend_movement(int amount);
    void mark_attacked();
    void add_kill();          // increments kill count, sets promotion_ready at 3
    void accept_promotion();  // +5 max_hp, full heal, clears promotion_ready
    void reset_turn(int base_movement);
};

// To read a unit's base stats:
const UnitDef& def = unit_def(unit.type());
if (def.abilities & ABILITY_RANGED) { /* 2-tile attack, no advance on kill */ }
```

**Why not virtual methods / inheritance?**
Virtual dispatch requires vtable pointer lookups on every call and prevents trivial
copying — both are problems in hot paths. The differences between unit types are
almost entirely data (different stats, different ability flags), not behavior.
The rules engine handles all behavior by checking flags on the `UnitDef`.

### City

```cpp
class City {
public:
    int  owner()      const;
    int  level()      const;     // 1–3
    int  population() const;
    int  tile_index() const;
    bool is_capital() const;
    bool is_sieged()  const;     // enemy unit on tile → 0★ income

    int  stars_per_turn() const; // level + capital bonus, 0 if sieged
    int  unit_capacity()  const; // level + 1

    void add_population(int n);  // auto-levels when population >= next level
    void capture(int new_owner);
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
int to_index(int x, int y) { return y * MAP_SIZE + x; }
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

## Fog of War

Two bitfields per player, each `uint16_t[FOG_WORDS]` where `FOG_WORDS = (MAP_TILES + 15) / 16`:

- **`explored`** — permanently revealed; set once, never cleared
- **`visible`** — currently in a unit or city's vision radius; rebuilt each turn

Bit layout: tile index `i` lives in word `i/16`, bit `i%16`.

```cpp
constexpr int FOG_WORDS = (MAP_TILES + 15) / 16;  // 8 words for 121 tiles

bool is_visible(int player, int tile)  const;
bool is_explored(int player, int tile) const;
void set_visible(int player, int tile);
void set_explored(int player, int tile);  // also marks visible
void clear_visible(int player);           // called at start of each turn
```

`set_explored` always calls `set_visible` — a newly explored tile is also currently visible.

---

## Tech Bitmask

Each player's unlocked technologies are stored as a single `uint32_t` bitmask, one bit per `TechType`. This is zero-initialised (no techs at game start).

```cpp
uint32_t techs[MAX_PLAYERS] = {};

bool has_tech(int player, TechType t) const {
    return (techs[player] >> static_cast<int>(t)) & 1;
}
void research_tech(int player, TechType t) {
    techs[player] |= (1 << static_cast<int>(t));
}
```

Tech costs and what each unlocks are defined in rules.md.

---

## Design Rules

- No dynamic allocation (`new`, `vector`, `string`) in hot paths (`apply_action`, `legal_actions`)
- No virtual dispatch in hot paths
- `GameState` must be trivially copyable
- All randomness goes through a seeded RNG passed as a parameter — never global state
