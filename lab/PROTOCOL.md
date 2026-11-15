# Polyshark Bot Protocol (v1)

How a bot plays Polyshark games in the Match Lab, the Position Analyser and native self-play.
The engine (`engine/`) is the single source of truth: every rule, legal move and outcome comes
from it, so a game is a deterministic function of its start position and the bots' choices.

- [Transport](#transport) · [Session](#session) · [Messages](#messages) · [Observation](#observation)
- [Actions](#actions) · [Game rules the lab adds](#game-rules-the-lab-adds) · [Replicability](#replicability)
- [Failure handling](#failure-handling) · [In-process bots](#in-process-bots-self-play) · [Engine notes](#engine-notes)

## Transport

A bot is a program. The lab starts it in its own directory, writes **one JSON object per line**
to its stdin, and reads **one JSON object per line** from its stdout.

- Flush after every reply.
- stderr goes to the game log (capped at 1 MiB per bot per game).
- stdout lines that do not start with `{` are treated as log output and ignored. The SDKs redirect
  stray prints to stderr anyway.
- Bots must not read the clock or any unseeded randomness when they need games to be reproducible.
  Use the `seed` from `hello`.

SDKs: [`bots/sdk/cpp/polyshark_bot.h`](../bots/sdk/cpp/polyshark_bot.h) and
[`bots/sdk/python/polyshark_bot.py`](../bots/sdk/python/polyshark_bot.py). Skeletons:
[`bots/base`](../bots/base) (C++) and [`bots/base-py`](../bots/base-py) (Python).

## Session

```
lab → bot   hello                      once
bot → lab   ready                      within timeouts.init_ms
lab → bot   act        ┐
bot → lab   action     ┘ repeated, one per decision, within timeouts.move_ms each
lab → bot   end                        once; the bot should exit (killed after 0.5 s)
```

In the Position Analyser the lab sends `hello` with `"mode": "analysis"`, then any number of
`analyse` (or `act`) requests. The process is reused between requests and is killed when idle.

## Messages

### `hello` (lab → bot)

```json
{"type": "hello", "protocol": 1, "player": 0, "seed": 1234567, "mode": "match",
 "match": {"start": {"kind": "generated", "size": 9}, "turn_limit": 50, "observation": "fog",
           "timeouts": {"init_ms": 10000, "move_ms": 10000}, "budget": {"nodes": 1000},
           "start_turn": 0, "first_player": 0},
 "args": {}, "rules": {...}}
```

| Field | Meaning |
|---|---|
| `player` | Your seat, 0 or 1. Fixed for the session. |
| `seed` | 64-bit seed derived from the match seed, game and seat. The **only** randomness source a bot should use. |
| `mode` | `match`, `selfplay` or `analysis`. |
| `match.start` | `kind` (`generated` or `state`) and map `size`. The generator seed is never sent; it would reveal the fog. |
| `match.turn_limit` | The game is drawn once the turn counter has advanced this many times (0 = none). |
| `match.observation` | `fog` (default) or `full`. |
| `match.budget` | Search budget hint chosen in the lab, e.g. `{"nodes": 1000}` or `{"ms": 200}`. Prefer node budgets for reproducible experiments. |
| `args` | Free-form JSON passed through from the lab (per bot). |
| `rules` | Engine tables: units, techs, resources, upgrades, constants, action names, direction order. Generated from the engine, so bots never hard-code numbers. |

### `ready` (bot → lab)

```json
{"type": "ready", "name": "base", "version": "1", "protocol": 1, "capabilities": ["analyse"]}
```

`capabilities` may contain `analyse` if the bot answers `analyse` requests.

### `act` (lab → bot)

```json
{"type": "act", "id": 17, "player": 0, "turn": 3, "step": 40,
 "obs": {...}, "legal": [{"type": "EndTurn", "from": -1, "to": -1, "param": 0}, ...]}
```

`obs` is the [observation](#observation). `legal` is the list of **playable** actions
([Actions](#actions)). It is authoritative and always non-empty. `step` counts actions applied
so far in the game.

### `action` (bot → lab)

```json
{"type": "action", "id": 17, "index": 0, "eval": 0.12, "info": {"nodes": 900, "pv": ["..."]}}
```

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Echo of the request id. A reply with another id is ignored (the clock keeps running). |
| `index` | one of | Index into `legal`. Preferred. |
| `action` | one of | An action object instead of an index. It is matched on `type`, `from`, `to` and `param`; `param` is ignored for `Move`. |
| `eval` | no | Your evaluation for yourself, in [-1, 1]. Recorded in the replay and drawn as the eval graph. |
| `info` | no | Small JSON diagnostics (≤ 4 KiB), recorded in the replay and shown in the viewer. |

### `analyse` (lab → bot, analysis mode)

Same fields as `act` plus `budget`. Reply:

```json
{"type": "analysis", "id": 3, "eval": 0.05,
 "moves": [{"index": 4, "score": 0.31, "visits": 812, "note": "wins the village"}],
 "pv": [{"type": "Move", "from": 30, "to": 31, "param": 0}], "info": {}}
```

`moves` may list any subset of the legal actions, in any order. Reply
`{"type": "analysis", "id": 3, "unsupported": true}` to decline. The lab then shows what `act`
would play.

### `end` (lab → bot)

```json
{"type": "end", "player": 0, "winner": 1, "reason": "capital", "turns": 23}
```

`winner` is `null` for a draw. `reason` is `capital`, `turn_limit` or `forfeit`.

### `error` (bot → lab)

`{"type": "error", "id": 17, "message": "..."}`. Sent by the SDKs when the bot code throws. In a
game it forfeits.

## Observation

`obs` is the engine's own serialisation (`GameState::serialise`) of the game **as the player
sees it**, so `GameState::deserialise(obs)` gives a real, playable `GameState` for simulation and
search. Extra keys are ignored by the engine:

| Key | Meaning |
|---|---|
| `player` | The observer. |
| `observation` | `fog` or `full`. |
| `tiles[i].fog` | `true` on unexplored tiles. |
| `tiles[i].territory` | Owner (0/1) of the city whose border covers an explored tile, even if that city is unexplored. |

Under `fog`, which follows CLAUDE.md's rule that the only hidden information is unexplored tiles:

- **Explored tiles show their current true state**: terrain, resource, building, unit, city and
  capture flags. Explored tiles never fog over again.
- **Unexplored tiles** have `terrain: 0` (None) and nothing else.
- **Units on unexplored tiles are omitted.** Visible units keep their engine `id` across turns. A
  visible enemy whose home city is unexplored has `city_id: -1`.
- **Cities on unexplored tiles are omitted, and cities are renumbered.** The engine indexes cities
  by position, so view ids differ from the referee's internal ids. Every city id in `obs` and in
  `legal` uses view ids consistently. `capital_city[q]` is `-1` while q's capital is unexplored.
- **Both players' stars and techs are visible.**
- **The opponent's explored set is not observable.** `players[opponent].visible` is a copy of
  yours. This keeps the view simulatable (the engine blocks moves into unexplored tiles).
- `turn`, `current_player` and `map_size` are exact.

Under `full`, `obs` is the true state (plus `player`, `observation` and `territory`).

A bot that runs `legal_actions()` on its deserialised view usually gets the same list as `legal`.
They can differ when an unexplored enemy exerts zone of control, so always choose from `legal`.

## Actions

```json
{"type": "Move", "from": 12, "to": 22, "param": 0, "path": [12, 21, 22]}
```

| `type` | `from` | `to` | `param` |
|---|---|---|---|
| `Move` | unit tile | destination | movement left after the move; `path` = tiles walked (engine cheapest path) |
| `Attack` | attacker tile | target tile | 0 |
| `TrainUnit` | city tile | city tile | unit type id (`rules.units`) |
| `ResearchTech` | -1 | -1 | tech id (`rules.techs`) |
| `HarvestResource` | city id (border owner) | resource tile | resource id |
| `CaptureCity` | city id, or -1 for a village | tile | 0 |
| `UpgradeCity` | city id | -1 | upgrade id (`rules.upgrades`) |
| `Recover` | unit tile | -1 | 0 |
| `EndTurn` | -1 | -1 | 0 |

Tiles are indices `y * map_size + x`. Ids match `rules` (and the engine enums).

The engine also lists unaffordable actions and the `DebugAddPop` cheat. **Neither is ever sent.**
While a city has a pending level-up choice, only its two `UpgradeCity` options are legal, and
`EndTurn` is not.

## Game rules the lab adds

The engine defines the game. The lab adds only these match rules, each frozen with the match:

| Rule | Value |
|---|---|
| Win | The engine's: a player whose capital is captured loses (`reason: capital`). |
| Turn limit | Draw when `turn - start_turn ≥ turn_limit` (`reason: turn_limit`). From a generated start (player 0 first), both players get exactly `turn_limit` turns. |
| Forfeit | A bot that times out, crashes, replies with an illegal or malformed action, or takes more than `max_actions_per_turn` (2000) actions in one turn loses (`reason: forfeit`). |
| Seats | Every pairing is played twice per map seed, sides swapped. |

## Replicability

A game is fully determined by:

1. **Start**: `{"kind": "generated", "generator": "drylands", "size": 9, "seed": S}` (the engine's
   seeded xorshift map generator), or `{"kind": "state", "state": {...}}`.
2. **Bot seeds**: `derive_seed(match_seed, game, seat)` (splitmix64), recorded per game.
3. **The bots' choices.** The engine itself has no other randomness. Its explorer RNG is seeded from
   tile, player and turn.

The lab freezes everything else per match:

- **Bot sources** are copied into the match and hashed. Builds are cached by content hash.
- **The engine**: `engine/`, `lab/native/` and the SDK are snapshotted per *engine hash*, and the
  referee is built from that snapshot. A match keeps replaying with its own engine after the code
  changes.
- **Compiler flags** include `-ffp-contract=off`, so combat arithmetic never depends on fused
  multiply-add hardware. Results are identical on x86 and ARM.
- **Replays** store the start, every applied action (with its engine path bits), every decision's
  time, eval and info, and the final state hash. The tool re-simulates a replay and checks the hash.

Wall-clock timeouts are the only machine-dependent part. Generous timeouts plus node budgets make a
deterministic bot reproduce its games bit for bit.

### Replay format

```json
{"format": "polyshark-replay", "version": 1, "engine": "<engine hash>", "game_id": "...",
 "start": {...}, "start_hash": "...", "turn_limit": 50, "observation": "fog",
 "players": [{"name": "base", "bot": "<bot id>", "seed": 1, "version": "1"}, ...],
 "actions": [[type, from, to, param, path_bits, path_steps], ...],
 "decisions": [{"p": 0, "ms": 0.4, "eval": 0.1, "info": {}}, ...],
 "result": {"winner": 0, "reason": "capital", "turns": 23, "final_hash": "...", ...}}
```

`actions` use engine ids (not view ids). `decisions[k]` belongs to `actions[k]`.

## Failure handling

| Event | Outcome |
|---|---|
| No `ready` within `init_ms`, or the bot exits during init | forfeit (`kind: init`) |
| No `action` within `move_ms` | forfeit (`timeout`) |
| Process exits, or stdout closes | forfeit (`crash`) |
| `index` out of range, or an `action` that is not legal | forfeit (`illegal`) |
| Unparseable JSON, wrong message type, or an `error` message | forfeit (`protocol`) |
| More than 2000 actions in one turn | forfeit (`action_limit`) |

The forfeit and its message are stored in the game result and shown in the lab.

## In-process bots (self-play)

`POLYSHARK_BOT(MyBot, "name", "version")` also exports a C ABI factory
(`polyshark_bot_create`, `polyshark_bot_abi`, `polyshark_bot_engine`). The native self-play
runner `dlopen`s the bot, built as a shared library against the same frozen engine, and calls
`act()` directly with the same `Observation`: the fog view as a `GameState` and the legal list in
view ids. There is no process, pipe or JSON per decision. The runner refuses a plugin built
against another engine hash or ABI version. Exceptions thrown by `act()` forfeit the game. A crash
takes the whole self-play run down, so debug new bots in Match Lab games first.

## Engine notes

The lab mirrors the engine exactly, including where it differs from `engine/docs/rules.md`:

- Archers have **Dash** in `UNIT_DEFS`. rules.md lists only Ranged and Fortify.
- Fruit requires **Organisation** (rules.md agrees; CLAUDE.md's scope says no tech).
- The tech tree includes Farming (Crop, 5★, +2 pop, Farm), Strategy (Defender) and the Giant super
  unit. Workshop gives +1★. The Park upgrade currently does nothing: `apply_upgrade_city` never
  calls `add_park()`.
- Defence bonus: ×1.5 on terrain your techs defend (Climbing → mountain, Archery → forest), +0.5 in
  a city, +0.5 more with walls. It applies to units with Fortify only.
- There is no action to accept a veteran promotion. Units become `promotion_ready` and stay so.
- `CaptureCity` re-homes the capturing unit to the captured city even when that city is full, so
  `check_invariants()` can report `units_owned` above capacity.
