# Polyshark Match Lab

A local website for developing Polyshark bots. You can:

- run matches between bots and get statistics,
- look at any position and see what a bot thinks of it,
- run fast self-play.

The C++ engine decides every rule, legal move and outcome. The Python server only schedules work
and serves pages.

```bash
cmake -S . -B build                      # once: fetches nlohmann/json, which the lab compiles against
python3 lab/lab.py serve                 # http://127.0.0.1:8780
```

On first start the server snapshots and compiles the engine runtime. This takes a few seconds and
needs `clang++` (or `$CXX`). The header shows `Engine <hash>` when it is ready. Run
`python3 lab/lab.py build` to do the same from the command line. Use `--max-workers N` to change
how many games run at once (default: CPUs − 2) and `--data DIR` to keep runs somewhere other than
`lab/data/`.

## Tabs

| Tab | What it does |
|---|---|
| **Match Lab** | Start matches (sidebar) and watch the queue. Open a match for Results, Games (replays, logs) and Frozen inputs. |
| **Position Analyser** | Load a position from a generated map, a saved file, a starred position, any step of a match or self-play game, or pasted JSON. Click units to play moves, compare bot analyses, branch what-ifs, and save positions. |
| **Self-play** | Native, multithreaded runs with C++ bots loaded in-process. Live throughput, results, game lengths, and replays of the first N games. |
| **Bots & Protocol** | The bot catalogue, a one-click protocol test (build, then a 3-turn game against itself), engine runtime status, and [PROTOCOL.md](PROTOCOL.md). |

### Match Lab

- **Modes**:
  - *Everyone vs everyone*: every pair of selected bots plays.
  - *Against a reference*: each challenger plays one reference bot.
  - *Self-play*: each bot plays a copy of itself. The score then measures first-move advantage.
- **Maps**:
  - *Generated* (`Drylands N×N`): every pair gets a fresh engine seed. Both games of the pair use
    it with seats swapped, and every matchup plays the same seeds.
  - *Saved positions*: any serialised `GameState` in `data/saved_gamestates/`,
    `data/phase_1_training_set/` or `lab/maps/`.
- **Settings**: pairs per map, simultaneous games, turn limit (the game is a draw when it is
  reached), fog or full observation, timeouts, search budget hints (`hello.match.budget`), replay
  saving, match seed, name, and per-bot JSON arguments.
- **Results**: only complete side-swapped pairs are scored. The Results tab shows:
  - each bot's score with a Wilson 95% interval and an Elo estimate,
  - how games ended (capital captured, turn limit, forfeits by kind),
  - a head-to-head heatmap,
  - results by map, with an opponent breakdown per map,
  - seat advantage, decision time, per-map game statistics, and game length.
- **Games**: filter and page through the games. Each game has **Watch** (replay viewer),
  **Analyse** (opens the Position Analyser at that game), the replay JSON, the stderr log of each
  player, and a cancel button while the game is running.
- **Control**: **Cancel** keeps completed games. **Requeue unfinished** reruns failed, cancelled
  and unfinished games. A server restart leaves running matches *paused* until you requeue them.

Bots come from `bots/` automatically. **Load another bot** links a folder: it stays live, so new
bots and edits appear. A ZIP upload is a snapshot. Only starred bots appear in the picker; manage
stars, display names and removal under **Manage starred bots**.

### Replay viewer

The viewer replays the game in the engine from the start and checks the final state hash
(*replay verified*). It has:

- a scrubber, playback and keyboard shortcuts: ←/→ for one action, Shift+←/→ for one turn,
  Space to play, Home/End;
- the fog of either player or the side to move;
- iso (the raylib visualizer's sprites and projection) or flat rendering;
- arrows for the last and the next action;
- an action log grouped by turn, with each decision's time, eval and `info`;
- graphs of eval, income, stars, units, cities and HP.

**Analyse position** opens the analyser at the current step. **Visualizer file** exports a
`.replay` for `visualizer --replay` (generated maps only).

### Position Analyser

- **Play moves**: click a unit of the side to move to see its moves (cyan) and attacks (red), then
  click a target to play it. The legal move list groups every playable action and previews each
  attack's HP outcome. Played moves form a what-if branch with undo/redo (←/→), and each step lists
  its consequences (moves, damage, deaths, captures, stars, tiles explored).
- **Analyse with a bot**: pick a bot, an optional node or millisecond budget, and fog or full
  observation. The analysis shows an eval bar, ranked moves with scores and visits, the PV and the
  bot's info and stderr. The top three moves are drawn as arrows. Analyses are kept per position, so
  you can compare bots. Bots without `analyse` show the move `act` would play. Bot processes stay
  warm between requests.
- **Replay steps**: from a game, step through the actual moves with ‹ › or `[` `]`.
- **Keep a position**: **Star** saves it in the lab. **Save as…** writes
  `data/saved_gamestates/<name>.json`, which loads in the raylib visualizer and appears as a Match
  Lab start position.

### Self-play

C++ bots are compiled as shared libraries and called through `psb::Bot::act` directly. There is no
process, pipe or JSON per move, the engine applies actions in place, and games run on N threads
with thread-local buffers. On a 10-core M-series laptop the builtin random policy reaches about
4.4M decisions/s (≈7,800 games/s on 11×11 with a 50-turn limit). The builtins `random` and
`endturn` benchmark the engine itself. Seats alternate every game, and each pair of games shares a
map seed. One run executes at a time; later runs queue.

## Layout

```
lab/
  lab.py            CLI: serve | build
  PROTOCOL.md       the bot protocol (read this to write a bot)
  native/           C++ runtime shared by everything that plays games
    lab_core.*      fog-of-war views, in-place apply, codec, rules tables, hashing, outcome rules
    referee.cpp     one game between two protocol bots → result + replay
    tool.cpp        JSON-lines engine service for the website (legal moves, apply, replays…)
    selfplay.cpp    in-process multithreaded self-play
    process.*       bot subprocesses: pipes, poll deadlines, stderr capture, process groups
    test_lab.cpp    checks apply_inplace ≡ engine apply, views, round trips, determinism
  server/           Python (stdlib only, 3.9+): catalog, native builds, scheduler, analysis, HTTP
  web/              the website (no build step; ES modules)
  maps/             extra start positions (optional)
  data/             runtime data, git-ignored: matches/, selfplay/, engines/, builds/, uploads/…
```

## Replicability

Every game is a deterministic function of its start position, the bots' seeds and the bots'
choices (see [PROTOCOL.md § Replicability](PROTOCOL.md#replicability)). For each match the lab
freezes:

- the **bot sources**: copied into `data/matches/<n>/frozen/` and hashed. Builds are cached by
  content.
- the **engine runtime**: `engine/`, `lab/native/` and the bot SDKs are snapshotted under
  `data/engines/<hash>/` and compiled with `-ffp-contract=off`. The referee of a match comes from
  its own snapshot, so requeuing after you change the engine still plays the original rules.
- the **seeds**: map seeds are `derive_seed(match_seed, map, pair)` and bot seeds are
  `derive_seed(match_seed, game, seat)`. Both are splitmix64 and identical in C++ and Python.

Replays store every applied action with its engine path, and the final state hash. The viewer
re-simulates the replay and verifies the hash.

## Native tests and dev builds

```bash
cmake --build build --target test_lab polyshark-referee polyshark-tool polyshark-selfplay
./build/lab/test_lab
```

The CMake targets serve IDEs and tests. The server builds its own hashed snapshots.
