# Bots

Every folder here containing a `bot.json` shows up in the Match Lab automatically. A bot speaks
the [Polyshark bot protocol](../lab/PROTOCOL.md): newline-delimited JSON on stdin and stdout. You
can write one in any language; the SDKs below do the plumbing.

| Folder | What it is |
|---|---|
| [`base/`](base) | C++ skeleton: ends every turn. Copy it to start a C++ bot. |
| [`base-py/`](base-py) | Python skeleton: ends every turn. Copy it to start a Python bot. |
| [`sdk/cpp/polyshark_bot.h`](sdk/cpp/polyshark_bot.h) | C++ SDK (header-only): `psb::Bot`, the protocol loop, the action codec. |
| [`sdk/python/polyshark_bot.py`](sdk/python/polyshark_bot.py) | Python SDK: `Bot`, `Observation`, `Decision`, `run()`. |

## Start a bot

```bash
cp -r bots/base bots/my-bot        # or bots/base-py
# edit bots/my-bot/bot.json → "name": "my-bot"
```

Open **Bots & Protocol** in the lab and press **Test**. It builds the bot and plays a 3-turn game
against itself through the real referee. Then pick it in **Match Lab**.

### C++

```cpp
#include "polyshark_bot.h"

class MyBot : public psb::Bot {
public:
    void init(const psb::Hello& hello) override { me_ = hello.player; }
    psb::Decision act(const psb::Observation& obs) override {
        const GameState& view = obs.view();             // your fog-of-war view, a real GameState
        for (int i = 0; i < obs.legal_count; i++) {     // obs.legal is authoritative
            GameState next = view.apply_action(obs.legal[i]);  // simulate with the engine
            // ...
        }
        int end = obs.find(ActionType::EndTurn);        // absent while a city upgrade is pending
        return { end >= 0 ? end : 0 };
    }
private:
    int me_ = 0;
};
POLYSHARK_BOT(MyBot, "my-bot", "1")
```

The lab compiles every `*.cpp` in the folder with the frozen engine, using
`-std=c++17 -O2 -ffp-contract=off`, and links `libengine`. It builds two things:

- **an executable**, for Match Lab games and the Position Analyser;
- **a plugin**, for native self-play. The plugin calls `act()` directly with the same
  `Observation`.

The same code serves both. Optional overrides:

- `analyse(obs, budget, out)`: fill `out.eval` and `out.moves` (scores and visits per legal index),
  plus `out.pv` and `out.info`. The Position Analyser shows them.
- `end(result)`: called once the game is over.
- `Decision::eval` / `Decision::info`: recorded in the replay. The viewer graphs `eval` and shows
  `info`.

### Python

```python
from polyshark_bot import Bot, Decision, run

class MyBot(Bot):
    name = "my-bot"
    def act(self, obs):              # obs.state (JSON view), obs.legal (list of actions)
        return obs.find("EndTurn") or 0
run(MyBot())
```

`obs.state` is the engine's serialisation of your view. It deserialises into the pybind
`polyshark.GameState` if you want to simulate. Anything you `print` goes to the game log.

## `bot.json`

| Field | Meaning |
|---|---|
| `name` | Display name (you can rename in the lab without editing the file). |
| `description` | Shown in pickers. |
| `language` | `cpp` (default), `python`, or `command`. |
| `sources` | C++ only: globs relative to the folder (default `["*.cpp"]`). |
| `include` | C++ only: extra include dirs relative to the folder. |
| `flags` | C++ only: extra compiler flags. Avoid `-ffast-math`, which breaks reproducibility. |
| `entry` | Python only: script to run (default `bot.py`). |
| `python` | Python only: interpreter, e.g. `.venv/bin/python` for torch (default: the server's). |
| `run` | `command` only: the command line, run in the frozen copy of the folder. |

Without a `bot.json`, a folder is still recognised if it has a `.cpp` using `POLYSHARK_BOT(` or a
`bot.py`.

## Rules of thumb

- Use only `hello.seed` for randomness, and prefer node budgets (`hello.match.budget.nodes`) to
  wall-clock limits. Your games then reproduce exactly.
- Always choose from `obs.legal`. Your own `legal_actions()` on the view can differ when an
  unexplored enemy exerts zone of control.
- A timeout, crash, illegal index or more than 2000 actions in one turn forfeits the game.
