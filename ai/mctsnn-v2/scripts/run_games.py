"""
Configurable game runner for the mctsnn-v2 arena.

Edit the CONFIG block below, then run from the repo root (with the venv active):

    source .venv/bin/activate
    python ai/mctsnn-v2/scripts/run_games.py

No CLI flags — everything is set in CONFIG. Pick the two agents, how many games,
the turn limit, seeds, side-swapping, whether to record training samples, and whether
to dump a .replay per game for the visualiser.
"""

import os
import sys

# --- make the mctsnn-v2 src package + engine bindings importable ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))               # ai/mctsnn-v2/scripts
_PKG  = os.path.dirname(_HERE)                                    # ai/mctsnn-v2
_ROOT = os.path.dirname(os.path.dirname(_PKG))                    # repo root
sys.path.insert(0, os.path.join(_PKG, "src"))
sys.path.insert(0, os.path.join(_ROOT, "build", "bindings"))

import polyshark  # noqa: E402
from arena import Arena, write_replay  # noqa: E402
from agents import build_agent  # noqa: E402  (spec dict -> Agent; shared with train.py)


# ============================================================================
# CONFIG — edit this
# ============================================================================

# Each agent is a spec dict, turned into an Agent by build_agent() below.
# Supported "type" values:
#   "random" : uniform over legal actions.  fields: name, seed
#   "mcts"   : PUCT tree search.            fields: name, evaluator, n_sims, c_puct,
#                                                   add_noise, temperature, temp_turns
#
# -------------------------------------------------------------------------------
# MCTS AGENT SPEC — every field, what it does, and its default (used if omitted):
# -------------------------------------------------------------------------------
#
#   "name": str
#       Display name for this agent. Shown in the per-game lines and the win tally.
#       Purely cosmetic — make the two agents' names distinct so results read cleanly.
#
#   "type": "mcts"
#       Selects the PUCT search strategy (mcts.MCTS wrapped by arena.MCTSStrategy).
#
#   "evaluator": "heuristic" | "network"        (default "heuristic")
#       What scores a leaf and supplies the per-stage priors inside the search:
#         "heuristic" — value = tanh(0.1 * (heuristic_score(me) - heuristic_score(opp))),
#                       priors uniform over legal choices. Fast, no torch weights needed.
#                       Good for exercising search / a baseline opponent.
#         "network"   — PolysharkNet value head + PolicyHead priors (the real model).
#                       Random-init weights unless you load a checkpoint in build_agent().
#                       Slower (a torch forward per new leaf state).
#
#   "n_sims": int                               (default 100)
#       Simulations run per decision — the search budget. Each sim descends the factored
#       tree, evaluates one new leaf, and backs the value up. Higher = stronger and slower
#       (cost is ~linear in n_sims). 20-60 is snappy; 100-400 is the doc's self-play range.
#
#   "c_puct": float                             (default 1.5)
#       PUCT exploration constant. Weights the exploration term
#       c_puct * prior * sqrt(sumN)/(1+N) against the exploitation term Q. Higher pushes
#       the search to try high-prior / low-visit moves more; lower trusts current Q sooner.
#       Typical 1.5-2.5.
#
#   "add_noise": bool                           (default False)
#       Mix Dirichlet noise into the ROOT priors so repeated searches explore different
#       first moves. Turn ON for self-play data generation (diversity), OFF for evaluation
#       or head-to-head (you want the agent's honest best move).
#
#   "temperature": float                        (default 1.0)
#       Controls how the played action is picked from the root visit counts, during the
#       opening (see temp_turns):
#         1.0 -> sample an action in proportion to its visit count (exploratory).
#         0.0 -> pick the most-visited action (greedy / argmax).
#       Values in between interpolate (counts ** (1/temperature)).
#
#   "temp_turns": int                           (default 6)
#       How many game turns the `temperature` above applies for. Once state.get_turn()
#       reaches this, temperature drops to 0.0 (greedy) for the rest of the game. Mirrors
#       AlphaZero's "explore the opening, then play greedily."
#
#   "checkpoint": str   (optional, evaluator="network" only)
#       Path to trained weights, RELATIVE TO THE REPO ROOT (absolute paths also work),
#       e.g. "ai/mctsnn-v2/data/checkpoints/gen005.pt". Loaded by _make_network_evaluator.
#       Expected format: torch.save({"net": net.state_dict(), "policy": policy.state_dict()}).
#       If omitted, the network uses random-init weights (prints a warning; plays badly).
#
# Presets:
#   - self-play data:  both agents "mcts", add_noise=True,  temperature=1.0
#   - evaluation:      both agents "mcts", add_noise=False, temperature=0.0 (greedy)

AGENT_0 = {
    "name": "agent 0",
    "type": "mcts",
    "evaluator": "network",
    "checkpoint": "ai/mctsnn-v2/data/checkpoints/gen019.pt",
    "n_sims": 1000,
    "c_puct": 1.5,
    "add_noise": False,
    "temperature": 0.0,
    "temp_turns": 6,
}

AGENT_1 = {
    "name": "agent 1",
    "type": "mcts",
    "evaluator": "network",
    "checkpoint": "ai/mctsnn-v2/data/checkpoints/gen019.pt",
    "n_sims": 1000,
    "c_puct": 1.5,
    "add_noise": False,
    "temperature": 0.0,
    "temp_turns": 6,
}

# AGENT_1 = {
#     "name": "random",
#     "type": "random",
#     "seed": 0,
# }

N_GAMES     = 1          # how many games to play
TURN_LIMIT  = 30         # max turns before a game is called (turn cap -> heuristic outcome)
BASE_SEED   = 100        # game i uses map seed BASE_SEED + i
SWAP_SIDES  = True       # alternate who is player 0 across games (fairness)
MAX_STEPS   = 4000       # hard safety cap on actions per game

COLLECT_SAMPLES = False   # record (state, action, policy targets, outcome) training rows
WRITE_REPLAYS   = True    # dump a .replay per game for the visualiser
REPLAY_DIR      = os.path.join(_PKG, "data", "runs")
VERBOSE         = True    # per-game one-line summary

# ============================================================================
# end CONFIG
# ============================================================================


def main():
    agent0 = build_agent(AGENT_0, _ROOT)
    agent1 = build_agent(AGENT_1, _ROOT)

    wins = {agent0.name: 0, agent1.name: 0}
    draws, all_turns, all_samples = 0, [], []

    print(f"{agent0.name}  vs  {agent1.name}   |   {N_GAMES} games, turn_limit={TURN_LIMIT}\n")

    for i in range(N_GAMES):
        # Swap seating on odd games so neither agent always plays player 0.
        swapped = SWAP_SIDES and (i % 2 == 1)
        order = [agent1, agent0] if swapped else [agent0, agent1]

        res = Arena(order).play_game(
            seed=BASE_SEED + i,
            max_turns=TURN_LIMIT,
            max_steps=MAX_STEPS,
            collect=COLLECT_SAMPLES,
        )

        # Tally by agent name (res.winner is an index into `order`).
        if res.winner >= 0:
            wins[order[res.winner].name] += 1
            outcome = f"{order[res.winner].name} wins"
        else:
            draws += 1
            outcome = "turn cap"
        all_turns.append(res.turns)
        all_samples.extend(res.samples)

        if WRITE_REPLAYS:
            os.makedirs(REPLAY_DIR, exist_ok=True)
            path = os.path.join(REPLAY_DIR, f"game{i:02d}_seed{res.seed}.replay")
            write_replay(path, res)

        if VERBOSE:
            p0, p1 = order[0].name, order[1].name
            print(f"game {i:2d}  seed={res.seed:<5}  {p0} vs {p1:<8}  "
                  f"{outcome:<14} turns={res.turns:<3} actions={len(res.history):<4} "
                  f"v=({res.v_finals[0]:+.2f},{res.v_finals[1]:+.2f})")

    avg_turns = sum(all_turns) / len(all_turns) if all_turns else 0.0
    n_policy = sum(1 for s in all_samples if s.targets)

    print("\n--- summary ---")
    for name, w in wins.items():
        print(f"  {name:<12} wins: {w}/{N_GAMES}  ({w / N_GAMES:.0%})")
    print(f"  draws (turn cap): {draws}")
    print(f"  avg turns/game:   {avg_turns:.1f}")
    if COLLECT_SAMPLES:
        print(f"  samples collected: {len(all_samples)} ({n_policy} with policy targets)")
    if WRITE_REPLAYS:
        print(f"  replays written to: {REPLAY_DIR}")


if __name__ == "__main__":
    main()
