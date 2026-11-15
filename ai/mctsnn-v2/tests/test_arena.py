"""
Checks for the agent / strategy / arena layer. Run:

    source .venv/bin/activate
    python ai/mctsnn-v2/tests/test_arena.py

Proves games terminate, every recorded sample is outcome-labelled and consistent with the
result, side-swapping is fair, and MCTS samples carry policy targets while random ones don't.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../build/bindings"))
import polyshark  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from arena import (  # noqa: E402
    Agent, Arena, RandomStrategy, MCTSStrategy, legal_actions,
)
from mcts import HeuristicEvaluator  # noqa: E402


def test_game_terminates_and_labels():
    a = Agent("A", RandomStrategy(seed=1))
    b = Agent("B", RandomStrategy(seed=2))
    res = Arena([a, b]).play_game(seed=7, max_turns=25)

    assert res.reason in ("capital", "turn_cap", "step_cap"), res.reason
    assert res.samples, "no samples recorded"
    # Every sample labelled and in range.
    for s in res.samples:
        assert s.outcome is not None, "unlabelled sample"
        assert -1.0 <= s.outcome <= 1.0, s.outcome
        assert s.targets is None, "random strategy should emit no policy targets"
    # Winner consistency: decisive => outcomes are exactly +/-1 and mirror the winner.
    if res.winner >= 0:
        for s in res.samples:
            assert s.outcome == (1.0 if s.player == res.winner else -1.0)
    print(f"[label] game: reason={res.reason} winner={res.winner} "
          f"samples={len(res.samples)} all labelled OK")


def test_no_collect():
    a = Agent("A", RandomStrategy(seed=1))
    b = Agent("B", RandomStrategy(seed=2))
    res = Arena([a, b]).play_game(seed=3, max_turns=15, collect=False)
    assert res.samples == [], "collect=False should record nothing"
    print("[collect] collect=False records no samples OK")


def test_match_aggregation_and_swap():
    a = Agent("A", RandomStrategy(seed=1))
    b = Agent("B", RandomStrategy(seed=2))
    stats = Arena([a, b]).play_matches(6, base_seed=50, max_turns=15, swap_sides=True)
    assert stats.games == 6
    assert set(stats.wins) == {"A", "B"}, stats.wins
    assert stats.wins["A"] + stats.wins["B"] + stats.draws == 6, "games unaccounted for"
    assert 0.0 <= stats.win_rate("A") <= 1.0
    assert len(stats.samples) > 0
    print(f"[match] {stats} OK")


def test_mcts_strategy_targets():
    mcts_agent = Agent("mcts", MCTSStrategy(HeuristicEvaluator(), n_sims=12, add_noise=False))
    rand_agent = Agent("rand", RandomStrategy(seed=4))
    res = Arena([mcts_agent, rand_agent]).play_game(seed=11, max_turns=8)

    mcts_samples = [s for s in res.samples if s.player == 0]
    rand_samples = [s for s in res.samples if s.player == 1]
    assert any(s.targets for s in mcts_samples), "MCTS agent produced no policy targets"
    assert all(s.targets is None for s in rand_samples), "random agent leaked targets"
    # Policy targets are per-stage (stage_name, {choice: visits}); the first stage is the
    # phase entry stage — "type" (Idle) or "upgrade" (UpgradingCity modal).
    for s in mcts_samples:
        if s.targets:
            assert s.targets[0][0] in ("type", "upgrade"), s.targets[0]
    print(f"[mcts] {len(mcts_samples)} mcts samples w/ targets, "
          f"{len(rand_samples)} random w/o OK")


def test_legal_action_space_excludes_debug():
    state = polyshark.make_random_game(1)
    acts = legal_actions(state)
    assert all(a.type != polyshark.ActionType.DebugAddPop for a in acts), "DebugAddPop leaked"
    assert acts, "no legal actions"
    print(f"[space] legal action space excludes debug ({len(acts)} actions) OK")


if __name__ == "__main__":
    test_game_terminates_and_labels()
    test_no_collect()
    test_match_aggregation_and_swap()
    test_mcts_strategy_targets()
    test_legal_action_space_excludes_debug()
    print("\nAll arena checks passed.")
