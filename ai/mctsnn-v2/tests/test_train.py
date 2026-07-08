"""
Smoke test for the Phase A trainer (see docs/training.md). Run:

    source .venv/bin/activate
    python ai/mctsnn-v2/tests/test_train.py

Runs a tiny end-to-end training loop (a couple of gens, few games, few steps) against the
real engine and asserts: the buffer fills, losses are finite, both value and policy losses
flow gradient, a checkpoint is written in the expected format, and it reloads into fresh
modules. Kept small so it runs in a few seconds.
"""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../build/bindings"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from replay_buffer import ReplayBuffer  # noqa: E402
from trainer import run_training, train_step  # noqa: E402
from model import PolysharkNet  # noqa: E402
from policy import PolicyHead  # noqa: E402
from arena import Arena, Agent, MCTSStrategy  # noqa: E402
from mcts import HeuristicEvaluator  # noqa: E402


def test_replay_buffer_ring():
    buf = ReplayBuffer(capacity=5)
    buf.extend(list(range(3)))
    assert len(buf) == 3
    buf.extend(list(range(10)))                 # overflow -> keeps last 5
    assert len(buf) == 5
    assert set(buf.buf) == {5, 6, 7, 8, 9}
    drawn = buf.sample(4)
    assert len(drawn) == 4 and all(x in buf.buf for x in drawn)
    assert ReplayBuffer(3).sample(2) == []      # empty buffer
    print("[buffer] FIFO ring + sampling OK")


def test_train_step_grads():
    """One train_step over real self-play samples: finite losses, grad reaches both heads."""
    ev = HeuristicEvaluator()
    a = Agent("p0", MCTSStrategy(ev, n_sims=8, add_noise=True))
    b = Agent("p1", MCTSStrategy(ev, n_sims=8, add_noise=True))
    res = Arena([a, b]).play_game(seed=1, max_turns=5, collect=True)
    batch = res.samples[:8]
    assert batch, "no samples produced"

    net, policy = PolysharkNet(), PolicyHead()
    net.train(); policy.train()
    opt = torch.optim.Adam(list(net.parameters()) + list(policy.parameters()), lr=1e-3)
    vloss, ploss = train_step(net, policy, opt, batch)

    assert np.isfinite(vloss) and np.isfinite(ploss), (vloss, ploss)
    # Both a value-head param and a policy-head param received gradient.
    vgrad = sum(p.grad.abs().sum().item() for p in net.value_head.parameters() if p.grad is not None)
    pgrad = sum(p.grad.abs().sum().item() for p in policy.type_head.parameters() if p.grad is not None)
    assert vgrad > 0, "value head got no gradient"
    assert pgrad > 0, "policy head got no gradient"
    print(f"[step]  value_loss={vloss:.4f} policy_loss={ploss:.4f}, both heads flow OK")


def test_run_training_end_to_end():
    """A tiny full loop: buffer fills, losses finite, checkpoint written and reloadable."""
    with tempfile.TemporaryDirectory() as tmp:
        history, net, policy = run_training(
            n_gens=2, games_per_gen=2, train_steps_per_gen=5, minibatch=8,
            buffer_capacity=5000, turn_limit=5, n_sims=8, base_seed=100,
            bootstrap_gen0=True, ckpt_dir=tmp, log=lambda *a, **k: None,
        )
        assert len(history) == 2
        for h in history:
            assert h["buffer"] > 0, "buffer never filled"
            assert np.isfinite(h["value_loss"]), h
            assert np.isfinite(h["policy_loss"]), h
        # gen 0 bootstraps on the heuristic; gen 1 uses the network.
        assert history[0]["eval"] == "heuristic", history[0]["eval"]
        assert history[1]["eval"] == "network", history[1]["eval"]

        # Checkpoint written in the {"net","policy"} format and reloadable into fresh modules.
        ckpt = history[-1]["ckpt"]
        assert ckpt and os.path.exists(ckpt), ckpt
        blob = torch.load(ckpt, map_location="cpu")
        assert "net" in blob and "policy" in blob, list(blob)
        PolysharkNet().load_state_dict(blob["net"])
        PolicyHead().load_state_dict(blob["policy"])
        # metrics.csv written with a row per generation.
        csv_path = os.path.join(tmp, "metrics.csv")
        assert os.path.exists(csv_path), "metrics.csv not written"
        with open(csv_path) as f:
            rows = f.read().splitlines()
        assert rows[0].startswith("gen,"), rows[0]
        assert len(rows) == 1 + 2, rows          # header + 2 gens
        print(f"[loop]  2 gens ran, losses finite, checkpoint reloads, metrics.csv OK "
              f"(v={history[-1]['value_loss']:.3f} p={history[-1]['policy_loss']:.3f})")


def test_parallel_selfplay():
    """num_workers>1 runs self-play across processes and returns the same well-formed data."""
    with tempfile.TemporaryDirectory() as tmp:
        history, _, _ = run_training(
            n_gens=2, games_per_gen=4, train_steps_per_gen=3, minibatch=8,
            buffer_capacity=5000, turn_limit=5, n_sims=8, base_seed=200,
            bootstrap_gen0=True, num_workers=2, ckpt_dir=tmp, log=lambda *a, **k: None,
        )
        assert len(history) == 2
        for h in history:
            assert h["buffer"] > 0, "buffer never filled (workers returned nothing?)"
            assert np.isfinite(h["value_loss"]) and np.isfinite(h["policy_loss"]), h
        assert history[0]["eval"] == "heuristic" and history[1]["eval"] == "network"
        print(f"[parallel] 2 workers x 2 gens ran, samples crossed processes OK "
              f"(buffer={history[-1]['buffer']})")


if __name__ == "__main__":
    test_replay_buffer_ring()
    test_train_step_grads()
    test_run_training_end_to_end()
    test_parallel_selfplay()
    print("\nAll trainer checks passed.")
