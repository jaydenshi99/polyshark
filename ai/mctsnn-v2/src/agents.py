"""
Agent construction from spec dicts — shared by scripts/run_games.py (human eval) and
scripts/train.py (the training loop). Keeping this here (not in a script) means both
entry points build agents and load checkpoints the same way, with one place to fix.

A spec is a plain dict; see run_games.py's CONFIG for the documented fields.
`root` is the repo root, used to resolve relative checkpoint paths.
"""

import os

from arena import Agent, RandomStrategy, MCTSStrategy
from mcts import HeuristicEvaluator, NetworkEvaluator


def resolve_checkpoint(path, root):
    """Absolute path stays as-is; a relative path is resolved from the repo root."""
    return path if os.path.isabs(path) else os.path.join(root, path)


def make_network_evaluator(spec, root):
    """Build a NetworkEvaluator, loading a checkpoint if `spec["checkpoint"]` is set.

    Checkpoint format: a torch-saved dict with "net" and/or "policy" state_dicts, i.e.
        torch.save({"net": net.state_dict(), "policy": policy.state_dict()}, path)
    A bare state_dict (no "net"/"policy" keys) is loaded into the value net only.
    """
    import torch  # local import so heuristic/random runs stay torch-free
    evaluator = NetworkEvaluator()

    ckpt = spec.get("checkpoint")
    if not ckpt:
        print(f"[{spec['name']}] WARNING: 'network' with random-init weights "
              f"(no 'checkpoint' in spec) — plays badly. Use 'heuristic' or add a checkpoint.")
        return evaluator

    full = resolve_checkpoint(ckpt, root)
    if not os.path.exists(full):
        raise FileNotFoundError(f"[{spec['name']}] checkpoint not found: {full}")
    blob = torch.load(full, map_location="cpu")
    if isinstance(blob, dict) and ("net" in blob or "policy" in blob):
        if "net" in blob:
            evaluator.net.load_state_dict(blob["net"])
        if "policy" in blob:
            evaluator.policy.load_state_dict(blob["policy"])
    else:
        evaluator.net.load_state_dict(blob)          # bare value-net state_dict
    evaluator.net.eval()
    evaluator.policy.eval()
    print(f"[{spec['name']}] loaded checkpoint: {full}")
    return evaluator


def build_agent(spec, root):
    """Turn a spec dict into an Agent. Extend here to add strategy types."""
    kind = spec["type"]

    if kind == "random":
        return Agent(spec["name"], RandomStrategy(seed=spec.get("seed")))

    if kind == "mcts":
        ev_name = spec.get("evaluator", "heuristic")
        if ev_name == "heuristic":
            evaluator = HeuristicEvaluator()
        elif ev_name == "network":
            evaluator = make_network_evaluator(spec, root)
        else:
            raise ValueError(f"unknown evaluator {ev_name!r}")
        return Agent(spec["name"], MCTSStrategy(
            evaluator,
            n_sims=spec.get("n_sims", 100),
            c_puct=spec.get("c_puct", 1.5),
            add_noise=spec.get("add_noise", False),
            temperature=spec.get("temperature", 1.0),
            temp_turns=spec.get("temp_turns", 6),
        ))

    raise ValueError(f"unknown agent type {kind!r}")
