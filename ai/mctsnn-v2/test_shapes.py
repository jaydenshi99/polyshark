"""
Shape + sanity checks for the full state->value net. Run:

    source .venv/bin/activate
    python ai/mctsnn-v2/test_shapes.py

Drives real make_random_game states through features -> PolysharkNet, asserting
input shapes, a finite value in [-1,1], and gradient flow to every parameter group.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/bindings"))
import polyshark  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from features import (  # noqa: E402
    encode_entities, encode_board, encode_globals, collate,
    UNIT_FEAT_DIM, CITY_FEAT_DIM, BOARD_CHANNELS, GLOBAL_DIM,
)
from model import PolysharkNet  # noqa: E402

_MODEL_ARG_ORDER = [
    "unit_types", "unit_feats", "unit_mask", "unit_tiles",
    "city_feats", "city_mask", "city_tiles", "board", "globals",
]


def _rollout_states(seed, n):
    """A few states from one game, so entity counts differ across the batch."""
    state = polyshark.make_random_game(seed)
    states = [state]
    while len(states) < n and not state.is_terminal():
        legal = [a for a in state.legal_actions() if a.affordable]
        if not legal:
            break
        state = state.apply_action(legal[len(states) % len(legal)])
        states.append(state)
    return states


def test_single_encode():
    s = polyshark.make_random_game(1)
    e = encode_entities(s)
    assert e["unit_feats"].shape[1] == UNIT_FEAT_DIM
    assert e["city_feats"].shape[1] == CITY_FEAT_DIM
    assert e["unit_tiles"].shape[0] == e["unit_types"].shape[0]
    assert e["city_tiles"].shape[0] == e["city_feats"].shape[0]

    sz = s.map_size()
    board = encode_board(s)
    assert board.shape == (BOARD_CHANNELS, sz, sz)
    g = encode_globals(s)
    assert g.shape == (GLOBAL_DIM,)
    assert np.isfinite(board).all() and np.isfinite(g).all()
    print(f"[single] units={e['unit_types'].shape[0]} cities={e['city_feats'].shape[0]} "
          f"board={board.shape} globals={g.shape} OK")


def test_full_forward():
    states = _rollout_states(seed=7, n=6)
    batch = collate(states)
    B = len(states)
    tensors = {k: torch.from_numpy(v) for k, v in batch.items()}

    model = PolysharkNet()
    value = model(*(tensors[k] for k in _MODEL_ARG_ORDER))

    assert value.shape == (B, 1), value.shape
    assert torch.isfinite(value).all(), "non-finite value (scatter/mask NaN?)"
    assert (value >= -1).all() and (value <= 1).all(), "value out of tanh range"
    print(f"[forward] B={B} value{tuple(value.shape)} range=[{value.min():.3f},{value.max():.3f}] OK")

    # Gradient flow: pretend targets, MSE, backward.
    target = torch.zeros(B, 1)
    loss = torch.nn.functional.mse_loss(value, target)
    loss.backward()

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_grad = [n for n, p in named if p.grad is None]
    assert not no_grad, f"params with no grad: {no_grad[:5]}"
    total = sum(p.grad.abs().sum().item() for _, p in named)
    assert total > 0 and np.isfinite(total), "bad gradient"
    # Confirm every major component actually received gradient.
    for comp in ("entity", "unit_scatter", "city_scatter", "stem", "blocks", "global_mlp", "trunk", "value_head"):
        g = sum(p.grad.abs().sum().item() for n, p in named if n.startswith(comp))
        assert g > 0, f"no gradient reached {comp}"
    print(f"[grad] loss={loss.item():.4f} grad_sum={total:.1f} — all components flow OK")


if __name__ == "__main__":
    test_single_encode()
    test_full_forward()
    print("\nAll checks passed.")
