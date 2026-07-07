"""
Shape + sanity checks for the entity encoder. Run:

    source .venv/bin/activate
    python ai/mctsnn-v2/test_shapes.py

Drives real make_random_game states through features -> model, and asserts
shapes, mask correctness, and gradient flow.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/bindings"))
import polyshark  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from features import encode_entities, collate, UNIT_FEAT_DIM, CITY_FEAT_DIM  # noqa: E402
from model import EntityEncoder, D_MODEL  # noqa: E402


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
    e = encode_entities(polyshark.make_random_game(1))
    assert e["unit_feats"].ndim == 2 and e["unit_feats"].shape[1] == UNIT_FEAT_DIM
    assert e["city_feats"].ndim == 2 and e["city_feats"].shape[1] == CITY_FEAT_DIM
    assert e["unit_types"].shape[0] == e["unit_feats"].shape[0]
    assert np.isfinite(e["unit_feats"]).all() and np.isfinite(e["city_feats"]).all()
    print(f"[single] units={e['unit_types'].shape[0]} cities={e['city_feats'].shape[0]} OK")


def test_batch_forward():
    states = _rollout_states(seed=7, n=6)
    batch = collate(states)
    B = len(states)
    Lu, Lc = batch["unit_mask"].shape[1], batch["city_mask"].shape[1]

    tensors = {k: torch.from_numpy(v) for k, v in batch.items()}
    model = EntityEncoder()

    tokens, real_mask = model(
        tensors["unit_types"], tensors["unit_feats"], tensors["unit_mask"],
        tensors["city_feats"], tensors["city_mask"],
    )

    assert tokens.shape == (B, Lu + Lc, D_MODEL), tokens.shape
    assert real_mask.shape == (B, Lu + Lc)
    assert torch.isfinite(tokens).all(), "non-finite tokens (masking/softmax NaN?)"
    # Every state must have >= 1 real entity (else its query rows all-mask -> NaN).
    assert real_mask.any(dim=1).all(), "a batch element has zero real entities"
    print(f"[batch]  B={B} L={Lu + Lc} tokens={tuple(tokens.shape)} finite OK")

    # Gradient flow: masked-mean over real tokens -> scalar -> backward.
    masked = tokens * real_mask.unsqueeze(-1)
    pooled = masked.sum(dim=1) / real_mask.sum(dim=1, keepdim=True)
    loss = pooled.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    total = sum(g.abs().sum().item() for g in grads)
    assert total > 0, "zero gradient — nothing learned"
    print(f"[grad]   loss={loss.item():.4f} grad_sum={total:.2f} flows OK")


if __name__ == "__main__":
    test_single_encode()
    test_batch_forward()
    print("\nAll checks passed.")
