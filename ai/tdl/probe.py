"""
Feature importance probe for a trained ValueNet checkpoint.

Two methods:
  - Gradient saliency: mean |dV/d_input| per channel
  - Occlusion:         zero out one channel, measure mean |output change|

Usage:
  python probe.py checkpoints/gen_010.pt
  python probe.py checkpoints/gen_010.pt --games 50 --method occlusion
"""

import sys, os, argparse
import numpy as np
import torch

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, '../../build/bindings'))

import polyshark
from encoder import encode_batch, C_IN, G
from model   import ValueNet

SPATIAL_NAMES = [
    "is_explored",                                              # 0
    "terrain_Field", "terrain_Forest", "terrain_Mountain",     # 1-3
    "terrain_Water", "terrain_Village",                        # 4-5
    "res_Fruit", "res_Crop", "res_Animal", "res_Metal",        # 6-9
    "my_Mine", "opp_Mine", "my_Farm", "opp_Farm",              # 10-13
    "my_Road", "opp_Road",                                     # 14-15
    "my_border", "opp_border",                                 # 16-17
    "my_has_unit",                                             # 18
    "my_unit_Warrior", "my_unit_Archer", "my_unit_Rider",      # 19-21
    "my_unit_Defender", "my_unit_Giant",                       # 22-23
    "my_unit_hp", "my_unit_attacked", "my_unit_move",          # 24-26
    "my_unit_kills", "my_unit_promo",                          # 27-28
    "opp_has_unit",                                            # 29
    "opp_unit_Warrior", "opp_unit_Archer", "opp_unit_Rider",   # 30-32
    "opp_unit_Defender", "opp_unit_Giant",                     # 33-34
    "opp_unit_hp", "opp_unit_attacked", "opp_unit_move",       # 35-37
    "opp_unit_kills", "opp_unit_promo",                        # 38-39
    "my_has_city", "my_city_capital", "my_city_level",         # 40-42
    "my_city_pop", "my_city_units", "my_city_sieged",          # 43-45
    "my_city_walls", "my_city_workshop", "my_city_pending",    # 46-48
    "my_city_capture",                                         # 49
    "opp_has_city", "opp_city_capital", "opp_city_level",      # 50-52
    "opp_city_pop", "opp_city_units", "opp_city_sieged",       # 53-55
    "opp_city_walls", "opp_city_workshop", "opp_city_pending", # 56-58
    "opp_city_capture",                                        # 59
]

GVEC_NAMES = [
    "turn", "my_stars", "my_income",
    "tech_Hunting", "tech_Organisation", "tech_Farming", "tech_Riding",
    "tech_Climbing", "tech_Archery", "tech_Mining", "tech_Strategy",
]


def collect_states(n_games, n_steps=10):
    states = []
    for seed in range(n_games):
        s = polyshark.make_random_game(seed=seed)
        for _ in range(n_steps):
            if s.is_terminal():
                break
            actions = s.legal_actions()
            affordable = [a for a in actions if a.affordable]
            if not affordable:
                break
            s = s.apply_action(affordable[np.random.randint(len(affordable))])
        states.append(s)
    return states


def saliency(model, sp_np, gv_np, device):
    sp = torch.from_numpy(sp_np).to(device).requires_grad_(True)
    gv = torch.from_numpy(gv_np).to(device).requires_grad_(True)

    val = model(sp, gv)
    val.sum().backward()

    sp_grad = sp.grad.abs().mean(dim=(0, 2, 3)).cpu().numpy()  # [C_IN]
    gv_grad = gv.grad.abs().mean(dim=0).cpu().numpy()          # [G]
    return sp_grad, gv_grad


def occlusion(model, sp_np, gv_np, device):
    """
    Zero out one channel at a time and measure mean absolute output change.
    More direct than gradient saliency: tells you how much the model actually
    relies on each channel, not just how sensitive gradients are.
    """
    sp = torch.from_numpy(sp_np).to(device)
    gv = torch.from_numpy(gv_np).to(device)

    with torch.no_grad():
        baseline = model(sp, gv).squeeze(-1)  # [N]

    sp_imp = np.zeros(C_IN,  dtype=np.float32)
    gv_imp = np.zeros(G,     dtype=np.float32)

    with torch.no_grad():
        for c in range(C_IN):
            sp_occ      = sp.clone()
            sp_occ[:, c, :, :] = 0.0
            delta       = (model(sp_occ, gv).squeeze(-1) - baseline).abs()
            sp_imp[c]   = delta.mean().item()

        for g in range(G):
            gv_occ      = gv.clone()
            gv_occ[:, g] = 0.0
            delta       = (model(sp, gv_occ).squeeze(-1) - baseline).abs()
            gv_imp[g]   = delta.mean().item()

    return sp_imp, gv_imp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt')
    parser.add_argument('--games',  type=int,   default=30)
    parser.add_argument('--top',    type=int,   default=15)
    parser.add_argument('--method', type=str,   default='saliency',
                        choices=['saliency', 'occlusion'])
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')

    model = ValueNet().to(device)
    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f'loaded {args.ckpt}  device={device}  method={args.method}')
    print(f'collecting states from {args.games} games...')
    states = collect_states(args.games)
    sp_np, gv_np = encode_batch(states)

    if args.method == 'occlusion':
        sp_imp, gv_imp = occlusion(model, sp_np, gv_np, device)
        label = 'mean |output change| on occlusion'
    else:
        sp_imp, gv_imp = saliency(model, sp_np, gv_np, device)
        label = 'mean |dV/d_input|'

    sp_rank = np.argsort(sp_imp)[::-1]
    print(f'\n--- Top {args.top} spatial channels ({label}) ---')
    for i in sp_rank[:args.top]:
        print(f'  ch {i:2d}  {SPATIAL_NAMES[i]:<30s}  {sp_imp[i]:.5f}')

    gv_rank = np.argsort(gv_imp)[::-1]
    print(f'\n--- Global vector importances ({label}) ---')
    for i in gv_rank:
        print(f'  g[{i:2d}]  {GVEC_NAMES[i]:<25s}  {gv_imp[i]:.5f}')


if __name__ == '__main__':
    main()
