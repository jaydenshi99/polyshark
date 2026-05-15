import sys
import os
import math
import random
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/bindings"))
import polyshark

sys.path.insert(0, os.path.dirname(__file__))
from model import PolysharkNet, ACTION_SIZE
from mcts import MCTS
from encoder import encode
from action_codec import index_to_action
from log_utils import format_action, log_state

# --- Hyperparameters ---
TURN_LIMIT         = 10
GAMES_PER_GEN      = 20
TRAIN_STEPS        = 500
BUFFER_SIZE        = 50_000
BATCH_SIZE         = 256
LR                 = 1e-3
N_SIMULATIONS      = 50
TEMPERATURE_CUTOFF = 6    # use temp=1.0 for first 6 turns, then greedy
HEURISTIC_DECAY_GENS = 100  # heuristic weight decays from 1.0 → 0.0 over this many gens

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "../../checkpoints")
REPLAYS_DIR    = os.path.join(os.path.dirname(__file__), "../../replays")


# --- Heuristic ---

def heuristic_value(state, player):
    opp = 1 - player
    sz  = state.map_size()

    my_income = my_cities = my_levels = my_units = 0
    op_income = op_cities = op_levels = op_units = 0

    for i in range(sz * sz):
        tile = state.tile_at(i)
        if tile.has_city:
            city = state.get_city(tile.city_id)
            if city.owner == player:
                my_income += city.stars_per_turn
                my_cities += 1
                my_levels += city.level
            else:
                op_income += city.stars_per_turn
                op_cities += 1
                op_levels += city.level
        if tile.has_unit:
            unit = state.get_unit(tile.unit_id)
            if unit.is_alive:
                if unit.owner == player:
                    my_units += 1
                else:
                    op_units += 1

    my_techs = bin(state.get_techs(player)).count('1')
    op_techs = bin(state.get_techs(opp)).count('1')

    score = (
        2.0 * (my_income - op_income) +
        1.5 * (my_cities - op_cities) +
        1.0 * (my_levels - op_levels) +
        0.8 * (my_techs  - op_techs)  +
        0.2 * (my_units  - op_units)
    )
    return math.tanh(score * 0.3)


# --- Replay Buffer ---

class ReplayBuffer:
    def __init__(self, maxlen=BUFFER_SIZE):
        self._buf = deque(maxlen=maxlen)

    def add(self, spatial, global_vec, policy, outcome: float):
        self._buf.append((spatial, global_vec, policy, np.float32(outcome)))

    def sample(self, batch_size, device):
        batch    = random.sample(self._buf, min(batch_size, len(self._buf)))
        spatials = torch.tensor(np.stack([x[0] for x in batch])).to(device)
        globals_ = torch.tensor(np.stack([x[1] for x in batch])).to(device)
        policies = torch.tensor(np.stack([x[2] for x in batch])).to(device)
        outcomes = torch.tensor(np.array([x[3] for x in batch])).to(device)
        return spatials, globals_, policies, outcomes

    def __len__(self):
        return len(self._buf)


# --- Replay ---

def save_replay(history, seed, gen, game_idx):
    os.makedirs(REPLAYS_DIR, exist_ok=True)
    path = os.path.join(REPLAYS_DIR, f"gen{gen:04d}_game{game_idx:03d}.replay")
    with open(path, "w") as f:
        f.write(f"seed {seed}\n")
        for a in history:
            f.write(f"{int(a.type)} {a.src} {a.dst} {a.param}\n")


# --- Self-play ---

def run_game(mcts, gen=0, game_idx=0) -> tuple[list, bool]:
    """
    Play one game. Returns (examples, terminal) where:
      examples  — list of (spatial, global_vec, policy, outcome)
      terminal  — True if game ended by capture, False if turn limit hit
    """
    seed       = random.randint(0, 2**32 - 1)
    state      = polyshark.make_random_game(seed)
    trajectory = []  # (spatial, global_vec, policy, current_player)
    history    = []  # raw actions for replay file

    print(f"\n--- Gen {gen} Game {game_idx} ---")
    last_logged_turn = -1

    while not state.is_terminal():
        if state.get_turn() >= TURN_LIMIT:
            break

        player = state.current_player()
        turn   = state.get_turn()

        if turn != last_logged_turn:
            print(f"  Turn {turn}")
            log_state(state)
            last_logged_turn = turn

        root   = mcts.search(state, n_simulations=N_SIMULATIONS, add_noise=True)
        temp   = 1.0 if turn < TEMPERATURE_CUTOFF else 0.0
        policy = mcts.get_policy(root, temperature=temp)

        spatial, global_vec = encode(state)
        trajectory.append((spatial, global_vec, policy, player))

        action_idx = int(np.random.choice(ACTION_SIZE, p=policy))
        action     = index_to_action(action_idx, state)
        print(f"    P{player}: {format_action(action)}")
        history.append(action)
        state      = state.apply_action(action)

    terminal = state.is_terminal()

    if terminal:
        winner   = state.winner()
        outcomes = [1.0 if p == winner else -1.0 for _, _, _, p in trajectory]
        print(f"  -> P{winner} wins")
    else:
        outcomes = [heuristic_value(state, p) for _, _, _, p in trajectory]
        h0 = heuristic_value(state, 0)
        print(f"  -> Turn limit hit | heuristic P0={h0:.2f} P1={-h0:.2f}")

    examples = [(s, g, pol, out) for (s, g, pol, _), out in zip(trajectory, outcomes)]
    save_replay(history, seed, gen, game_idx)
    return examples, terminal


# --- Training ---

def train_step(model, optimizer, buffer, device):
    if len(buffer) < BATCH_SIZE:
        return None, None

    spatial, global_vec, target_policy, target_value = buffer.sample(BATCH_SIZE, device)

    model.train()
    policy, value = model(spatial, global_vec)

    policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
    value_loss  = F.mse_loss(value.squeeze(-1), target_value)
    loss        = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.eval()

    return policy_loss.item(), value_loss.item()



# --- Checkpointing ---

def save_checkpoint(model, buffer, gen):
    import pickle
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"model_{gen:04d}.pt")
    torch.save({'gen': gen, 'model': model.state_dict()}, path)
    buf_path = os.path.join(CHECKPOINT_DIR, "buffer.pkl")
    with open(buf_path, 'wb') as f:
        pickle.dump(buffer._buf, f)
    return path


def load_latest_checkpoint(model, buffer):
    import pickle
    if not os.path.exists(CHECKPOINT_DIR):
        return 0
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
    if not files:
        return 0
    latest = max(files, key=lambda f: int(f.split('_')[1].split('.')[0]))
    ckpt   = torch.load(os.path.join(CHECKPOINT_DIR, latest), weights_only=True)
    model.load_state_dict(ckpt['model'])
    buf_path = os.path.join(CHECKPOINT_DIR, "buffer.pkl")
    if os.path.exists(buf_path):
        with open(buf_path, 'rb') as f:
            buffer._buf = pickle.load(f)
        print(f"Resumed buffer ({len(buffer)} positions)")
    print(f"Resumed from generation {ckpt['gen']}")
    return ckpt['gen']


# --- Main loop ---

def train(n_generations=200):
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model     = PolysharkNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    buffer    = ReplayBuffer()
    mcts      = MCTS(model, device=device, heuristic_fn=heuristic_value, heuristic_weight=1.0)

    start_gen = load_latest_checkpoint(model, buffer)

    log_path = os.path.join(CHECKPOINT_DIR, "training_log.csv")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    log_exists = os.path.exists(log_path)
    log_file = open(log_path, "a")
    if not log_exists:
        log_file.write("gen,terminals,positions,buffer,policy_loss,value_loss,heuristic_weight\n")
        log_file.flush()

    for gen in range(start_gen, start_gen + n_generations):
        hw = max(0.0, 1.0 - gen / HEURISTIC_DECAY_GENS)
        mcts.heuristic_weight = hw

        # Self-play phase
        terminals = 0
        positions = 0
        for game_idx in range(GAMES_PER_GEN):
            examples, terminal = run_game(mcts, gen=gen + 1, game_idx=game_idx)
            for example in examples:
                buffer.add(*example)
            positions += len(examples)
            terminals += int(terminal)

        # Training phase
        total_ploss = total_vloss = 0.0
        steps_done  = 0
        for _ in range(TRAIN_STEPS):
            pl, vl = train_step(model, optimizer, buffer, device)
            if pl is None:
                break
            total_ploss += pl
            total_vloss += vl
            steps_done  += 1

        avg_pl = total_ploss / max(steps_done, 1)
        avg_vl = total_vloss / max(steps_done, 1)
        print(
            f"Gen {gen+1:4d} | "
            f"terminals={terminals}/{GAMES_PER_GEN} positions={positions} buf={len(buffer)} | "
            f"policy_loss={avg_pl:.4f} value_loss={avg_vl:.4f} heuristic_w={hw:.2f}"
        )

        path = save_checkpoint(model, buffer, gen + 1)
        print(f"  Saved {path}")

        log_file.write(f"{gen+1},{terminals},{positions},{len(buffer)},{avg_pl:.6f},{avg_vl:.6f},{hw:.4f}\n")
        log_file.flush()

    log_file.close()


if __name__ == "__main__":
    train()
