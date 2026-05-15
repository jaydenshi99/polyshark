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

# --- Hyperparameters ---
TURN_LIMIT         = 10
GAMES_PER_GEN      = 20
TRAIN_STEPS        = 500
BUFFER_SIZE        = 50_000
BATCH_SIZE         = 256
LR                 = 1e-3
N_SIMULATIONS      = 50
TEMPERATURE_CUTOFF = 6    # use temp=1.0 for first 6 turns, then greedy

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
    my_stars = state.get_stars(player)
    op_stars = state.get_stars(opp)

    score = (
        2.0 * (my_income - op_income) +
        1.5 * (my_cities - op_cities) +
        1.0 * (my_levels - op_levels) +
        0.8 * (my_techs  - op_techs)  +
        0.3 * (my_stars  - op_stars)  +
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

def save_replay(history, gen, game_idx):
    os.makedirs(REPLAYS_DIR, exist_ok=True)
    path = os.path.join(REPLAYS_DIR, f"gen{gen:04d}_game{game_idx:03d}.replay")
    with open(path, "w") as f:
        for a in history:
            f.write(f"{int(a.type)} {a.src} {a.dst} {a.param}\n")


# --- Self-play ---

def run_game(mcts, gen=0, game_idx=0) -> tuple[list, bool]:
    """
    Play one game. Returns (examples, terminal) where:
      examples  — list of (spatial, global_vec, policy, outcome)
      terminal  — True if game ended by capture, False if turn limit hit
    """
    state      = polyshark.make_game()
    trajectory = []  # (spatial, global_vec, policy, current_player)
    history    = []  # raw actions for replay file

    while not state.is_terminal():
        if state.get_turn() >= TURN_LIMIT:
            break

        player = state.current_player()
        turn   = state.get_turn()

        root   = mcts.search(state, n_simulations=N_SIMULATIONS, add_noise=True)
        temp   = 1.0 if turn < TEMPERATURE_CUTOFF else 0.0
        policy = mcts.get_policy(root, temperature=temp)

        spatial, global_vec = encode(state)
        trajectory.append((spatial, global_vec, policy, player))

        action_idx = int(np.random.choice(ACTION_SIZE, p=policy))
        action     = index_to_action(action_idx, state)
        history.append(action)
        state      = state.apply_action(action)

    terminal = state.is_terminal()

    if terminal:
        winner   = state.winner()
        outcomes = [1.0 if p == winner else -1.0 for _, _, _, p in trajectory]
    else:
        outcomes = [heuristic_value(state, p) for _, _, _, p in trajectory]

    examples = [(s, g, pol, out) for (s, g, pol, _), out in zip(trajectory, outcomes)]
    save_replay(history, gen, game_idx)
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
    optimizer.step()
    model.eval()

    return policy_loss.item(), value_loss.item()


# --- Evaluation ---

def evaluate(new_model, old_model, device, n_games=20):
    """Pit new_model against old_model. Returns win rate of new_model."""
    new_mcts = MCTS(new_model, device=device)
    old_mcts = MCTS(old_model, device=device)

    new_wins = 0
    for game_idx in range(n_games):
        # Alternate which side new_model plays to avoid first-mover bias
        new_is_p0 = (game_idx % 2 == 0)
        state = polyshark.make_game()

        while not state.is_terminal():
            if state.get_turn() >= TURN_LIMIT:
                break
            player  = state.current_player()
            is_new  = (player == 0 and new_is_p0) or (player == 1 and not new_is_p0)
            mcts    = new_mcts if is_new else old_mcts
            root    = mcts.search(state, n_simulations=N_SIMULATIONS, add_noise=False)
            policy  = mcts.get_policy(root, temperature=0)
            action_idx = int(np.argmax(policy))
            action  = index_to_action(action_idx, state)
            state   = state.apply_action(action)

        if state.is_terminal():
            winner        = state.winner()
            new_player_id = 0 if new_is_p0 else 1
            if winner == new_player_id:
                new_wins += 1
        else:
            # No terminal — use heuristic to decide
            new_player_id = 0 if new_is_p0 else 1
            h = heuristic_value(state, new_player_id)
            if h > 0:
                new_wins += 1

    return new_wins / n_games


# --- Checkpointing ---

def save_checkpoint(model, gen):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"model_{gen:04d}.pt")
    torch.save({'gen': gen, 'model': model.state_dict()}, path)
    return path


def load_latest_checkpoint(model):
    if not os.path.exists(CHECKPOINT_DIR):
        return 0
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
    if not files:
        return 0
    latest = max(files, key=lambda f: int(f.split('_')[1].split('.')[0]))
    ckpt   = torch.load(os.path.join(CHECKPOINT_DIR, latest), weights_only=True)
    model.load_state_dict(ckpt['model'])
    print(f"Resumed from generation {ckpt['gen']}")
    return ckpt['gen']


# --- Main loop ---

def train(n_generations=1000):
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model     = PolysharkNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    buffer    = ReplayBuffer()
    mcts      = MCTS(model, device=device)

    start_gen = load_latest_checkpoint(model)

    for gen in range(start_gen, start_gen + n_generations):
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
            f"policy_loss={avg_pl:.4f} value_loss={avg_vl:.4f}"
        )

        path = save_checkpoint(model, gen + 1)
        print(f"  Saved {path}")


if __name__ == "__main__":
    train()
