"""
Self-play data generation using C++ MCTS + heuristic/NN leaf evaluator.

Replay file format:
    seed <seed> <sz>
    outcome <v_p0> <v_p1>
    <type> <src> <dst> <param> <path_bits> <path_steps>
    ...

Training targets are the final game outcome (±1 terminal, heuristic tanh at turn
limit) — not MCTS leaf values. MCTS uses eval_fn for action selection only.

Usage:
  python selfplay.py                # 5 games, 200 sims
  python selfplay.py --games 40 --sims 100 --workers 8
"""

import sys, os, random, math, argparse, time
import multiprocessing as mp

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, '../../build/bindings'))
import polyshark

REPLAYS_DIR  = os.path.join(_HERE, 'replays')

MAP_SIZE     = 11
C_UCT        = 1.5
BATCH_SIZE   = 32
VIRTUAL_LOSS = 1.0
TURN_LIMIT   = 40
C_HEURISTIC  = 15.0
TEMPERATURE  = 0.0


# ---------------------------------------------------------------------------
# Eval functions
# ---------------------------------------------------------------------------

def heuristic_eval_fn(states):
    out = []
    for s in states:
        p    = s.current_player()
        diff = polyshark.heuristic_score(s, p) - polyshark.heuristic_score(s, 1 - p)
        out.append(math.tanh(diff / C_HEURISTIC))
    return out

eval_fn = heuristic_eval_fn


def make_nn_eval_fn(ckpt_path):
    import torch
    from model   import ValueNet
    from encoder import encode_batch
    device = torch.device('cpu')
    model  = ValueNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True)['model'])
    model.eval()
    def _fn(states):
        sp, gv = encode_batch(states)
        with torch.no_grad():
            pred = model(torch.from_numpy(sp), torch.from_numpy(gv))
        return pred.squeeze(-1).tolist()
    return _fn


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def run_game(engine, seed, n_sims, eval_fns=None):
    """
    Play one game. Returns (pairs, history, sz, terminal, winner, v_finals).
    Training targets are the final outcome, not MCTS values.
    eval_fns: callable or [p0_fn, p1_fn].
    """
    if eval_fns is None:
        eval_fns = heuristic_eval_fn
    if callable(eval_fns):
        eval_fns = [eval_fns, eval_fns]

    state      = polyshark.make_random_game(seed=seed, sz=MAP_SIZE)
    sz         = state.map_size()
    all_states = [[], []]
    history    = []

    while not state.is_terminal() and state.get_turn() < TURN_LIMIT:
        p         = state.current_player()
        action, _ = engine.search(state, n_sims, eval_fns[p], TEMPERATURE)
        all_states[p].append(state)
        history.append(action)
        state = state.apply_action(action)

    terminal = state.is_terminal()
    winner   = state.winner() if terminal else -1

    pairs    = []
    v_finals = []
    for p in range(2):
        if terminal:
            v_final = 1.0 if winner == p else -1.0
        else:
            diff    = polyshark.heuristic_score(state, p) - polyshark.heuristic_score(state, 1 - p)
            v_final = math.tanh(diff / C_HEURISTIC)
        v_finals.append(v_final)
        for s in all_states[p]:
            pairs.append((s, v_final))

    return pairs, history, sz, terminal, winner, v_finals


def _save_replay(history, v_finals, seed, sz, gen, game_idx):
    gen_dir = os.path.join(REPLAYS_DIR, f'gen_{gen:03d}')
    os.makedirs(gen_dir, exist_ok=True)
    path = os.path.join(gen_dir, f'game_{game_idx:02d}_seed_{seed}.replay')
    with open(path, 'w') as f:
        f.write(f'seed {seed} {sz}\n')
        f.write(f'outcome {v_finals[0]:.6f} {v_finals[1]:.6f}\n')
        for a in history:
            f.write(f'{int(a.type)} {a.src} {a.dst} {a.param} {a.path_bits} {a.path_steps}\n')
    return path


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

_worker_engine = None

def _worker_init():
    global _worker_engine
    _worker_engine = polyshark.MCTSEngine(
        c_uct=C_UCT, batch_size=BATCH_SIZE, virtual_loss=VIRTUAL_LOSS,
    )

def _worker_task(args):
    seed, n_sims, ckpt_path, gen, game_idx = args
    fns = make_nn_eval_fn(ckpt_path) if ckpt_path else heuristic_eval_fn
    pairs, history, sz, terminal, winner, v_finals = run_game(_worker_engine, seed, n_sims, fns)
    replay = _save_replay(history, v_finals, seed, sz, gen, game_idx)
    outcome = f'P{winner} wins' if terminal else 'turn limit'
    return replay, terminal, winner, v_finals, seed, len(history), outcome


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_selfplay(n_games=5, n_sims=200, eval_fns=None, gen=0,
                 n_workers=1, ckpt_path=None):
    """
    Run n_games games and save replay files.
    Returns (train_paths, val_paths) — 10% of games held out for validation.
    """
    seeds = [random.randint(1, 2**32 - 1) for _ in range(n_games)]
    t0    = time.time()
    paths = []   # all replay paths in order

    if n_workers > 1:
        task_args = [(seeds[i], n_sims, ckpt_path, gen, i) for i in range(n_games)]
        with mp.Pool(n_workers, initializer=_worker_init) as pool:
            for replay, terminal, winner, v_finals, seed, n_actions, outcome in \
                    pool.imap_unordered(_worker_task, task_args):
                paths.append(replay)
                print(f'  game seed={seed}  {n_actions} actions  '
                      f'{outcome}  P0={v_finals[0]:+.3f} P1={v_finals[1]:+.3f}  → {replay}')
    else:
        engine = polyshark.MCTSEngine(
            c_uct=C_UCT, batch_size=BATCH_SIZE, virtual_loss=VIRTUAL_LOSS,
        )
        fns = make_nn_eval_fn(ckpt_path) if ckpt_path else (eval_fns or heuristic_eval_fn)
        for i in range(n_games):
            seed = seeds[i]
            gt   = time.time()
            _, history, sz, terminal, winner, v_finals = run_game(engine, seed, n_sims, fns)
            replay  = _save_replay(history, v_finals, seed, sz, gen, i)
            paths.append(replay)
            outcome = f'P{winner} wins' if terminal else 'turn limit'
            print(f'[{i+1}/{n_games}] seed={seed}  {len(history)} actions  '
                  f'{outcome}  P0={v_finals[0]:+.3f} P1={v_finals[1]:+.3f}  ({time.time()-gt:.1f}s)  → {replay}')

    # Split by game: 10% → val, rest → train.
    n_val   = max(1, n_games // 10)
    order   = list(range(len(paths)))
    random.shuffle(order)
    val_paths   = [paths[i] for i in order[:n_val]]
    train_paths = [paths[i] for i in order[n_val:]]

    elapsed = time.time() - t0
    print(f'\n{len(train_paths)} train  {len(val_paths)} val  games in {elapsed:.1f}s')
    return train_paths, val_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games',   type=int, default=5)
    parser.add_argument('--sims',    type=int, default=200)
    parser.add_argument('--gen',     type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    run_selfplay(n_games=args.games, n_sims=args.sims, gen=args.gen, n_workers=args.workers)
