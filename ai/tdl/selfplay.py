"""
Gen 0 self-play using the C++ MCTS + hand-crafted heuristic.

Each game collects one training pair per state visited during each player's turn:
  input  : every s_i where current_player == P  (current player's fog)
  target : V_mcts(s_0')  where s_0' is the same player's next turn-start

All states in a single turn share one target (the V_mcts of the next turn start).

Usage:
  python selfplay.py                # 5 games, 200 sims
  python selfplay.py --games 20 --sims 400 --workers 8
"""

import sys, os, random, math, argparse, time
import multiprocessing as mp

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, '../../build/bindings'))
import polyshark

REPLAYS_DIR = os.path.join(_HERE, 'replays')

MAP_SIZE     = 11
C_UCT        = 1.5
BATCH_SIZE   = 32
VIRTUAL_LOSS = 1.0
TURN_LIMIT   = 15
C_HEURISTIC  = 15.0
TEMPERATURE  = 0.0   # self-play exploration; 0 = deterministic (use for eval)


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

# Keep eval_fn as an alias so train.py imports still work.
eval_fn = heuristic_eval_fn


def make_nn_eval_fn(ckpt_path):
    """Load a ValueNet from ckpt_path and return an eval_fn callable."""
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


def make_blended_eval_fn(ckpt_path, heuristic_weight):
    """Blend NN and heuristic: (1-w)*NN + w*heuristic."""
    nn_fn  = make_nn_eval_fn(ckpt_path)
    nn_w   = 1.0 - heuristic_weight
    def _fn(states):
        nn_vals = nn_fn(states)
        h_vals  = heuristic_eval_fn(states)
        return [nn_w * n + heuristic_weight * h for n, h in zip(nn_vals, h_vals)]
    return _fn


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def run_game(engine, seed, n_sims, eval_fns=None):
    """
    Play one game.

    eval_fns : callable or list of two callables.
               Pass a single callable to use the same evaluator for both players.
               Pass [eval_p0, eval_p1] to pit two different evaluators against
               each other (used for convergence evaluation).
    """
    if eval_fns is None:
        eval_fns = heuristic_eval_fn
    if callable(eval_fns):
        eval_fns = [eval_fns, eval_fns]

    state = polyshark.make_random_game(seed=seed, sz=MAP_SIZE)
    sz    = state.map_size()

    turn_buf    = [[], []]
    pairs       = []
    history     = []
    prev_player = -1

    while not state.is_terminal() and state.get_turn() < TURN_LIMIT:
        p              = state.current_player()
        action, v_mcts = engine.search(state, n_sims, eval_fns[p], TEMPERATURE)

        if p != prev_player:
            for s in turn_buf[p]:
                pairs.append((s, v_mcts))
            turn_buf[p] = []
            prev_player = p

        turn_buf[p].append(state)
        history.append(action)
        state = state.apply_action(action)

    terminal = state.is_terminal()
    winner   = state.winner() if terminal else -1
    for p in range(2):
        if not turn_buf[p]:
            continue
        if terminal:
            v_final = 1.0 if winner == p else -1.0
        else:
            diff    = polyshark.heuristic_score(state, p) - polyshark.heuristic_score(state, 1 - p)
            v_final = math.tanh(diff / C_HEURISTIC)
        for s in turn_buf[p]:
            pairs.append((s, v_final))

    return pairs, history, sz, terminal, winner


def _save_replay(history, seed, sz, gen, game_idx):
    gen_dir = os.path.join(REPLAYS_DIR, f'gen_{gen:03d}')
    os.makedirs(gen_dir, exist_ok=True)
    path = os.path.join(gen_dir, f'game_{game_idx:02d}_seed_{seed}.replay')
    with open(path, 'w') as f:
        f.write(f'seed {seed} {sz}\n')
        for a in history:
            f.write(f'{int(a.type)} {a.src} {a.dst} {a.param}\n')
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
    seed, n_sims, ckpt_path, heuristic_weight, gen, game_idx = args
    if ckpt_path:
        fns = make_blended_eval_fn(ckpt_path, heuristic_weight) if heuristic_weight > 0 else make_nn_eval_fn(ckpt_path)
    else:
        fns = heuristic_eval_fn
    pairs, history, sz, terminal, winner = run_game(_worker_engine, seed, n_sims, fns)
    replay = _save_replay(history, seed, sz, gen, game_idx)
    # Encode here — GameState objects can't be pickled across processes.
    from encoder import encode_batch
    import numpy as np
    if pairs:
        states, targets = zip(*pairs)
        sp, gv = encode_batch(list(states))
        tgt    = np.array(targets, dtype=np.float32)
    else:
        from encoder import C_IN, G
        sp  = np.empty((0, C_IN, 11, 11), dtype=np.float32)
        gv  = np.empty((0, G),            dtype=np.float32)
        tgt = np.empty((0,),              dtype=np.float32)
    return sp, gv, tgt, terminal, winner, seed, len(history), replay


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_selfplay(n_games=5, n_sims=200, eval_fns=None, gen=0,
                 n_workers=1, ckpt_path=None, heuristic_weight=0.0):
    """
    n_workers  : number of parallel worker processes (1 = sequential).
    ckpt_path  : path to a ValueNet checkpoint for the leaf evaluator.
                 None = use the Gen 0 heuristic.
                 In parallel mode this is always used; eval_fns is ignored.
    eval_fns   : callable or [p0_fn, p1_fn] — used in sequential mode only.
    """
    seeds = [random.randint(1, 2**32 - 1) for _ in range(n_games)]
    t0    = time.time()

    # Encoded buffer — arrays are accumulated regardless of sequential/parallel path.
    from encoder import encode_batch, C_IN, G
    import numpy as np
    sp_all  = np.empty((0, C_IN, 11, 11), dtype=np.float32)
    gv_all  = np.empty((0, G),            dtype=np.float32)
    tgt_all = np.empty((0,),              dtype=np.float32)

    if n_workers > 1:
        task_args = [(seeds[i], n_sims, ckpt_path, heuristic_weight, gen, i) for i in range(n_games)]
        with mp.Pool(n_workers, initializer=_worker_init) as pool:
            for sp, gv, tgt, terminal, winner, seed, n_actions, replay in \
                    pool.imap_unordered(_worker_task, task_args):
                sp_all  = np.concatenate([sp_all,  sp],  axis=0)
                gv_all  = np.concatenate([gv_all,  gv],  axis=0)
                tgt_all = np.concatenate([tgt_all, tgt], axis=0)
                outcome = f'P{winner} wins' if terminal else 'turn limit'
                print(f'  game seed={seed}  {n_actions} actions  '
                      f'{len(tgt)} pairs  {outcome}  → {replay}')
    else:
        engine = polyshark.MCTSEngine(
            c_uct=C_UCT, batch_size=BATCH_SIZE, virtual_loss=VIRTUAL_LOSS,
        )
        if ckpt_path:
            fns = make_blended_eval_fn(ckpt_path, heuristic_weight) if heuristic_weight > 0 else make_nn_eval_fn(ckpt_path)
        else:
            fns = eval_fns or heuristic_eval_fn
        for i in range(n_games):
            seed = seeds[i]
            gt   = time.time()
            pairs, history, sz, terminal, winner = run_game(engine, seed, n_sims, fns)
            if pairs:
                states, targets = zip(*pairs)
                sp, gv = encode_batch(list(states))
                tgt    = np.array(targets, dtype=np.float32)
                sp_all  = np.concatenate([sp_all,  sp],  axis=0)
                gv_all  = np.concatenate([gv_all,  gv],  axis=0)
                tgt_all = np.concatenate([tgt_all, tgt], axis=0)
            replay  = _save_replay(history, seed, sz, gen, i)
            outcome = f'P{winner} wins' if terminal else 'turn limit'
            print(f'[{i+1}/{n_games}] seed={seed}  {len(history)} actions  '
                  f'{len(pairs)} pairs  {outcome}  ({time.time()-gt:.1f}s)  → {replay}')

    elapsed = time.time() - t0
    print(f'\n{len(tgt_all)} total training pairs from {n_games} games in {elapsed:.1f}s')
    return sp_all, gv_all, tgt_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games',   type=int, default=5)
    parser.add_argument('--sims',    type=int, default=200)
    parser.add_argument('--gen',     type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    run_selfplay(n_games=args.games, n_sims=args.sims, gen=args.gen, n_workers=args.workers)
