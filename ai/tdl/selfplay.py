"""
Gen 0 self-play using the C++ MCTS + hand-crafted heuristic.

Each game collects one training pair per state visited during each player's turn:
  input  : every s_i where current_player == P  (current player's fog)
  target : V_mcts(s_0')  where s_0' is the same player's next turn-start

All states in a single turn share one target (the V_mcts of the next turn start).

Usage:
  python selfplay.py                # 5 games, 200 sims
  python selfplay.py --games 20 --sims 400
"""

import sys, os, random, math, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/bindings'))
import polyshark

REPLAYS_DIR = os.path.join(os.path.dirname(__file__), '../../replays')

MAP_SIZE     = 11
C_UCT        = 1.5
BATCH_SIZE   = 32
VIRTUAL_LOSS = 1.0
TURN_LIMIT   = 30
C_HEURISTIC  = 15.0


# ---------------------------------------------------------------------------
# Gen 0 heuristic
# ---------------------------------------------------------------------------

def eval_fn(states):
    out = []
    for s in states:
        p    = s.current_player()
        diff = polyshark.heuristic_score(s, p) - polyshark.heuristic_score(s, 1 - p)
        out.append(math.tanh(diff / C_HEURISTIC))
    return out


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
        eval_fns = eval_fn
    if callable(eval_fns):
        eval_fns = [eval_fns, eval_fns]

    state = polyshark.make_random_game(seed=seed, sz=MAP_SIZE)
    sz    = state.map_size()

    # Per-player buffer of every state visited during the current turn.
    # All states in the buffer share the same target: V_mcts(s_0') of
    # that player's *next* turn start.
    turn_buf    = [[], []]
    pairs       = []   # (GameState, float)
    history     = []
    prev_player = -1

    while not state.is_terminal() and state.get_turn() < TURN_LIMIT:
        p              = state.current_player()
        action, v_mcts = engine.search(state, n_sims, eval_fns[p])

        if p != prev_player:
            # New turn for player p.
            # v_mcts(s_0) of this turn is the TARGET for every state
            # buffered from p's previous turn.
            for s in turn_buf[p]:
                pairs.append((s, v_mcts))
            turn_buf[p] = []
            prev_player = p

        # Collect current state before advancing.
        turn_buf[p].append(state)

        history.append(action)
        state = state.apply_action(action)

    # Final pairs: no next turn — use terminal outcome or heuristic.
    terminal = state.is_terminal()
    winner   = state.winner() if terminal else -1
    for p in range(2):
        if not turn_buf[p]:
            continue
        if terminal:
            v_final = 1.0 if winner == p else -1.0
        else:
            diff    = _score(state, p) - _score(state, 1 - p)
            v_final = math.tanh(diff / C_HEURISTIC)
        for s in turn_buf[p]:
            pairs.append((s, v_final))

    return pairs, history, sz, terminal, winner


def _save_replay(history, seed, sz, idx):
    os.makedirs(REPLAYS_DIR, exist_ok=True)
    path = os.path.join(REPLAYS_DIR, f'tdl_{idx:05d}.replay')
    with open(path, 'w') as f:
        f.write(f'seed {seed} {sz}\n')
        for a in history:
            f.write(f'{int(a.type)} {a.src} {a.dst} {a.param}\n')
    return path


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_selfplay(n_games=5, n_sims=200, eval_fns=None):
    """
    eval_fns : single callable or [eval_p0, eval_p1].
               Defaults to the Gen 0 heuristic for both players.
    """
    engine = polyshark.MCTSEngine(
        c_uct        = C_UCT,
        batch_size   = BATCH_SIZE,
        virtual_loss = VIRTUAL_LOSS,
    )

    all_pairs = []
    t0 = time.time()

    for i in range(n_games):
        seed = random.randint(1, 2**32 - 1)
        gt   = time.time()

        pairs, history, sz, terminal, winner = run_game(engine, seed, n_sims, eval_fns)
        all_pairs.extend(pairs)

        replay  = _save_replay(history, seed, sz, i)
        outcome = f'P{winner} wins' if terminal else 'turn limit'
        print(f'[{i+1}/{n_games}] seed={seed}  {len(history)} actions  '
              f'{len(pairs)} pairs  {outcome}  ({time.time()-gt:.1f}s)  → {os.path.basename(replay)}')

    elapsed = time.time() - t0
    print(f'\n{len(all_pairs)} total training pairs from {n_games} games in {elapsed:.1f}s')
    return all_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=5)
    parser.add_argument('--sims',  type=int, default=200)
    args = parser.parse_args()
    run_selfplay(n_games=args.games, n_sims=args.sims)
