"""
Head-to-head comparison between two bots.

Each bot is "heuristic" or a path to a .pt checkpoint.
Games are split evenly: half with bot0 as P0, half with bot0 as P1
to control for first-mover advantage.

Usage:
  python eval.py --bot0 heuristic --bot1 checkpoints/gen_008.pt
  python eval.py --bot0 checkpoints/gen_003.pt --bot1 checkpoints/gen_008.pt
  python eval.py --bot0 heuristic --bot1 checkpoints/gen_008.pt --games 40 --sims 200 --turn_limit 40
"""

import sys, os, argparse, math, random, time

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, '../../build/bindings'))
import polyshark

from selfplay import (heuristic_eval_fn, make_nn_eval_fn,
                      MAP_SIZE, C_UCT, BATCH_SIZE, VIRTUAL_LOSS, C_HEURISTIC)


def make_eval_fn(spec):
    if spec == 'heuristic':
        return heuristic_eval_fn
    return make_nn_eval_fn(spec)


def play_game(engine, seed, n_sims, fn0, fn1, turn_limit):
    """Play one game with fn0=P0, fn1=P1. Returns (winner or -1, terminal, v_finals)."""
    state = polyshark.make_random_game(seed=seed, sz=MAP_SIZE)
    fns   = [fn0, fn1]

    while not state.is_terminal() and state.get_turn() < turn_limit:
        p         = state.current_player()
        action, _ = engine.search(state, n_sims, fns[p], 0.0)
        state     = state.apply_action(action)

    terminal = state.is_terminal()
    winner   = state.winner() if terminal else -1

    v_finals = []
    for p in range(2):
        if terminal:
            v_finals.append(1.0 if winner == p else -1.0)
        else:
            diff = polyshark.heuristic_score(state, p) - polyshark.heuristic_score(state, 1 - p)
            v_finals.append(math.tanh(diff / C_HEURISTIC))

    return winner, terminal, v_finals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bot0',       type=str, required=True,  help='"heuristic" or path to .pt')
    parser.add_argument('--bot1',       type=str, required=True,  help='"heuristic" or path to .pt')
    parser.add_argument('--games',      type=int, default=40,     help='total games (split evenly across sides)')
    parser.add_argument('--sims',       type=int, default=100)
    parser.add_argument('--turn_limit', type=int, default=40)
    args = parser.parse_args()

    print(f'Bot0: {args.bot0}')
    print(f'Bot1: {args.bot1}')
    print(f'Games: {args.games}  Sims: {args.sims}  Turn limit: {args.turn_limit}')
    print()

    fn0 = make_eval_fn(args.bot0)
    fn1 = make_eval_fn(args.bot1)
    engine = polyshark.MCTSEngine(c_uct=C_UCT, batch_size=BATCH_SIZE, virtual_loss=VIRTUAL_LOSS)

    n_games = args.games
    n_half  = n_games // 2
    seeds   = [random.randint(1, 2**32 - 1) for _ in range(n_games)]

    bot0_wins    = 0
    bot1_wins    = 0
    decisive     = 0  # terminal games
    bot0_v       = []  # outcome from bot0's perspective each game

    t0 = time.time()
    for i in range(n_games):
        seed = seeds[i]
        swap = i >= n_half  # second half: swap sides

        if not swap:
            winner, terminal, v_finals = play_game(engine, seed, args.sims, fn0, fn1, args.turn_limit)
            bot0_p = 0
        else:
            winner, terminal, v_finals = play_game(engine, seed, args.sims, fn1, fn0, args.turn_limit)
            bot0_p = 1

        v = v_finals[bot0_p]
        bot0_v.append(v)

        # Decide winner by heuristic outcome when game is not conclusive
        if terminal:
            decisive += 1
            bot0_won = (winner == bot0_p)
        else:
            bot0_won = v > 0  # bot0 has higher heuristic score

        if bot0_won:
            bot0_wins += 1
        else:
            bot1_wins += 1

        flag   = ' (terminal)' if terminal else ' (heuristic)'
        result = f'bot0 wins{flag}' if bot0_won else f'bot1 wins{flag}'
        side   = 'bot0=P0' if not swap else 'bot0=P1'
        print(f'[{i+1:2d}/{n_games}] seed={seed}  {side}  bot0={v:+.3f}  → {result}')

    elapsed = time.time() - t0
    avg_v   = sum(bot0_v) / len(bot0_v)

    print(f'\n{"─"*52}')
    print(f'  Bot0 ({args.bot0}):')
    print(f'    Wins:   {bot0_wins}/{n_games}  ({100*bot0_wins/n_games:.1f}%)')
    print(f'  Bot1 ({args.bot1}):')
    print(f'    Wins:   {bot1_wins}/{n_games}  ({100*bot1_wins/n_games:.1f}%)')
    print(f'  Terminal games: {decisive}/{n_games}')
    print(f'  Avg outcome (bot0): {avg_v:+.3f}')
    print(f'  Elapsed: {elapsed:.1f}s')
    print(f'{"─"*52}')


if __name__ == '__main__':
    main()
