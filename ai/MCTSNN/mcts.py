import math
import numpy as np
import torch

from encoder import encode
from model import ACTION_SIZE
from action_codec import legal_action_indices, index_to_action

END_TURN_ACTION = ACTION_SIZE - 1  # 7994

C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
VIRTUAL_LOSS = 1.0
BATCH_SIZE = 8  # leaves evaluated per wave


class Node:
    __slots__ = ('p', 'w', 'n', 'children', 'is_expanded')

    def __init__(self):
        self.p: dict[int, float] = {}
        self.w: dict[int, float] = {}
        self.n: dict[int, int] = {}
        self.children: dict[int, 'Node'] = {}
        self.is_expanded = False

    def init_priors(self, policy: np.ndarray, legal: list[int]):
        for a in legal:
            self.p[a] = float(policy[a])
            self.w[a] = 0.0
            self.n[a] = 0
            self.children[a] = Node()
        self.is_expanded = True

    def select_action(self) -> int:
        sqrt_N = math.sqrt(max(sum(self.n.values()), 1))
        best_score = -float('inf')
        best_action = -1
        for a in self.p:
            q = self.w[a] / self.n[a] if self.n[a] > 0 else 0.0
            u = C_PUCT * self.p[a] * sqrt_N / (1 + self.n[a])
            if q + u > best_score:
                best_score = q + u
                best_action = a
        return best_action


class MCTS:
    def __init__(self, model, device: str = 'cpu', heuristic_fn=None, heuristic_weight: float = 0.0):
        self.model          = model
        self.device         = device
        self.heuristic_fn   = heuristic_fn
        self.heuristic_weight = heuristic_weight
        self.model.eval()

    def search(self, root_state, n_simulations: int = 800, add_noise: bool = False) -> Node:
        """
        Run MCTS in waves of BATCH_SIZE simulations. Within each wave all leaves
        are evaluated in a single NN forward pass, then the tree is updated before
        the next wave begins — so the tree grows progressively as in sequential MCTS.
        """
        root = Node()
        self._expand_nodes([root_state], [root])

        if add_noise and root.p:
            self._add_dirichlet_noise(root)

        done = 0
        while done < n_simulations:
            wave = min(BATCH_SIZE, n_simulations - done)

            paths, leaf_nodes, leaf_states = [], [], []
            for _ in range(wave):
                path, node, state = self._select(root, root_state)
                paths.append(path)
                leaf_nodes.append(node)
                leaf_states.append(state)

            values = self._evaluate_leaves(leaf_states, leaf_nodes)

            for path, v in zip(paths, values):
                for parent, a in reversed(path):
                    if a == END_TURN_ACTION:
                        v = -v
                    parent.w[a] += VIRTUAL_LOSS + v
                    # n[a] already incremented during _select via virtual loss

            done += wave

        return root

    def get_policy(self, root: Node, temperature: float = 1.0) -> np.ndarray:
        visits = np.zeros(ACTION_SIZE, dtype=np.float32)
        for a, count in root.n.items():
            visits[a] = count

        if temperature == 0:
            probs = np.zeros(ACTION_SIZE, dtype=np.float32)
            probs[int(np.argmax(visits))] = 1.0
            return probs

        visits **= 1.0 / temperature
        total = visits.sum()
        return visits / total if total > 0 else visits

    # ------------------------------------------------------------------

    def _select(self, root: Node, root_state) -> tuple:
        """Walk tree to a leaf using PUCT, applying virtual loss on each edge."""
        node = root
        state = root_state
        path: list[tuple[Node, int]] = []

        while node.is_expanded:
            a = node.select_action()
            node.w[a] -= VIRTUAL_LOSS
            node.n[a] += 1
            path.append((node, a))
            node = node.children[a]
            state = state.apply_action(index_to_action(a, state))

        return path, node, state

    def _evaluate_leaves(self, states: list, nodes: list[Node]) -> list[float]:
        """
        Batch-evaluate a list of leaves. Handles four cases:
          - terminal state       → exact outcome
          - EndTurn cached       → reuse cached value
          - EndTurn uncached     → value-only NN, cache result
          - unexpanded normal    → full NN (policy + value), expand node
          - already expanded     → value-only NN (another sim in wave expanded it first)
        """
        values = [None] * len(states)
        expand_idx: list[int] = []
        value_only_idx: list[int] = []

        for i, (state, node) in enumerate(zip(states, nodes)):
            if state.is_terminal():
                winner = state.winner()
                values[i] = 1.0 if winner == state.current_player() else -1.0
            elif not node.is_expanded:
                expand_idx.append(i)
            else:
                value_only_idx.append(i)

        if expand_idx:
            vs = self._expand_nodes(
                [states[i] for i in expand_idx],
                [nodes[i]  for i in expand_idx],
            )
            for j, i in enumerate(expand_idx):
                values[i] = vs[j]

        if value_only_idx:
            vs = self._value_only([states[i] for i in value_only_idx])
            for j, i in enumerate(value_only_idx):
                values[i] = vs[j]

        return values

    def _blend(self, nn_val: float, state) -> float:
        if self.heuristic_fn is None or self.heuristic_weight <= 0.0:
            return nn_val
        h = self.heuristic_fn(state, state.current_player())
        return (1.0 - self.heuristic_weight) * nn_val + self.heuristic_weight * h

    def _expand_nodes(self, states: list, nodes: list[Node]) -> list[float]:
        """Batch NN call: populate priors, return value estimates."""
        spatial_t, global_t, legals, mask = self._encode_batch(states)
        with torch.no_grad():
            policies, vals = self.model(spatial_t, global_t, mask)
        policies_np = policies.cpu().numpy()
        vals_np = vals.squeeze(-1).cpu().numpy()
        for i, node in enumerate(nodes):
            if not node.is_expanded:
                node.init_priors(policies_np[i], legals[i])
        return [self._blend(float(vals_np[i]), states[i]) for i in range(len(states))]

    def _value_only(self, states: list) -> list[float]:
        """Batch NN call: value head only, no expansion."""
        spatial_t, global_t, _, _ = self._encode_batch(states)
        with torch.no_grad():
            _, vals = self.model(spatial_t, global_t)
        return [self._blend(float(v), s) for v, s in zip(vals.squeeze(-1).cpu().numpy(), states)]

    def _encode_batch(self, states: list):
        spatials, globals_, legals = [], [], []
        for state in states:
            spatial, global_vec = encode(state)
            spatials.append(spatial)
            globals_.append(global_vec)
            legals.append(legal_action_indices(state))

        spatial_t = torch.tensor(np.stack(spatials)).to(self.device)
        global_t  = torch.tensor(np.stack(globals_)).to(self.device)

        B = len(states)
        mask = torch.zeros(B, ACTION_SIZE, dtype=torch.bool, device=self.device)
        for i, legal in enumerate(legals):
            mask[i, legal] = True

        return spatial_t, global_t, legals, mask

    def _add_dirichlet_noise(self, root: Node):
        actions = list(root.p.keys())
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(actions))
        for a, eta in zip(actions, noise):
            root.p[a] = (1 - DIRICHLET_EPSILON) * root.p[a] + DIRICHLET_EPSILON * eta
