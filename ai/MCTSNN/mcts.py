import math
import numpy as np
import torch

from .encoder import encode
from .model import ACTION_SIZE
from .action_codec import legal_action_indices, index_to_action

END_TURN_ACTION = ACTION_SIZE - 1  # 7994

C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25


class Node:
    __slots__ = ('p', 'w', 'n', 'children', 'is_expanded', 'is_terminal', 'cached_value')

    def __init__(self, terminal: bool = False):
        self.p: dict[int, float] = {}         # prior probability per action
        self.w: dict[int, float] = {}         # total value per action
        self.n: dict[int, int] = {}           # visit count per action
        self.children: dict[int, 'Node'] = {}
        self.is_expanded = False
        self.is_terminal = terminal
        self.cached_value: float | None = None  # cached for EndTurn leaves

    def init_priors(self, policy: np.ndarray, legal: list[int]):
        for a in legal:
            self.p[a] = float(policy[a])
            self.w[a] = 0.0
            self.n[a] = 0
            self.children[a] = Node(terminal=(a == END_TURN_ACTION))
        self.is_expanded = True

    def select_action(self) -> int:
        """PUCT selection over legal actions."""
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
    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()

    def search(self, root_state, n_simulations: int = 800, add_noise: bool = False) -> Node:
        """
        Run MCTS from root_state and return the root Node.

        Requires root_state to expose:
          .legal_actions()   — list[Action]
          .apply_action(a)   — returns a new GameState (does not mutate)
          .is_terminal()     — bool
          .terminal_value()  — float in {-1, 0, 1} from current player's view
        """
        root = Node()
        self._expand_node(root_state, root)

        if add_noise and root.p:
            self._add_dirichlet_noise(root)

        for _ in range(n_simulations):
            node = root
            state = root_state   # apply_action returns new states; root_state is never mutated
            path: list[tuple[Node, int]] = []

            # Selection — walk down using PUCT
            while node.is_expanded and not node.is_terminal:
                a = node.select_action()
                path.append((node, a))
                node = node.children[a]
                state = state.apply_action(index_to_action(a, state))

            # Expansion — new node (or terminal evaluation)
            if state.is_terminal():
                v = float(state.terminal_value())
            elif node.is_terminal:
                # EndTurn leaf: state is after applying EndTurn (opponent's view)
                if node.cached_value is None:
                    node.cached_value = self._evaluate(state)
                v = node.cached_value
            else:
                v = self._expand_node(state, node)

            # Backprop — only negate at the turn boundary (EndTurn action)
            for parent, a in reversed(path):
                if a == END_TURN_ACTION:
                    v = -v
                parent.w[a] += v
                parent.n[a] += 1

        return root

    def get_policy(self, root: Node, temperature: float = 1.0) -> np.ndarray:
        """Visit-count distribution over all actions."""
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

    def _expand_node(self, state, node: Node) -> float:
        """Call NN, populate node with priors, return value estimate."""
        spatial, global_vec = encode(state)
        spatial_t = torch.tensor(spatial).unsqueeze(0).to(self.device)
        global_t  = torch.tensor(global_vec).unsqueeze(0).to(self.device)

        legal = legal_action_indices(state)
        mask = torch.zeros(1, ACTION_SIZE, dtype=torch.bool, device=self.device)
        mask[0, legal] = True

        with torch.no_grad():
            policy, value = self.model(spatial_t, global_t, mask)

        node.init_priors(policy.squeeze(0).cpu().numpy(), legal)
        return float(value.item())

    def _evaluate(self, state) -> float:
        """Value-only NN call — no expansion."""
        spatial, global_vec = encode(state)
        spatial_t = torch.tensor(spatial).unsqueeze(0).to(self.device)
        global_t  = torch.tensor(global_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.model(spatial_t, global_t)
        return float(value.item())

    def _add_dirichlet_noise(self, root: Node):
        actions = list(root.p.keys())
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(actions))
        for a, eta in zip(actions, noise):
            root.p[a] = (1 - DIRICHLET_EPSILON) * root.p[a] + DIRICHLET_EPSILON * eta
