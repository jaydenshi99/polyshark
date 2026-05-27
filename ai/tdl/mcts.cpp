#include "mcts.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

MCTSEngine::MCTSEngine(float c_uct, int batch_size, float virtual_loss)
    : _c_uct(c_uct), _batch_size(batch_size), _virtual_loss(virtual_loss) {}

MCTSNode* MCTSEngine::_alloc() {
    _pool.push_back(std::make_unique<MCTSNode>());
    return _pool.back().get();
}

// Populate node edges with all affordable legal actions.
// END_TURN is always affordable and always included.
void MCTSEngine::_expand(MCTSNode* node, const GameState& state) {
    Action buf[256];
    int count = 0;
    state.legal_actions(buf, count);
    for (int i = 0; i < count; i++) {
        if (!buf[i].affordable) continue;
        if (buf[i].type == ActionType::DebugAddPop) continue;
        node->edges.push_back({buf[i], 0.0f, 0, nullptr});
    }
    node->expanded = true;
}

// UCT without prior (uniform): Q(a) + C * sqrt(N_parent) / (1 + N(a)).
// Returns edge index of the selected action.
int MCTSEngine::_uct_pick(const MCTSNode* node) const {
    int N = 0;
    for (const auto& e : node->edges) N += e.n;
    float sqrt_N = std::sqrt((float)std::max(N, 1));

    float best  = std::numeric_limits<float>::lowest();
    int   best_i = 0;
    for (int i = 0; i < (int)node->edges.size(); i++) {
        const auto& e = node->edges[i];
        float q     = (e.n > 0) ? (e.w / (float)e.n) : 0.0f;
        float score = q + _c_uct * sqrt_N / (1.0f + (float)e.n);
        if (score > best) {
            best   = score;
            best_i = i;
        }
    }
    return best_i;
}

// Walk the tree from root to a leaf applying virtual loss along the path.
// Stops without traversing END_TURN — the pre-END_TURN state is the leaf.
MCTSEngine::SelResult MCTSEngine::_select(MCTSNode* root, const GameState& root_state) {
    SelResult r;
    MCTSNode* node  = root;
    GameState state = root_state;

    while (true) {
        if (state.is_terminal()) {
            r.is_terminal  = true;
            int w          = state.winner();
            r.terminal_val = (w == state.current_player()) ? 1.0f : -1.0f;
            r.leaf_state   = state;
            return r;
        }

        if (!node->expanded) {
            _expand(node, state);
            r.leaf_state = state;
            return r;
        }

        int   ei   = _uct_pick(node);
        auto& edge = node->edges[ei];

        // Apply virtual loss before any branch decision.
        edge.w -= _virtual_loss;
        edge.n += 1;
        r.path.push_back({node, ei});

        // END_TURN stop rule: evaluate the pre-END_TURN state.
        if (edge.action.type == ActionType::EndTurn) {
            r.leaf_state = state;
            return r;
        }

        // Traverse into the child.
        if (!edge.child) edge.child = _alloc();
        state = state.apply_action(edge.action);
        node  = edge.child;
    }
}

// Restore virtual loss and credit the actual evaluated value.
// No negation — the MCTS stays within a single player's turn.
void MCTSEngine::_backprop(SelResult& r, float value) {
    for (auto& [node, ei] : r.path) {
        auto& edge = node->edges[ei];
        edge.w += _virtual_loss + value;
        // edge.n was already incremented during _select
    }
}

std::pair<Action, float> MCTSEngine::search(const GameState& root_state,
                                             int              n_sims,
                                             EvalFn           eval_fn,
                                             float            temperature) {
    _pool.clear();
    MCTSNode* root = _alloc();
    _expand(root, root_state);

    int done = 0;
    while (done < n_sims) {
        int wave = std::min(_batch_size, n_sims - done);

        std::vector<SelResult>  pending;
        std::vector<GameState>  leaf_states;
        pending.reserve(wave);
        leaf_states.reserve(wave);

        for (int i = 0; i < wave; i++) {
            SelResult r = _select(root, root_state);
            if (r.is_terminal) {
                _backprop(r, r.terminal_val);
            } else {
                leaf_states.push_back(r.leaf_state);
                pending.push_back(std::move(r));
            }
        }

        if (!pending.empty()) {
            std::vector<float> values = eval_fn(leaf_states);
            for (int i = 0; i < (int)pending.size(); i++) {
                _backprop(pending[i], values[i]);
            }
        }

        done += wave;
    }

    // Compute root value and select action.
    float total_w = 0.0f, total_n = 0.0f;
    for (const auto& e : root->edges) { total_w += e.w; total_n += (float)e.n; }
    float root_value = (total_n > 0.0f) ? (total_w / total_n) : 0.0f;

    int best_i = 0;
    if (temperature <= 0.0f) {
        // Argmax by visit count.
        int best_n = -1;
        for (int i = 0; i < (int)root->edges.size(); i++) {
            if (root->edges[i].n > best_n) { best_n = root->edges[i].n; best_i = i; }
        }
    } else {
        // Sample proportional to N^(1/T).
        float inv_t = 1.0f / temperature;
        std::vector<float> weights(root->edges.size());
        float sum = 0.0f;
        for (int i = 0; i < (int)root->edges.size(); i++) {
            weights[i] = std::pow((float)root->edges[i].n, inv_t);
            sum += weights[i];
        }
        float r = ((float)std::rand() / (float)RAND_MAX) * sum;
        float cum = 0.0f;
        for (int i = 0; i < (int)weights.size(); i++) {
            cum += weights[i];
            if (r <= cum) { best_i = i; break; }
        }
    }

    return {root->edges[best_i].action, root_value};
}
