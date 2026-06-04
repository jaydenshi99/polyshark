#pragma once

#include "game_state.h"
#include <functional>
#include <memory>
#include <utility>
#include <vector>

// Batch leaf evaluator: receives a batch of pre-END_TURN GameStates from the
// current player's perspective, returns one value in [-1, 1] per state.
using EvalFn = std::function<std::vector<float>(const std::vector<GameState>&)>;

struct MCTSNode {
    struct Edge {
        Action    action;
        float     w     = 0.0f;  // sum of backpropped values (virtual loss already restored)
        int       n     = 0;     // visit count (includes virtual loss increments)
        MCTSNode* child = nullptr;
    };

    std::vector<Edge> edges;
    bool expanded = false;
};

// Per-turn MCTS: searches entirely within the current player's turn.
// Stops at END_TURN edges and evaluates the pre-END_TURN state — matching
// the training signal, which is always encoded from the current player's fog.
//
// Backpropagation never negates values (no player switch within a turn).
// The returned root value is the visit-weighted mean Q across all root edges.
class MCTSEngine {
public:
    explicit MCTSEngine(float c_uct = 1.5f, int batch_size = 8, float virtual_loss = 1.0f);

    // Run n_sims simulations from root_state using eval_fn at leaves.
    // temperature=0 → argmax (deterministic); temperature>0 → sample proportional to N^(1/T).
    // Returns: {selected_action, root_value (mean W/N)}.
    std::pair<Action, float> search(const GameState& root_state,
                                    int              n_sims,
                                    EvalFn           eval_fn,
                                    float            temperature = 0.0f);

private:
    float _c_uct;
    int   _batch_size;
    float _virtual_loss;

    std::vector<std::unique_ptr<MCTSNode>> _pool;

    MCTSNode* _alloc();

    struct SelResult {
        std::vector<std::pair<MCTSNode*, int>> path;  // (node, edge_idx); VL applied
        GameState leaf_state;
        bool      is_terminal  = false;
        float     terminal_val = 0.0f;
    };

    void      _expand   (MCTSNode* node, const GameState& state);
    int       _uct_pick (const MCTSNode* node) const;
    SelResult _select   (MCTSNode* root, const GameState& root_state);
    void      _backprop (SelResult& r, float value);
};
