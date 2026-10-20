// base — the smallest complete Polyshark bot. Copy this folder to start a new one.
//
// The lab gives act() the player's fog-of-war view as a real GameState plus the
// authoritative list of playable actions; the bot returns an index into that list.
// Everything the engine offers (legal_actions, apply_action, ...) works on the
// view, so search can be built directly on top of it.
#include "polyshark_bot.h"

class BaseBot : public psb::Bot {
public:
    void init(const psb::Hello& hello) override {
        player_ = hello.player;
        seed_ = hello.seed;
    }

    psb::Decision act(const psb::Observation& obs) override {
        psb::Decision d;
        d.index = choose(obs);
        return d;
    }

    // Position Analyser hook: scores every legal action by evaluate() one ply ahead.
    bool analyse(const psb::Observation& obs, const psb::Budget&, psb::Analysis& out) override {
        out.eval = evaluate(obs.view());
        for (int i = 0; i < obs.legal_count; i++) {
            GameState next = obs.view().apply_action(obs.legal[i]);
            out.moves.push_back({ i, evaluate(next), -1, "" });
        }
        return true;
    }

private:
    // Strategy goes here. For now: end the turn, or take the first option when a
    // city upgrade must be chosen (EndTurn is not offered until it is resolved).
    int choose(const psb::Observation& obs) {
        int end = obs.find(ActionType::EndTurn);
        return end >= 0 ? end : 0;
    }

    // Evaluation of `view` for this bot, in [-1, 1].
    float evaluate(const GameState& /*view*/) const { return 0.0f; }

    int      player_ = 0;
    uint64_t seed_   = 0;
};

POLYSHARK_BOT(BaseBot, "base", "1")
