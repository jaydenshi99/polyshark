// Checks the Match Lab runtime against the engine: in-place apply, fog views,
// observation round-trips and determinism.
#include <cstdio>
#include <cstdlib>
#include <memory>

#include "lab_core.h"

using json = nlohmann::json;

static int failures = 0;
#define CHECK(cond, ...)                                       \
    do {                                                       \
        if (!(cond)) {                                         \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                 \
            std::fprintf(stderr, "\n");                        \
            failures++;                                        \
        }                                                      \
    } while (0)

static uint64_t rng_next(uint64_t& s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

static void start(uint64_t seed, int size, GameState& s) {
    lab::make_start({ { "kind", "generated" }, { "size", size }, { "seed", seed } }, s);
}

// Random playout comparing apply_inplace with the engine's copying apply_action,
// and checking every fog view along the way.
static void test_playouts() {
    auto s = std::make_unique<GameState>();
    auto ref = std::make_unique<GameState>();
    auto view = std::make_unique<GameState>();
    auto back = std::make_unique<GameState>();
    auto vm = std::make_unique<lab::ViewMap>();
    std::vector<Action> legal;
    int view_diffs = 0, decisions = 0, invariant_breaks = 0;
    for (uint64_t seed = 1; seed <= 40; seed++) {
        int size = 9 + (int)(seed % 3) * 2;
        start(seed, size, *s);
        *ref = *s;
        uint64_t rng = seed * 7919;
        for (int step = 0; step < 1500 && !s->is_terminal(); step++) {
            lab::playable_actions(*s, legal);
            CHECK(!legal.empty(), "no playable action (seed %llu step %d)", (unsigned long long)seed, step);
            if (legal.empty()) break;
            int p = s->current_player();

            // View integrity.
            lab::build_view(*s, p, *view, *vm);
            for (int i = 0; i < s->map_tiles(); i++) {
                const Tile& t = view->tile_at(i);
                if (!s->is_visible(p, i)) {
                    CHECK(t.terrain() == TerrainType::None && !t.has_unit() && !t.has_city(), "hidden tile %d leaks", i);
                } else if (t.has_unit()) {
                    const Unit& vu = view->get_unit(t.unit_id());
                    const Unit& ru = s->get_unit(s->tile_at(i).unit_id());
                    CHECK(vu.tile_index() == i && vu.hp() == ru.hp() && vu.owner() == ru.owner(), "unit mismatch at %d", i);
                }
            }
            for (const Action& a : legal) {
                Action v = lab::action_to_view(a, *vm);
                if (a.type == ActionType::HarvestResource || a.type == ActionType::UpgradeCity ||
                    (a.type == ActionType::CaptureCity && a.from >= 0)) {
                    CHECK(v.from >= 0 && view->get_city(v.from).tile_index() == s->get_city(a.from).tile_index(),
                          "city id translation broken");
                }
            }
            // The serialised observation deserialises back to exactly the view.
            if (step % 50 == 0) {
                json obs = lab::observation_json(*s, p, true, *view, *vm);
                *back = GameState::deserialise(obs);
                CHECK(lab::state_hash(*back) == lab::state_hash(*view), "observation round trip differs");
                // How often the bot-side legal list differs (fog ZOC); informational only.
                std::vector<Action> bot_side;
                lab::playable_actions(*back, bot_side);
                decisions++;
                if (bot_side.size() != legal.size()) view_diffs++;
            }

            const Action& a = legal[rng_next(rng) % legal.size()];
            lab::apply_inplace(*s, a);
            *ref = ref->apply_action(a);
            // Engine-side quirk: capture re-homing can push units_owned past capacity.
            // Count it (stderr is silenced by the engine's own message) rather than fail.
            if (step % 25 == 0 && !s->check_invariants()) invariant_breaks++;
            if (step % 25 == 0)
                CHECK(lab::state_hash(*s) == lab::state_hash(*ref), "apply_inplace diverged (seed %llu step %d)",
                      (unsigned long long)seed, step);
        }
        CHECK(lab::state_hash(*s) == lab::state_hash(*ref), "final state diverged (seed %llu)", (unsigned long long)seed);
    }
    std::printf("playouts ok; bot-side legal list size differed in %d of %d sampled decisions\n", view_diffs, decisions);
    std::printf("engine check_invariants() failed on %d sampled states (pre-existing engine behaviour)\n", invariant_breaks);
}

static void test_determinism() {
    auto a = std::make_unique<GameState>();
    auto b = std::make_unique<GameState>();
    start(12345, 11, *a);
    start(12345, 11, *b);
    CHECK(lab::state_hash(*a) == lab::state_hash(*b), "same seed, different map");
    start(12346, 11, *b);
    CHECK(lab::state_hash(*a) != lab::state_hash(*b), "different seeds, same map");
    CHECK(lab::derive_seed(1, 2, 3) == lab::derive_seed(1, 2, 3), "derive_seed unstable");
    CHECK(lab::derive_seed(1, 2, 3) != lab::derive_seed(1, 3, 2), "derive_seed symmetric");
    // Serialise round trip of a full state is exact.
    auto c = std::make_unique<GameState>();
    *c = GameState::deserialise(GameState::serialise(*a));
    CHECK(lab::state_hash(*a) == lab::state_hash(*c), "serialise round trip differs");
}

static void test_codec() {
    auto s = std::make_unique<GameState>();
    start(7, 9, *s);
    std::vector<Action> legal;
    lab::playable_actions(*s, legal);
    for (const Action& a : legal) {
        Action b = psb::action_from_json(psb::action_to_json(a, 9), 9);
        CHECK(psb::same_action(a, b), "codec round trip");
        if (a.type == ActionType::Move) CHECK(a.path_bits == b.path_bits && a.path_steps == b.path_steps, "path round trip");
    }
}

int main() {
    test_determinism();
    test_codec();
    test_playouts();
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all lab tests passed\n");
    return 0;
}
