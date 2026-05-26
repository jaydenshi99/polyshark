#include "game_state.h"
#include "unit_def.h"
#include "resource_def.h"
#include "tech_def.h"
#include "grid.h"

using nlohmann::json;

nlohmann::json GameState::serialise(const GameState& s) {
    const int mtsz = s.map_tiles();

    json j;
    j["turn"]           = s.get_turn();
    j["current_player"] = s.current_player();
    j["map_size"]       = s.map_size();

    json players = json::array();
    for (int p = 0; p < MAX_PLAYERS; p++) {
        json visible = json::array();
        for (int w = 0; w < FOG_WORDS; w++)
            visible.push_back((uint32_t)s.players[p].visible[w]);
        players.push_back({
            { "stars",      s.get_stars(p)   },
            { "techs_mask", s.techs_mask(p)  },
            { "visible",    visible          },
        });
    }
    j["players"] = players;

    // Tiles. Optional fields are omitted in their default state to keep
    // ordinary "blank field" tiles compact in the JSON.
    json tiles = json::array();
    for (int i = 0; i < mtsz; i++) {
        const Tile& t = s.tile_at(i);
        json tj = { { "i", i }, { "terrain", (int)t.terrain() } };
        if (t.has_unit())                          tj["unit_id"]        = t.unit_id();
        if (t.has_city())                          tj["city_id"]        = t.city_id();
        if (t.border_city_id() >= 0)               tj["border_city_id"] = t.border_city_id();
        if (t.resource() != ResourceType::None)    tj["resource"]       = (int)t.resource();
        if (t.building() != BuildingType::None)    tj["building"]       = (int)t.building();
        if (t.capture_ready())                     tj["capture_ready"]  = true;
        tiles.push_back(tj);
    }
    j["tiles"] = tiles;

    json units = json::array();
    for (int i = 0; i < s.unit_count; i++) {
        const Unit& u = s.units[i];
        if (!u.is_alive()) continue;
        units.push_back({
            { "id",              i                  },
            { "type",            (int)u.type()      },
            { "owner",           u.owner()          },
            { "tile",            u.tile_index()     },
            { "hp",              u.hp()             },
            { "max_hp",          u.max_hp()         },
            { "mp",              u.move_points()    },
            { "city_id",         u.city_id()        },
            { "kills",           u.kills()          },
            { "has_attacked",    u.has_attacked()   },
            { "promotion_ready", u.promotion_ready()},
            { "last_dir",        (int)u.last_dir()  },
        });
    }
    j["units"] = units;

    json cities = json::array();
    for (int i = 0; i < s.city_count; i++) {
        const City& c = s.cities[i];
        cities.push_back({
            { "id",               i                     },
            { "tile",             c.tile_index()        },
            { "owner",            c.owner()             },
            { "level",            c.level()             },
            { "population",       c.population()        },
            { "is_capital",       c.is_capital()        },
            { "border_radius",    c.border_radius()     },
            { "units_owned",      c.units_owned()       },
            { "parks",            c._parks              },
            { "is_sieged",        c.is_sieged()         },
            { "capture_ready",    c.capture_ready()     },
            { "has_walls",        c.has_walls()         },
            { "has_workshop",     c.has_workshop()      },
            { "pending_upgrade",  c.has_pending_upgrade()},
        });
    }
    j["cities"] = cities;
    return j;
}

// Mirror of serialise(). Anything serialise() drops (fog of war, attack/move
// flags, city walls/workshop/sieged state, tile resource/building, etc.) stays
// at default values here — round-tripping is lossy until serialise widens.
// Vision is approximated at the end by revealing around the player's own
// cities and units so the loaded state is at least playable.
GameState GameState::deserialise(const nlohmann::json& j) {
    extern void reveal_around(GameState& s, int player, int tile_index, int radius);

    GameState out;

    out.turn       = j.value("turn", 0);
    out.cur_player = j.value("current_player", 0);
    out._map_size  = j.value("map_size", (int)MAP_SIZE);

    // ---- Players ----
    if (j.contains("players")) {
        const auto& pjs = j["players"];
        for (int p = 0; p < MAX_PLAYERS && p < (int)pjs.size(); p++) {
            out.players[p].stars = pjs[p].value("stars", 0);
            uint32_t mask = pjs[p].value("techs_mask", 0u);
            for (int t = 0; t < 32; t++)
                if (mask & (1u << t)) out.players[p].research_tech((TechType)t);
            if (pjs[p].contains("visible") && pjs[p]["visible"].is_array()) {
                const auto& vj = pjs[p]["visible"];
                int n = std::min((int)vj.size(), (int)FOG_WORDS);
                for (int w = 0; w < n; w++)
                    out.players[p].visible[w] = (uint16_t)vj[w].get<uint32_t>();
            }
        }
    }

    // ---- Tiles ----
    const int mtsz = out.map_tiles();
    if (j.contains("tiles")) {
        for (const auto& tj : j["tiles"]) {
            int i = tj.value("i", -1);
            if (i < 0 || i >= mtsz) continue;
            Tile& t = out.map[i];
            t.set_terrain((TerrainType)tj.value("terrain", (int)TerrainType::Field));
            if (tj.contains("unit_id"))        t.place_unit(tj["unit_id"].get<int>());
            if (tj.contains("city_id"))        t.place_city(tj["city_id"].get<int>());
            if (tj.contains("border_city_id")) t.set_border_city(tj["border_city_id"].get<int>());
            if (tj.contains("resource"))       t.set_resource((ResourceType)tj["resource"].get<int>());
            if (tj.contains("building"))       t.set_building((BuildingType)tj["building"].get<int>());
            if (tj.contains("capture_ready"))  t.set_capture_ready(tj["capture_ready"].get<bool>());
        }
    }

    // ---- Cities. units_owned is serialised explicitly, so we restore it
    //      directly here instead of recounting in the units pass.
    if (j.contains("cities")) {
        const auto& cjs = j["cities"];
        out.city_count = std::min((int)cjs.size(), (int)MAX_MAP_TILES);
        for (int i = 0; i < out.city_count; i++) {
            const auto& cj = cjs[i];
            City& c = out.cities[i];
            c.set_tile(cj.value("tile", -1));
            c.set_owner(cj.value("owner", -1));
            c.set_capital(cj.value("is_capital", false));
            c._level           = cj.value("level", 1);
            c._population      = cj.value("population", 0);
            c._border_radius   = cj.value("border_radius", 1);
            c._units_owned     = cj.value("units_owned", 0);
            c._parks           = cj.value("parks", 0);
            c._is_sieged       = cj.value("is_sieged", false);
            c._capture_ready   = cj.value("capture_ready", false);
            c._has_walls       = cj.value("has_walls", false);
            c._has_workshop    = cj.value("has_workshop", false);
            c._pending_upgrade = cj.value("pending_upgrade", false);
            if (c.is_capital() && c.owner() >= 0 && c.owner() < MAX_PLAYERS)
                out.capital_city[c.owner()] = i;
        }
    }

    // ---- Units. The serialised array is sparse (dead slots skipped); pack
    //      live units back into their original ids so tile.unit_id refs remain
    //      valid, and bump unit_count to the highest id + 1.
    if (j.contains("units")) {
        int max_id = -1;
        for (const auto& uj : j["units"]) {
            int id = uj.value("id", -1);
            if (id < 0 || id >= (int)MAX_MAP_TILES) continue;
            if (id > max_id) max_id = id;
            Unit& u = out.units[id];
            u.set_type((UnitType)uj.value("type", (int)UnitType::None));
            u.set_owner(uj.value("owner", -1));
            u.set_tile(uj.value("tile", -1));
            u.set_max_hp(uj.value("max_hp", 0));
            u.set_hp(uj.value("hp", 0));
            u.set_city_id(uj.value("city_id", -1));
            u.set_last_dir((int8_t)uj.value("last_dir", -1));
            // No public setters for the rest — direct via friendship.
            u._move_points     = uj.value("mp", 0);
            u._kills           = uj.value("kills", 0);
            u._has_attacked    = uj.value("has_attacked", false);
            u._promotion_ready = uj.value("promotion_ready", false);
        }
        out.unit_count = max_id + 1;
    }
    return out;
}


void GameState::print(const GameState& s) {
    Logger::print("%s", serialise(s).dump().c_str());
}