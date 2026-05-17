#pragma once

#include "types.h"
#include "grid.h"
#include <cstdint>
#include <vector>

constexpr int FOG_WORDS = (MAX_MAP_TILES + 15) / 16;

struct Player {
    uint16_t explored[FOG_WORDS] = {};
    uint16_t visible[FOG_WORDS]  = {};
    uint32_t techs               = 0;
    int      stars               = 0;

    bool     is_explored(int tile) const { return (explored[tile / 16] >> (tile % 16)) & 1; }
    bool     is_visible(int tile)  const { return (visible[tile / 16]  >> (tile % 16)) & 1; }

    void set_explored(int tile) { explored[tile / 16] |= (1 << (tile % 16)); set_visible(tile); }
    void set_visible(int tile)  { visible[tile / 16]  |= (1 << (tile % 16)); }
    void clear_visible()        { for (int i = 0; i < FOG_WORDS; i++) visible[i] = 0; }

    bool     has_tech(TechType t)  const { return (techs >> static_cast<int>(t)) & 1; }
    void     research_tech(TechType t)   { techs |= (1u << static_cast<int>(t)); }
    uint32_t techs_mask()          const { return techs; }
    std::vector<TechType> get_techs() const {
        std::vector<TechType> out;
        uint32_t m = techs;
        while (m) {
            int i = __builtin_ctz(m);
            out.push_back(static_cast<TechType>(i));
            m &= m - 1;
        }
        return out;
    }
};
