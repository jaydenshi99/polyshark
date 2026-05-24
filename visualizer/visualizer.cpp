#include "raylib.h"
#include "rlgl.h"
#include "mapgen.h"
#include "unit_def.h"
#include "resource_def.h"
#include "building_def.h"
#include "tech_def.h"
#include "game_state.h"
#include "logger.h"
#include <utility>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>
#include <string>

// Isometric tiles: each tile is a diamond with TILE_W (full width) and TILE_H (full height).
// Aspect ratio matches the grass terrain sprite (840x507 ≈ 5:3 = 1.667). Both values are
// recomputed each map gen so the diamond grid stretches to fill MAP_CANVAS horizontally.
static int            TILE_W   = 56;   // diamond full width  (recomputed; do not make constexpr)
static int            TILE_H   = 34;   // diamond full height (recomputed; ~ TILE_W * 3/5)
static constexpr int PAD      = 4;
static constexpr int LABEL    = 20;   // (legacy padding; iso layout doesn't draw axis labels)
static constexpr int TOP_HUD  = 64;   // info bar across the top
static constexpr int SIDEBAR  = 230;
static constexpr int TECH_W   = 210;  // width of empty buffer column on the left (was tech tree)
static constexpr int MAP_OFF  = TECH_W;  // map x-offset
// Fixed canvas: constant pixel budget. TILE_W/TILE_H scale to fill this regardless of map_size.
static constexpr int MAP_CANVAS = 616;
static constexpr int MAP_PX     = MAP_CANVAS + PAD * 2 + LABEL;
static constexpr int LOG_W      = 260;
static constexpr int TECH_H     = 150;  // bottom tech tree strip height
static constexpr int W          = TECH_W + MAP_PX + SIDEBAR + LOG_W;
// CONTENT_H = bottom of the main content area (map / sidebar / log).
// H = full window height including the bottom tech strip. Existing panels anchor to
// CONTENT_H so they don't extend into the tech strip; only the strip itself uses H.
static constexpr int CONTENT_H  = TOP_HUD + MAP_CANVAS + PAD * 2 + LABEL;
static constexpr int H          = CONTENT_H + TECH_H;

enum class ViewMode { Omni, P0, P1, Current };

// Reusable colors
static constexpr Color PANEL_BG   = {  22,  22,  22, 255 };
static constexpr Color PANEL_LINE = {  65,  65,  65, 255 };
static constexpr Color SIDEBAR_BG = {  38,  38,  38, 255 };
static constexpr Color COL_P0     = {  80, 140, 220, 255 };
static constexpr Color COL_P1     = { 220,  80,  80, 255 };

// Isometric layout (diamond grid).
//   Grid axes:
//     (r=0,  c=0)  → BOTTOM of the diamond
//     (r=N,  c=N)  → TOP
//     (r=0,  c=N)  → RIGHT
//     (r=N,  c=0)  → LEFT
//   Tile (r, c) center in screen space:
//     cx = origin_x + (c - r) * TILE_W/2
//     cy = origin_y - (c + r) * TILE_H/2
//   The grid spans from the LEFT edge of the window (reclaiming the former empty
//   buffer column above) out to the right edge of MAP_CANVAS. It's centred vertically
//   between TOP_HUD and CONTENT_H, with (0,0) at the bottom of the centred grid.
static inline int iso_grid_left()    { return 0; }
static inline int iso_grid_right()   { return MAP_OFF + PAD + LABEL + MAP_CANVAS; }
static inline int iso_grid_width()   { return iso_grid_right() - iso_grid_left(); }
static inline int iso_grid_top()     { return TOP_HUD; }
static inline int iso_grid_bottom()  { return CONTENT_H; }
static inline int iso_grid_height()  { return iso_grid_bottom() - iso_grid_top(); }

static inline Vector2 tile_center(int r, int c, int msz) {
    float origin_x = (float)iso_grid_left() + iso_grid_width() * 0.5f;
    float grid_h   = (float)msz * (float)TILE_H;
    float top_pad  = (iso_grid_height() - grid_h) * 0.5f;
    float origin_y = (float)iso_grid_top() + top_pad + grid_h - (float)TILE_H * 0.5f;
    float cx = origin_x + (float)(c - r) * (float)TILE_W * 0.5f;
    float cy = origin_y - (float)(c + r) * (float)TILE_H * 0.5f;
    return { cx, cy };
}

// Inverse of tile_center. Returns (r, c) of the tile containing (mx, my) via out
// params, or (-1, -1) if the point is outside the grid. Uses the fact that each
// tile's Voronoi region is exactly its diamond shape, so rounding the continuous
// (r, c) coordinates picks the right tile.
static inline void tile_from_screen(float mx, float my, int msz, int& out_r, int& out_c) {
    float origin_x = (float)iso_grid_left() + iso_grid_width() * 0.5f;
    float grid_h   = (float)msz * (float)TILE_H;
    float top_pad  = (iso_grid_height() - grid_h) * 0.5f;
    float origin_y = (float)iso_grid_top() + top_pad + grid_h - (float)TILE_H * 0.5f;
    float u_f = 2.0f * (mx - origin_x) / (float)TILE_W;     // c - r
    float s_f = 2.0f * (origin_y - my) / (float)TILE_H;     // c + r
    float c_f = (s_f + u_f) * 0.5f;
    float r_f = (s_f - u_f) * 0.5f;
    int rr = (int)floorf(r_f + 0.5f);
    int cc = (int)floorf(c_f + 0.5f);
    if (rr < 0 || cc < 0 || rr >= msz || cc >= msz) { out_r = -1; out_c = -1; return; }
    out_r = rr; out_c = cc;
}

// --- Per-unit-type visual tweaks ------------------------------------------------
// Layered ON TOP of the auto-bbox centering done at sprite-load time. Use this table
// to nudge a unit's resting position or shrink/grow it on the tile without changing
// the source PNGs. Offsets are in TILE-FRACTIONS (so they scale with map size):
//   x_offset =  0.10  → shift right by 10% of TILE_W
//   y_offset = -0.05  → shift up    by  5% of TILE_H
//   scale    =  1.15  → 15% larger than the default auto-scale (1.0 = no change)
// Generic tile-relative tweak. Anything drawn on a tile (sprite, icon, label,
// badge) that needs to be tweakable gets a Transformer instance. Convention:
// any new tile-anchored element should declare its own Transformer constant so
// it stays tweakable without touching draw code.
struct Transformer {
    float x_offset;
    float y_offset;
    float scale;
};

// Per-unit-type sprite transformer. Layered on top of the auto-bbox centering done
// at sprite-load time.
static const Transformer UNIT_TRANSFORMERS[(int)UnitType::Count] = {
    /* None     */ { 0.00f,  0.00f, 0.75f },
    /* Warrior  */ { -0.03f, -0.04f, 0.75f },
    /* Archer   */ { 0.05f, -0.05f, 0.76f },
    /* Rider    */ { 0.00f, -0.08f, 0.65f },
    /* Defender */ { 0.00f, -0.08f, 0.78f },
};

// HP number transformer (shared by all units). Anchored at the top of the diamond
// by default; scale multiplies the 11pt base font.
static const Transformer HP_TRANSFORMER = { -0.22f, 0.24f, 1.00f };

// RingTransformer — like Transformer, plus an opacity multiplier (0..1) applied to
// the ring's color alpha at draw time. Rings tend to need their visual weight tuned
// independently, so this gives a single dial per-ring.
struct RingTransformer {
    float x_offset;
    float y_offset;
    float scale;
    float opacity;
};

// "Ready" ring — cyan/yellow ellipse under units that haven't moved this turn.
// Base ring radius is 35% of TILE_W / TILE_H; scale multiplies that.
static const RingTransformer READY_RING_TRANSFORMER = { 0.00f, 0.14f, 0.6f, 1.00f };

// Movement-ring object — shown on every tile a selected unit can move to. Three
// concentric bands, all fractions of TILE_W (the helper squishes them vertically
// by TILE_H/TILE_W so they lie flat on the iso ground):
//   1. solid inner ellipse                          (cyan)
//   2. white donut between r=white_inner..white_outer
//   3. outer cyan donut between r=donut_inner..donut_outer
// The Transformer shifts/scales the whole composite.
struct MovementRing {
    float inner_solid_r;   // solid inner ellipse radius
    float white_inner_r;   // white donut: inner edge
    float white_outer_r;   // white donut: outer edge
    float donut_inner_r;   // outer cyan donut: inner edge
    float donut_outer_r;   // outer cyan donut: outer edge
    RingTransformer transform;
};
static const MovementRing MOVEMENT_RING = {
    /* inner_solid_r */ 0.09f,
    /* white_inner_r */ 0.09f,
    /* white_outer_r */ 0.18f,
    /* donut_inner_r */ 0.18f,
    /* donut_outer_r */ 0.27f,
    /* transform     */ { 0.00f, 0.00f, 1.00f, 1.00f },
};

// Multiplies a colour's alpha by `mul` (clamped to [0, 255]).
static inline Color color_alpha_mul(Color c, float mul) {
    int a = (int)(c.a * mul + 0.5f);
    if (a < 0)   a = 0;
    if (a > 255) a = 255;
    c.a = (unsigned char)a;
    return c;
}

static inline void draw_movement_ring(Vector2 ctr, Color color) {
    const MovementRing& m = MOVEMENT_RING;
    const RingTransformer& rt = m.transform;
    float scale  = rt.scale;
    float cx     = ctr.x + rt.x_offset * (float)TILE_W;
    float cy     = ctr.y + rt.y_offset * (float)TILE_H;
    float aspect = (float)TILE_H / (float)TILE_W;

    Color cyan_col  = color_alpha_mul(color, rt.opacity);
    Color white_col = color_alpha_mul(Color{ 255, 255, 255, 180 }, rt.opacity);

    // Solid inner ellipse (axis-aligned, so we can use raylib's primitive directly).
    float ir_w = m.inner_solid_r * (float)TILE_W * scale;
    float ir_h = ir_w * aspect;
    DrawEllipse((int)cx, (int)cy, ir_w, ir_h, cyan_col);

    // White accent donut + outer cyan donut. DrawRing only produces circular rings,
    // so we squish the matrix vertically to flatten both into iso-shaped ellipses.
    rlPushMatrix();
    rlTranslatef(cx, cy, 0);
    rlScalef(1.0f, aspect, 1.0f);
    DrawRing({ 0, 0 },
             m.white_inner_r * (float)TILE_W * scale,
             m.white_outer_r * (float)TILE_W * scale,
             0.0f, 360.0f, 48, white_col);
    DrawRing({ 0, 0 },
             m.donut_inner_r * (float)TILE_W * scale,
             m.donut_outer_r * (float)TILE_W * scale,
             0.0f, 360.0f, 48, cyan_col);
    rlPopMatrix();
}

// --- Terrain sprites -----------------------------------------------------------
// Each non-water tile gets a terrain sprite drawn anchored to the diamond bottom.
// Mountains (taller than the tile diamond) extend above; back-to-front iteration
// guarantees a closer mountain overdraws tiles behind it.
enum class TerrainSprite { Grass = 0, Mountain, Count };
struct TerrainAlphaBBox { int min_x, max_x, min_y, max_y; };
static Texture2D       terrain_tex [(int)TerrainSprite::Count] = {};
static TerrainAlphaBBox terrain_bbox[(int)TerrainSprite::Count] = {};

// Per-sprite tweak — same Transformer convention as units/rings. Default = the
// alpha bbox is scaled so its width == TILE_W and its bottom sits at the diamond
// bottom-centre.
static Transformer TERRAIN_TRANSFORMERS[(int)TerrainSprite::Count] = {
    /* Grass    */ { 0.00f, 0.00f, 1.00f },
    /* Mountain */ { 0.00f, 0.00f, 1.00f },
};

static inline void draw_terrain_sprite(Vector2 ctr, TerrainSprite ts) {
    const Texture2D& tex = terrain_tex[(int)ts];
    if (tex.id == 0) return;
    const TerrainAlphaBBox& bb = terrain_bbox[(int)ts];
    const Transformer&      xf = TERRAIN_TRANSFORMERS[(int)ts];

    int bbox_w = bb.max_x - bb.min_x;
    if (bbox_w <= 0) return;

    // Scale so the bbox width matches TILE_W * scale (sub 1.0 = smaller than tile).
    float scale = (float)TILE_W * xf.scale / (float)bbox_w;
    float dst_w = tex.width  * scale;
    float dst_h = tex.height * scale;

    // Anchor the bbox bottom-centre to the diamond's bottom point (cx, cy + TILE_H/2).
    float bbox_cx_src   = (bb.min_x + bb.max_x) * 0.5f;
    float bbox_bot_src  = (float)bb.max_y;
    float anchor_x = ctr.x + xf.x_offset * (float)TILE_W;
    float anchor_y = ctr.y + xf.y_offset * (float)TILE_H + (float)TILE_H * 0.5f;
    float dst_x = anchor_x - bbox_cx_src  * scale;
    float dst_y = anchor_y - bbox_bot_src * scale;

    DrawTexturePro(tex,
        { 0, 0, (float)tex.width, (float)tex.height },
        { dst_x, dst_y, dst_w, dst_h },
        { 0, 0 }, 0.0f, WHITE);
}

// --- Resource sprites ----------------------------------------------------------
// Overlays drawn on top of the terrain pass for tiles that hold a harvestable
// resource. Same alpha-bbox + bottom-anchor approach as terrain, just with a
// smaller default footprint (~50% of TILE_W) so the sprite sits on the tile
// without dominating it.
//
// Forest is included here even though it's a TerrainType (not a ResourceType) —
// rendering it as an overlay on grass keeps the base ground layer simple and
// lets the trees sit naturally on top of the field.
enum class ResourceSprite { Fruit = 0, Crop, Animal, Metal, Forest, Count };
static Texture2D        resource_tex [(int)ResourceSprite::Count] = {};
static TerrainAlphaBBox resource_bbox[(int)ResourceSprite::Count] = {};

// Each entry tweaks placement (offsets in tile-fractions) and scale (multiplier
// on the 50%-of-TILE_W base). Tune these to nudge sprites without re-exporting PNGs.
static Transformer RESOURCE_TRANSFORMERS[(int)ResourceSprite::Count] = {
    /* Fruit  */ { 0.00f, -0.4f,  0.7f },
    /* Crop   */ { 0.00f, -0.2f,  1.4f },
    /* Animal */ { 0.00f, -0.35f, 0.4f },
    /* Metal  */ { 0.00f, -0.38f, 0.53f },
    /* Forest */ { 0.02f, -0.1f,  1.5f },
};

static inline ResourceSprite resource_type_to_sprite(ResourceType r) {
    switch (r) {
        case ResourceType::Fruit:  return ResourceSprite::Fruit;
        case ResourceType::Crop:   return ResourceSprite::Crop;
        case ResourceType::Animal: return ResourceSprite::Animal;
        case ResourceType::Metal:  return ResourceSprite::Metal;
        default:                   return ResourceSprite::Count;
    }
}

static inline void draw_resource_sprite(Vector2 ctr, ResourceSprite rs) {
    if (rs == ResourceSprite::Count) return;
    const Texture2D& tex = resource_tex[(int)rs];
    if (tex.id == 0) return;
    const TerrainAlphaBBox& bb = resource_bbox[(int)rs];
    const Transformer&      xf = RESOURCE_TRANSFORMERS[(int)rs];

    int bbox_w = bb.max_x - bb.min_x;
    if (bbox_w <= 0) return;

    // Base footprint = 50% of TILE_W; Transformer scale multiplies on top.
    float scale = (float)TILE_W * 0.5f * xf.scale / (float)bbox_w;
    float dst_w = tex.width  * scale;
    float dst_h = tex.height * scale;

    float bbox_cx_src  = (bb.min_x + bb.max_x) * 0.5f;
    float bbox_bot_src = (float)bb.max_y;
    float anchor_x = ctr.x + xf.x_offset * (float)TILE_W;
    float anchor_y = ctr.y + xf.y_offset * (float)TILE_H + (float)TILE_H * 0.5f;
    float dst_x = anchor_x - bbox_cx_src  * scale;
    float dst_y = anchor_y - bbox_bot_src * scale;

    DrawTexturePro(tex,
        { 0, 0, (float)tex.width, (float)tex.height },
        { dst_x, dst_y, dst_w, dst_h },
        { 0, 0 }, 0.0f, WHITE);
}

// --- Captured-city marker ------------------------------------------------------
// Simple flag-on-pole indicator drawn on any tile that hosts an established city
// (Tile::has_city() — uncaptured villages are TerrainType::Village without a
// city object). The flag is filled in the owner's player colour.
//   pole_h_frac : pole height as fraction of TILE_H
//   flag_w_frac : flag width  as fraction of TILE_W
//   flag_h_frac : flag height as fraction of TILE_H
//   pole_thick  : pole thickness in pixels (before transformer scale)
struct CityMarkerSpec {
    float pole_h_frac;
    float flag_w_frac;
    float flag_h_frac;
    float pole_thick;
};
static Transformer          CITY_MARKER_TRANSFORMER = { 0.00f, -0.05f, 1.00f };
static const CityMarkerSpec CITY_MARKER_SPEC = {
    /* pole_h_frac */ 0.60f,
    /* flag_w_frac */ 0.18f,
    /* flag_h_frac */ 0.22f,
    /* pole_thick  */ 2.0f,
};

static inline void draw_city_marker(Vector2 ctr, int owner) {
    const Transformer&    xf = CITY_MARKER_TRANSFORMER;
    const CityMarkerSpec& sp = CITY_MARKER_SPEC;
    float scale = xf.scale;

    float anchor_x = ctr.x + xf.x_offset * (float)TILE_W;
    float anchor_y = ctr.y + xf.y_offset * (float)TILE_H + (float)TILE_H * 0.5f;
    float pole_h = (float)TILE_H * sp.pole_h_frac * scale;
    float flag_w = (float)TILE_W * sp.flag_w_frac * scale;
    float flag_h = (float)TILE_H * sp.flag_h_frac * scale;
    float pole_top = anchor_y - pole_h;

    DrawLineEx({ anchor_x, anchor_y }, { anchor_x, pole_top },
               sp.pole_thick * scale, { 40, 30, 20, 240 });

    Color flag_col = (owner == 0) ? COL_P0 : COL_P1;
    Vector2 v1 = { anchor_x,          pole_top };
    Vector2 v2 = { anchor_x + flag_w, pole_top + flag_h * 0.5f };
    Vector2 v3 = { anchor_x,          pole_top + flag_h };
    DrawTriangle(v1, v2, v3, flag_col);
    Vector2 outline[] = { v1, v2, v3, v1 };
    DrawLineStrip(outline, 4, BLACK);
}

static inline void draw_diamond(Vector2 ctr, Color fill) {
    Vector2 top   = { ctr.x,                  ctr.y - TILE_H * 0.5f };
    Vector2 right = { ctr.x + TILE_W * 0.5f,  ctr.y                 };
    Vector2 bot   = { ctr.x,                  ctr.y + TILE_H * 0.5f };
    Vector2 left  = { ctr.x - TILE_W * 0.5f,  ctr.y                 };
    // Two triangles fan from the top vertex. Winding kept CCW in raylib's coord system.
    DrawTriangle(top, left, bot, fill);
    DrawTriangle(top, bot, right, fill);
}

static inline void draw_diamond_outline(Vector2 ctr, Color outline, float thick) {
    Vector2 top   = { ctr.x,                  ctr.y - TILE_H * 0.5f };
    Vector2 right = { ctr.x + TILE_W * 0.5f,  ctr.y                 };
    Vector2 bot   = { ctr.x,                  ctr.y + TILE_H * 0.5f };
    Vector2 left  = { ctr.x - TILE_W * 0.5f,  ctr.y                 };
    DrawLineEx(top,   right, thick, outline);
    DrawLineEx(right, bot,   thick, outline);
    DrawLineEx(bot,   left,  thick, outline);
    DrawLineEx(left,  top,   thick, outline);
}

static std::string format_action_str(const Action& a, int player, int turn, int sz) {
    auto coords = [sz](int idx) {
        char buf[12]; snprintf(buf, sizeof(buf), "(%d,%d)", idx / sz, idx % sz); return std::string(buf);
    };
    static const char* unit_names[] = {"?","Warrior","Archer","Rider","Defender"};
    static const char* tech_names[] = {"Origin","Hunting","Org","Farming","Riding","Climb","Archery","Mining"};
    char prefix[24]; snprintf(prefix, sizeof(prefix), "P%d T%d: ", player, turn);
    std::string p = prefix;
    switch (a.type) {
        case ActionType::Move:              return p + "Move "    + coords(a.from) + "->" + coords(a.to);
        case ActionType::Attack:            return p + "Attack "  + coords(a.from) + "->" + coords(a.to);
        case ActionType::CaptureCity:       return p + "Capture " + coords(a.to);
        case ActionType::HarvestResource:   return p + "Harvest " + coords(a.from);
        case ActionType::ConstructBuilding: return p + "Build @ " + coords(a.from);
        case ActionType::UpgradeCity: {
            static const char* unames[] = {"Workshop","Explorer","Resources","Walls","BorderGrowth","PopGrowth","Park","SuperUnit"};
            return p + "Upgrade: " + (a.param >= 0 && a.param < 8 ? unames[a.param] : "?");
        }
        case ActionType::EndTurn:           return p + "EndTurn";
        case ActionType::TrainUnit:
            return p + "Train " + (a.param > 0 && a.param < (int)UnitType::Count ? unit_names[a.param] : "?") + " @ " + coords(a.from);
        case ActionType::ResearchTech:
            return p + "Tech " + (a.param >= 0 && a.param < 8 ? tech_names[a.param] : "?");
        default: return p + "Action";
    }
}

// ---------------------------------------------------------------------------
// Colourer — assigns each city a distinct-but-similar colour and tracks which
// tiles fall inside that city's territory.
// ---------------------------------------------------------------------------
struct Colourer {
    static constexpr int PALETTE_SIZE = 8;

    Color palette[PALETTE_SIZE]                 = {};
    int   next_idx                              = 0;
    int   city_palette_idx[MAX_MAP_TILES]       = {};
    int   city_border_size[MAX_MAP_TILES]       = {};

    void init(Color base) {
        next_idx = 0;
        for (int i = 0; i < MAX_MAP_TILES; i++) { city_palette_idx[i] = -1; city_border_size[i] = -1; }
        // Build palette: variations around base colour
        auto clamp = [](int v) -> uint8_t { return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v; };
        for (int i = 0; i < PALETTE_SIZE; i++) {
            int dr = (i * 23) % 60 - 30;
            int dg = (i * 17) % 60 - 30;
            int db = (i * 13) % 60 - 30;
            palette[i] = { clamp(base.r + dr), clamp(base.g + dg), clamp(base.b + db), 30 };
        }
    }

    // Returns current palette colour and advances the index.
    Color get_colour() {
        Color c = palette[next_idx];
        next_idx = (next_idx + 1) % PALETTE_SIZE;
        return c;
    }

    // Assign the next palette slot to this city (call once per city on init).
    void assign_city(int city_id) {
        city_palette_idx[city_id] = next_idx;
        next_idx = (next_idx + 1) % PALETTE_SIZE;
    }

    // Returns the territory colour for this tile, or BLANK if unclaimed.
    Color tile_colour(int tile_idx, const GameState& s) const {
        int cid = s.tile_at(tile_idx).border_city_id();
        if (cid < 0 || city_palette_idx[cid] < 0) return BLANK;
        return palette[city_palette_idx[cid]];
    }
};

static Color terrain_color(TerrainType t) {
    switch (t) {
        case TerrainType::Field:    return { 162, 210, 110, 255 };
        case TerrainType::Forest:   return {  52, 130,  52, 255 };
        case TerrainType::Mountain: return { 160, 145, 120, 255 };
        case TerrainType::Water:    return {  70, 140, 210, 255 };
        case TerrainType::Village:  return { 210, 180,  80, 255 };
        default:                    return BLACK;
    }
}


static void action_cost(const Action& a, char* buf, int size, int owned_cities) {
    switch (a.type) {
        case ActionType::TrainUnit:
            snprintf(buf, size, "%d*", unit_def(static_cast<UnitType>(a.param)).cost);
            break;
        case ActionType::HarvestResource:
            snprintf(buf, size, "%d*",
                resource_def(static_cast<ResourceType>(a.param)).star_cost);
            break;
        case ActionType::ResearchTech:
            snprintf(buf, size, "%d*",
                tech_cost(static_cast<TechType>(a.param), owned_cities));
            break;
        default:
            buf[0] = '\0';
            break;
    }
}

enum class ActionCategory { None, Train, Move, Attack, Capture, Harvest, Research, Upgrade, Debug, EndTurn };

static ActionCategory action_category(const Action& a) {
    switch (a.type) {
        case ActionType::TrainUnit:       return ActionCategory::Train;
        case ActionType::Move:            return ActionCategory::Move;
        case ActionType::Attack:          return ActionCategory::Attack;
        case ActionType::CaptureCity:     return ActionCategory::Capture;
        case ActionType::HarvestResource: return ActionCategory::Harvest;
        case ActionType::ResearchTech:    return ActionCategory::Research;
        case ActionType::UpgradeCity:     return ActionCategory::Upgrade;
        case ActionType::DebugAddPop:     return ActionCategory::Debug;
        case ActionType::EndTurn:         return ActionCategory::EndTurn;
        default:                          return ActionCategory::None;
    }
}

static const char* category_name(ActionCategory c) {
    switch (c) {
        case ActionCategory::Train:   return "Train Unit";
        case ActionCategory::Move:    return "Move";
        case ActionCategory::Attack:  return "Attack";
        case ActionCategory::Capture: return "Capture";
        case ActionCategory::Harvest: return "Harvest";
        case ActionCategory::Research:return "Research";
        case ActionCategory::Upgrade: return "Upgrade City";
        case ActionCategory::Debug:   return "Debug";
        case ActionCategory::EndTurn: return "End Turn";
        default:                      return nullptr;
    }
}

static void action_label(const Action& a, char* buf, int size) {
    int fx, fy, tx, ty;
    switch (a.type) {
        case ActionType::EndTurn:
            snprintf(buf, size, "End Turn (SPACE)");
            break;
        case ActionType::Move:
            to_coords(a.from, fx, fy); to_coords(a.to, tx, ty);
            snprintf(buf, size, "(%d,%d) -> (%d,%d)", fx, fy, tx, ty);
            break;
        case ActionType::Attack:
            to_coords(a.from, fx, fy); to_coords(a.to, tx, ty);
            snprintf(buf, size, "(%d,%d) -> (%d,%d)", fx, fy, tx, ty);
            break;
        case ActionType::TrainUnit: {
            static const char* unit_names[] = { "?", "Warrior", "Archer", "Rider", "Defender" };
            const char* uname = (a.param > 0 && a.param < (int)UnitType::Count) ? unit_names[a.param] : "?";
            snprintf(buf, size, "%s", uname);
            break;
        }
        case ActionType::ResearchTech:
            snprintf(buf, size, "%s", tech_def(static_cast<TechType>(a.param)).name);
            break;
        case ActionType::CaptureCity:
            to_coords(a.to, tx, ty);
            snprintf(buf, size, "Capture (%d,%d)", tx, ty);
            break;
        case ActionType::HarvestResource: {
            to_coords(a.to, tx, ty);
            snprintf(buf, size, "%s (%d,%d)",
                resource_def(static_cast<ResourceType>(a.param)).name, tx, ty);
            break;
        }
        case ActionType::DebugAddPop:
            snprintf(buf, size, "+%d pop city %d", a.param, a.from);
            break;
        case ActionType::UpgradeCity: {
            static const char* upgrade_names[] = {
                "Workshop",        "Explorer",
                "Resources (+5)",  "Walls",
                "Border Growth",   "Population Growth",
                "Park",            "Super Unit",
            };
            const char* uname = (a.param >= 0 && a.param < 8) ? upgrade_names[a.param] : "?";
            snprintf(buf, size, "City %d: %s", a.from, uname);
            break;
        }
        default:
            snprintf(buf, size, "Unknown");
            break;
    }
}

struct SidebarRow {
    bool           is_header;
    ActionCategory category;
    int            action_idx;
    int            y;
};

static constexpr int ROW_H       = 26;
static constexpr int HEADER_H    = 18;
static constexpr int SIDEBAR_TOP = TOP_HUD + 38;

static int build_sidebar_layout(const Action* actions, int count,
                                SidebarRow* rows, int max_rows) {
    int n = 0, y = SIDEBAR_TOP;
    ActionCategory cur = ActionCategory::None;

    // Pass 1: everything except Debug and EndTurn
    for (int i = 0; i < count && n < max_rows - 1; i++) {
        ActionCategory cat = action_category(actions[i]);
        if (cat == ActionCategory::Debug || cat == ActionCategory::EndTurn) continue;
        if (cat != ActionCategory::None && cat != cur) {
            cur = cat;
            rows[n++] = { true, cat, -1, y };
            y += HEADER_H;
        }
        rows[n++] = { false, cat, i, y };
        y += ROW_H;
    }

    // Pass 2: Debug
    cur = ActionCategory::None;
    for (int i = 0; i < count && n < max_rows - 1; i++) {
        ActionCategory cat = action_category(actions[i]);
        if (cat != ActionCategory::Debug) continue;
        if (cur != ActionCategory::Debug) {
            cur = ActionCategory::Debug;
            rows[n++] = { true, cat, -1, y };
            y += HEADER_H;
        }
        rows[n++] = { false, cat, i, y };
        y += ROW_H;
    }

    // Pass 3: EndTurn — single clickable header
    for (int i = 0; i < count && n < max_rows - 1; i++) {
        if (action_category(actions[i]) != ActionCategory::EndTurn) continue;
        rows[n++] = { true, ActionCategory::EndTurn, i, y };
        y += HEADER_H;
        break;
    }

    return n;
}


// ---------------------------------------------------------------------------
// Tech tree layout
// Tiers:    0=Origin  1=Hunting,Organisation,Riding,Climbing  2=Archery,Mining
// Positions hard-coded to match the TECH_DEFS order.
// ---------------------------------------------------------------------------
struct TechNode {
    TechType type;
    int      tier;   // 0,1,2
    float    slot;   // fractional column within tier (0-based)
    int      parent; // index into NODES, or -1
};

static const TechNode NODES[] = {
    // idx  type                      tier  slot  parent
    { TechType::Origin,       0,  0.0f,  -1 },  // 0
    { TechType::Hunting,      1,  0.0f,   0 },  // 1
    { TechType::Organisation, 1,  1.0f,   0 },  // 2
    { TechType::Riding,       1,  2.0f,   0 },  // 3
    { TechType::Climbing,     1,  3.0f,   0 },  // 4
    { TechType::Archery,      2,  0.5f,   1 },  // 5  child of Hunting
    { TechType::Farming,      2,  1.5f,   2 },  // 6  child of Organisation
    { TechType::Mining,       2,  3.5f,   4 },  // 7  child of Climbing
};
static constexpr int NODE_COUNT = 8;

// hover_tech: -1 = none, else TechType int being hovered for research
// scale: zoom factor (1.0 = default); >1 = zoomed in, nodes spread farther apart
// Draws the tech tree into the rect (rx, ry, rw, rh). Tiers stack top → bottom (Origin
// on top); slot positions spread left → right within each tier row.
static void draw_tech_tree(uint32_t owned_techs, int player_idx,
                           int rx, int ry, int rw, int rh, const char* label,
                           int hover_tech = -1, float scale = 1.0f)
{
    const Color owned_col  = (player_idx == 0) ? COL_P0 : COL_P1;
    const Color avail_col  = { 130, 130, 130, 255 };
    const Color locked_col = {  55,  55,  55, 255 };
    const Color edge_col   = {  70,  70,  70, 255 };
    const Color hover_col  = { 255, 220,  60, 255 };

    constexpr int   TIERS   = 3;
    constexpr int   FONT_SZ = 9;
    const float     NODE_R  = 11.0f * scale;

    int header_h = (label != nullptr) ? 16 : 0;

    if (label) {
        int lw = MeasureText(label, FONT_SZ + 2);
        DrawText(label, rx + rw / 2 - lw / 2, ry + 2, FONT_SZ + 2, owned_col);
    }

    // Content area (below header). The node label sits just below each node, so we
    // reserve `label_room` of vertical margin at the bottom so the last tier's name fits.
    float content_y  = (float)ry + header_h;
    float content_h  = (float)rh - header_h;
    float label_room = NODE_R + FONT_SZ + 4;

    // Find the largest slot value used so we can map slots into the available width.
    // (Mining sits at slot 3.5 — the layout must accommodate that.)
    float max_slot = 0.0f;
    for (int i = 0; i < NODE_COUNT; i++)
        if (NODES[i].slot > max_slot) max_slot = NODES[i].slot;

    // Vertical: tiers stack from top to bottom across content_h, leaving label_room
    // at the bottom so the last tier's tech name doesn't get clipped.
    float v_top    = content_y + NODE_R;
    float v_bot    = content_y + content_h - label_room;
    float v_range  = v_bot - v_top;
    auto tier_y = [&](int tier) -> float {
        if (TIERS <= 1) return v_top;
        return v_top + (tier / (float)(TIERS - 1)) * v_range;
    };

    // Horizontal: slots 0..max_slot spread across content width. We inset extra padding
    // beyond the node radius so the outermost techs don't crowd the panel border.
    const float H_MARGIN = NODE_R + 32.0f;
    float h_left  = rx + H_MARGIN;
    float h_right = rx + rw - H_MARGIN;
    float h_range = h_right - h_left;
    auto node_x = [&](float slot, int tier) -> float {
        if (tier == 0) return rx + rw * 0.5f;
        return h_left + (max_slot > 0 ? slot / max_slot : 0.0f) * h_range;
    };

    uint32_t avail = available_techs(owned_techs);

    // Edges first so nodes draw on top.
    for (int i = 0; i < NODE_COUNT; i++) {
        if (NODES[i].parent < 0) continue;
        int p = NODES[i].parent;
        float x1 = node_x(NODES[p].slot, NODES[p].tier);
        float y1 = tier_y(NODES[p].tier);
        float x2 = node_x(NODES[i].slot, NODES[i].tier);
        float y2 = tier_y(NODES[i].tier);
        bool  is_hover_edge = (hover_tech == static_cast<int>(NODES[i].type));
        Color ec = is_hover_edge ? hover_col : edge_col;
        float thick = is_hover_edge ? 2.5f : 1.5f;
        DrawLineEx({x1, y1}, {x2, y2}, thick, ec);
    }

    // Nodes
    for (int i = 0; i < NODE_COUNT; i++) {
        float nx = node_x(NODES[i].slot, NODES[i].tier);
        float ny = tier_y(NODES[i].tier);
        bool  is_owned  = (owned_techs >> static_cast<int>(NODES[i].type)) & 1;
        bool  is_avail  = (avail       >> static_cast<int>(NODES[i].type)) & 1;
        bool  is_hov    = (hover_tech  == static_cast<int>(NODES[i].type));

        Color fill = is_hov    ? hover_col
                   : is_owned  ? owned_col
                   : is_avail  ? avail_col
                   : locked_col;
        DrawCircleV({nx, ny}, NODE_R, fill);
        Color outline = is_hov ? WHITE : Color{ 0, 0, 0, 180 };
        DrawCircleLines((int)nx, (int)ny, (int)NODE_R, outline);

        if (NODES[i].type != TechType::Origin) {
            const char* name = tech_def(NODES[i].type).name;
            int tw = MeasureText(name, FONT_SZ);
            Color tc = (is_owned || is_hov) ? WHITE
                     : is_avail ? Color{200,200,200,255}
                     : Color{100,100,100,255};
            DrawText(name, (int)(nx - tw * 0.5f), (int)(ny + NODE_R + 2), FONT_SZ, tc);
        }
    }
}

int main(int argc, char** argv) {
    Logger::debugEnabled = true;
    uint64_t gen_seed = 1;
    int climate[MAX_MAP_TILES] = {};
    auto new_map = [&]() {
        MapGenParams p = MapGen::drylands_defaults();
        p.seed = gen_seed++;
        MapGenResult r = MapGen(p).generate();
        int mtsz = r.state.map_tiles();
        for (int i = 0; i < mtsz; i++) climate[i] = r.climate[i];

        // Debug roster: spawn one of every unit type next to each player's capital so
        // the visualizer shows all 4 unit types side-by-side for transformer tweaking.
        // Comment out this block once you don't need the comparison spawn anymore.
        int msz = r.state.map_size();
        const UnitType TYPES[]  = { UnitType::Warrior, UnitType::Archer,
                                    UnitType::Rider,   UnitType::Defender };
        const int OFF[8][2]     = {
            {-1,-1},{0,-1},{1,-1},
            {-1, 0},       {1, 0},
            {-1, 1},{0, 1},{1, 1}
        };
        for (int player = 0; player < 2; player++) {
            int cap_tile = -1;
            for (int i = 0; i < msz * msz; i++) {
                const Tile& t = r.state.tile_at(i);
                if (!t.has_city()) continue;
                const City& c = r.state.get_city(t.city_id());
                if (c.owner() == player && c.is_capital()) { cap_tile = i; break; }
            }
            if (cap_tile < 0) continue;
            int cx = cap_tile % msz, cy = cap_tile / msz;
            int placed = 0;
            for (int o = 0; o < 8 && placed < (int)(sizeof(TYPES)/sizeof(TYPES[0])); o++) {
                int nx = cx + OFF[o][0], ny = cy + OFF[o][1];
                if (nx < 0 || ny < 0 || nx >= msz || ny >= msz) continue;
                int ntile = to_index(nx, ny, msz);
                const Tile& nt = r.state.tile_at(ntile);
                if (nt.has_unit()) continue;
                TerrainType ter = nt.terrain();
                if (ter == TerrainType::Water || ter == TerrainType::Mountain) continue;
                r.state.spawn_unit(TYPES[placed], player, ntile);
                placed++;
            }
        }
        return r.state;
    };
    GameState initial = new_map();
    TILE_W = iso_grid_width() / initial.map_size();
    TILE_H = (TILE_W * 3) / 5;  // 5:3 ratio matches the grass terrain sprite
    GameState s = initial;

    // Per-unit facing for the live-play visualizer (purely visual; not in GameState).
    // Indexed by unit slot id. false = facing right (PNG default), true = flipped.
    // Reset on every map regen/reset. Slot reuse on unit respawn may briefly inherit
    // the dead unit's facing — that's tolerable for a visual cue.
    bool unit_facing_left[MAX_MAP_TILES] = {};
    auto reset_facing = [&]() {
        for (auto& f : unit_facing_left) f = false;
    };

    // Move animation — at most one active at a time. When a new Move is applied
    // (or a replay step lands on a Move), we decode its path, record start time,
    // and the unit's draw position is interpolated along the path tiles until the
    // animation completes. A new Move cancels the previous animation: the unit
    // snaps to its current tile (which is the destination of the cancelled move).
    // Animation duration scales sub-linearly with path length: each extra hop adds
    // less time than the first so longer paths play faster per step.
    //   1 hop  → 0.15s
    //   2 hops → 0.15 + 0.08 = 0.23s
    //   3 hops → 0.15 + 0.16 = 0.31s
    //
    // The same struct is reused for Attack animations:
    //   - Kill+advance: attacker ends on the target tile, animated as a 1-hop move.
    //   - Lunge: attacker stays put. `lunge=true` triggers a triangle-wave offset
    //     toward `path_tiles[1]` (the target) and back to `path_tiles[0]`.
    constexpr double MOVE_ANIM_FIRST_HOP_SECS = 0.2;
    constexpr double MOVE_ANIM_EXTRA_HOP_SECS = 0.06;
    constexpr double ATTACK_LUNGE_SECS = 0.2;
    constexpr float  ATTACK_LUNGE_FRAC = 0.35f;  // peak offset toward target, fraction of tile-to-tile vector
    constexpr double ATTACK_RANGED_SECS = 0.20;
    struct MoveAnimation {
        bool   active     = false;
        bool   lunge      = false;
        int    unit_id    = -1;
        int    path_tiles[MAX_MOVE_PATH_STEPS + 1] = {};
        int    path_count = 0;
        double start_time = 0.0;
        double duration   = 0.0;
    };
    MoveAnimation move_anim;

    // Ranged-attack projectile animation. Independent of move_anim: the attacker
    // doesn't move; instead a "ball" arcs from from_tile → to_tile, and during the
    // flight we render the target tile from a pre-attack snapshot so the defender
    // keeps full pre-HP (or stays visible if doomed) until the moment of impact.
    struct RangedAttackAnim {
        bool       active     = false;
        int        from_tile  = -1;
        int        to_tile    = -1;
        double     start_time = 0.0;
        double     duration   = 0.0;
        GameState  pre_state;
    };
    RangedAttackAnim ranged_anim;

    auto start_move_anim = [&](const GameState& pre_state, const Action& a) {
        // A new Move cancels any in-flight ranged projectile so the next frame
        // shows the live state at the previous target tile.
        ranged_anim.active = false;
        if (a.type != ActionType::Move || a.path_steps == 0) {
            move_anim.active = false;
            return;
        }
        int uid = pre_state.tile_at(a.from).unit_id();
        if (uid < 0) { move_anim.active = false; return; }
        int msz = pre_state.map_size();
        int count = 0;
        decode_path_bits(a.from, a.path_bits, a.path_steps, msz,
                         move_anim.path_tiles, &count);
        if (count < 2) { move_anim.active = false; return; }
        move_anim.unit_id    = uid;
        move_anim.path_count = count;
        move_anim.start_time = GetTime();
        int hops = count - 1;
        move_anim.duration   = MOVE_ANIM_FIRST_HOP_SECS
                             + MOVE_ANIM_EXTRA_HOP_SECS * (hops - 1);
        move_anim.lunge      = false;
        move_anim.active     = true;
    };

    // Attack animation. Three variants:
    //   - Ranged: kick off the projectile arc + tile snapshot (see RangedAttackAnim).
    //   - Melee kill+advance: attacker ends up on target tile → 1-hop move animation.
    //   - Melee lunge: attacker stays put → triangle-wave offset toward target and back.
    // If the attacker died (slot empty at both tiles post-attack), skip the melee
    // path — no sprite left to animate. Ranged attackers don't die from attacking,
    // so the ranged path doesn't need that check.
    auto start_attack_anim = [&](const GameState& pre_state,
                                 const GameState& post_state, const Action& a) {
        move_anim.active   = false;
        ranged_anim.active = false;
        if (a.type != ActionType::Attack) return;
        int uid = pre_state.tile_at(a.from).unit_id();
        if (uid < 0) return;
        const Unit&    attacker = pre_state.get_unit(uid);
        const UnitDef& udef     = unit_def(attacker.type());
        bool is_ranged = (udef.abilities & ABILITY_RANGED) != 0;
        if (is_ranged) {
            ranged_anim.from_tile  = a.from;
            ranged_anim.to_tile    = a.to;
            ranged_anim.start_time = GetTime();
            ranged_anim.duration   = ATTACK_RANGED_SECS;
            ranged_anim.pre_state  = pre_state;
            ranged_anim.active     = true;
            return;
        }
        bool advanced    = (post_state.tile_at(a.to).unit_id()   == uid);
        bool still_there = (post_state.tile_at(a.from).unit_id() == uid);
        if (!advanced && !still_there) return;
        move_anim.unit_id       = uid;
        move_anim.path_tiles[0] = a.from;
        move_anim.path_tiles[1] = a.to;
        move_anim.path_count    = 2;
        move_anim.start_time    = GetTime();
        if (advanced) {
            move_anim.lunge    = false;
            move_anim.duration = MOVE_ANIM_FIRST_HOP_SECS;
        } else {
            move_anim.lunge    = true;
            move_anim.duration = ATTACK_LUNGE_SECS;
        }
        move_anim.active = true;
    };

    // Action log (replay: all actions; live: accumulated).
    // Parallel `_debug` vector marks entries that came from Logger::print
    // — those render in grey instead of the player colour.
    std::vector<std::string> action_log;
    std::vector<bool>        action_log_debug;
    int last_logger_count = 0;
    // Trails left by Explorer upgrades — each entry is the sequence of tile
    // indices the explorer walked. Visualizer-only state.
    std::vector<std::vector<int>> explorer_trails;
    int log_scroll = 0;

    auto clear_visuals = [&]() {
        explorer_trails.clear();
    };

    // Replay mode: --replay <path>
    bool replay_mode = false;
    std::vector<GameState> replay_states;
    std::vector<Action>    replay_actions;  // one Action per applied step (paths populated for Move)
    std::vector<std::string> replay_log;   // one entry per action
    int replay_step = 0;
    // Tracks the previous frame's replay_step so we can spot single-step advances
    // (which trigger a move animation) vs jumps/rewinds (which cancel any animation).
    int last_replay_step = -1;
    // Helper: legacy replay files don't store path_bits; recompute via BFS so the
    // visualizer's animation code has the same path the engine derives at apply time.
    auto populate_move_path = [&](Action& a, const GameState& pre) {
        if (a.type != ActionType::Move) return;
        int uid = pre.tile_at(a.from).unit_id();
        if (uid < 0) return;
        int8_t mp_at[MAX_MAP_TILES];
        int    parent[MAX_MAP_TILES];
        reachable_tiles(pre, uid, mp_at, parent);
        encode_path_bits(parent, a.from, a.to, pre.map_size(), &a.path_bits, &a.path_steps);
    };
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--replay") {
            replay_mode = true;
            std::ifstream f(argv[i + 1]);
            if (!f) { fprintf(stderr, "Cannot open replay: %s\n", argv[i + 1]); return 1; }
            GameState rs;
            std::string first;
            f >> first;
            if (first == "seed") {
                uint64_t seed; f >> seed;
                MapGenParams p = MapGen::drylands_defaults();
                p.seed = seed;
                rs = MapGen(p).generate().state;
            } else {
                rs = MapGen(MapGen::drylands_defaults()).generate().state;
                // first token was already consumed — push it back as action fields
                int t = std::stoi(first), fr, to, pa;
                f >> fr >> to >> pa;
                Action act0 = {(ActionType)t, fr, to, pa, true};
                populate_move_path(act0, rs);
                replay_log.push_back(format_action_str(act0, rs.current_player(), rs.get_turn(), rs.map_size()));
                replay_actions.push_back(act0);
                replay_states.push_back(rs);
                replay_states.push_back(rs = rs.apply_action(act0));
            }
            replay_states.push_back(rs);
            int t, fr, to, pa;
            while (f >> t >> fr >> to >> pa) {
                Action act = {(ActionType)t, fr, to, pa, true};
                populate_move_path(act, rs);
                replay_log.push_back(format_action_str(act, rs.current_player(), rs.get_turn(), rs.map_size()));
                replay_actions.push_back(act);
                replay_states.push_back(rs = rs.apply_action(act));
            }
            initial = s = replay_states[0];
            TILE_W = iso_grid_width() / s.map_size();
            TILE_H = (TILE_W * 3) / 5;
            break;
        }
    }

    ViewMode  view = ViewMode::Current;

    Colourer colourers[2];

    auto init_colourers = [&]() {
        colourers[0].init({ 80,  130, 255, 80 });  // blue family for P0
        colourers[1].init({ 255,  80,  80, 80 });  // red  family for P1
        for (int i = 0; i < MAP_TILES; i++) {
            const Tile& t = s.tile_at(i);
            if (!t.has_city()) continue;
            int cid   = t.city_id();
            int owner = s.get_city(cid).owner();
            colourers[owner].assign_city(cid);
        }
    };
    init_colourers();

    auto refresh_borders = [&]() { init_colourers(); };

    int selected_tile = -1;

    Action actions[256];
    int    action_count   = 0;
    int    sidebar_scroll = 0;
    float  tech_zoom      = 1.0f;
    s.legal_actions(actions, action_count);

    // HiDPI: render at the display's physical pixel density (Retina). Without this raylib
    // renders at logical resolution and macOS upscales the framebuffer, making everything
    // — text, shapes, sprites — look pixelated on Retina displays.
    SetConfigFlags(FLAG_WINDOW_HIGHDPI);
    InitWindow(W, H, "Polyshark");
    SetTargetFPS(60);

    // Unit sprites, per owner. [owner][UnitType]. Loaded after GL context exists.
    // Paths are relative to the project root (the visualizer's expected launch directory).
    // Missing files are tolerated — affected units fall back to a circle+letter.
    Texture2D unit_tex[2][(int)UnitType::Count] = {};
    // Offset (in source-image pixels) of the figure's alpha-bbox center relative to image
    // center. Positive y = figure sits below image center (transparent padding on top);
    // positive x = figure sits right of center. Used to centre the visible character on
    // the tile rather than the (often padded) image canvas.
    float unit_tex_xoff[2][(int)UnitType::Count] = {};
    float unit_tex_yoff[2][(int)UnitType::Count] = {};
    auto load_sprite = [&](int owner, UnitType ut, const char* path) {
        if (!FileExists(path)) { fprintf(stderr, "missing sprite: %s\n", path); return; }
        Image img = LoadImage(path);
        if (img.data == nullptr) return;

        // Find first/last opaque row AND column to determine the figure's bbox center.
        Color* px = LoadImageColors(img);
        auto row_opaque = [&](int y) {
            for (int x = 0; x < img.width; x++)
                if (px[y * img.width + x].a > 16) return true;
            return false;
        };
        auto col_opaque = [&](int x) {
            for (int y = 0; y < img.height; y++)
                if (px[y * img.width + x].a > 16) return true;
            return false;
        };
        int min_y = -1, max_y = -1;
        for (int y = 0; y < img.height; y++) if (row_opaque(y)) { min_y = y; break; }
        for (int y = img.height - 1; y >= 0; y--) if (row_opaque(y)) { max_y = y; break; }
        int min_x = -1, max_x = -1;
        for (int x = 0; x < img.width; x++) if (col_opaque(x)) { min_x = x; break; }
        for (int x = img.width - 1; x >= 0; x--) if (col_opaque(x)) { max_x = x; break; }
        UnloadImageColors(px);
        if (min_y >= 0 && max_y >= min_y) {
            float fig_cy = (min_y + max_y) * 0.5f;
            unit_tex_yoff[owner][(int)ut] = fig_cy - img.height * 0.5f;
        }
        if (min_x >= 0 && max_x >= min_x) {
            float fig_cx = (min_x + max_x) * 0.5f;
            unit_tex_xoff[owner][(int)ut] = fig_cx - img.width * 0.5f;
        }

        Texture2D t = LoadTextureFromImage(img);
        UnloadImage(img);
        if (t.id != 0) {
            // Mipmaps + anisotropic = clean downscaling from ~300px source to ~50px tile.
            GenTextureMipmaps(&t);
            SetTextureFilter(t, TEXTURE_FILTER_ANISOTROPIC_16X);
            unit_tex[owner][(int)ut] = t;
        }
    };
    load_sprite(0, UnitType::Warrior,  "visualizer/sprites/units/blue/warrior_blue.png");
    load_sprite(0, UnitType::Archer,   "visualizer/sprites/units/blue/archer_blue.png");
    load_sprite(0, UnitType::Rider,    "visualizer/sprites/units/blue/rider_blue.png");
    load_sprite(0, UnitType::Defender, "visualizer/sprites/units/blue/defender_blue.png");
    load_sprite(1, UnitType::Warrior,  "visualizer/sprites/units/red/warrior_red.png");
    load_sprite(1, UnitType::Archer,   "visualizer/sprites/units/red/archer_red.png");
    load_sprite(1, UnitType::Rider,    "visualizer/sprites/units/red/rider_red.png");
    load_sprite(1, UnitType::Defender, "visualizer/sprites/units/red/defender_red.png");

    auto load_terrain = [&](TerrainSprite ts, const char* path) {
        if (!FileExists(path)) { fprintf(stderr, "missing sprite: %s\n", path); return; }
        Image img = LoadImage(path);
        if (img.data == nullptr) return;

        Color* px = LoadImageColors(img);
        int min_x = img.width, max_x = -1, min_y = img.height, max_y = -1;
        for (int y = 0; y < img.height; y++) {
            for (int x = 0; x < img.width; x++) {
                if (px[y * img.width + x].a > 16) {
                    if (x < min_x) min_x = x;
                    if (x > max_x) max_x = x;
                    if (y < min_y) min_y = y;
                    if (y > max_y) max_y = y;
                }
            }
        }
        UnloadImageColors(px);

        Texture2D t = LoadTextureFromImage(img);
        UnloadImage(img);
        if (t.id != 0) {
            GenTextureMipmaps(&t);
            SetTextureFilter(t, TEXTURE_FILTER_ANISOTROPIC_16X);
            terrain_tex [(int)ts]      = t;
            terrain_bbox[(int)ts]      = { min_x, max_x + 1, min_y, max_y + 1 };
        }
    };
    load_terrain(TerrainSprite::Grass,    "visualizer/sprites/terrain/grass.png");
    load_terrain(TerrainSprite::Mountain, "visualizer/sprites/terrain/mountain.png");

    // Resource sprites share the same loader pattern as terrain — load the image,
    // compute its alpha bbox, and stash texture + bbox in the resource arrays.
    auto load_resource = [&](ResourceSprite rs, const char* path) {
        if (!FileExists(path)) { fprintf(stderr, "missing sprite: %s\n", path); return; }
        Image img = LoadImage(path);
        if (img.data == nullptr) return;
        Color* px = LoadImageColors(img);
        int min_x = img.width, max_x = -1, min_y = img.height, max_y = -1;
        for (int y = 0; y < img.height; y++) {
            for (int x = 0; x < img.width; x++) {
                if (px[y * img.width + x].a > 16) {
                    if (x < min_x) min_x = x;
                    if (x > max_x) max_x = x;
                    if (y < min_y) min_y = y;
                    if (y > max_y) max_y = y;
                }
            }
        }
        UnloadImageColors(px);
        Texture2D t = LoadTextureFromImage(img);
        UnloadImage(img);
        if (t.id != 0) {
            GenTextureMipmaps(&t);
            SetTextureFilter(t, TEXTURE_FILTER_ANISOTROPIC_16X);
            resource_tex [(int)rs] = t;
            resource_bbox[(int)rs] = { min_x, max_x + 1, min_y, max_y + 1 };
        }
    };
    load_resource(ResourceSprite::Fruit,  "visualizer/sprites/terrain/imperius/Imperius_fruit.png");
    load_resource(ResourceSprite::Crop,   "visualizer/sprites/terrain/crop.png");
    load_resource(ResourceSprite::Animal, "visualizer/sprites/terrain/imperius/Imperius_game.png");
    load_resource(ResourceSprite::Metal,  "visualizer/sprites/terrain/metal.png");
    load_resource(ResourceSprite::Forest, "visualizer/sprites/terrain/imperius/Imperius_forest.png");

    while (!WindowShouldClose()) {

        if (IsKeyPressed(KEY_TAB)) {
            view = (view == ViewMode::Omni)    ? ViewMode::P0      :
                   (view == ViewMode::P0)      ? ViewMode::P1      :
                   (view == ViewMode::P1)      ? ViewMode::Current : ViewMode::Omni;
        }

        // Build sidebar layout (headers + action rows)
        static SidebarRow layout[512];
        int layout_count = build_sidebar_layout(actions, action_count, layout, 512);

        // Compute hovered action before drawing so map tiles can react to it
        Vector2 mouse   = GetMousePosition();
        bool    clicked = IsMouseButtonPressed(MOUSE_LEFT_BUTTON);

        // Scroll wheel zooms tech tree when cursor is over the bottom strip panel
        if (mouse.y >= CONTENT_H && mouse.x >= W / 6 && mouse.x < W - W / 6) {
            float wheel = GetMouseWheelMove();
            if (wheel != 0.0f) {
                tech_zoom *= (wheel > 0 ? 1.15f : 1.0f / 1.15f);
                if (tech_zoom < 0.4f) tech_zoom = 0.4f;
                if (tech_zoom > 3.0f) tech_zoom = 3.0f;
            }
        }

        // Scroll wheel scrolls the sidebar actions list
        int SB_CONTENT_TOP = SIDEBAR_TOP;
        int SB_CONTENT_BOT = CONTENT_H - 80;  // above regen + reset buttons
        {
            int SB = MAP_OFF + MAP_PX;
            if (mouse.x >= SB && mouse.x < SB + SIDEBAR &&
                mouse.y >= SB_CONTENT_TOP && mouse.y < SB_CONTENT_BOT) {
                float wheel = GetMouseWheelMove();
                if (wheel != 0.0f) sidebar_scroll -= (int)(wheel * 24);
            }
        }
        // Compute total layout height and clamp scroll
        {
            int total_h = (layout_count > 0)
                ? (layout[layout_count-1].y +
                   (layout[layout_count-1].is_header ? HEADER_H : ROW_H) - SB_CONTENT_TOP)
                : 0;
            int visible_h = SB_CONTENT_BOT - SB_CONTENT_TOP;
            int max_scroll = total_h > visible_h ? total_h - visible_h : 0;
            if (sidebar_scroll < 0) sidebar_scroll = 0;
            if (sidebar_scroll > max_scroll) sidebar_scroll = max_scroll;
        }

        int hovered_action = -1;
        for (int i = 0; i < layout_count; i++) {
            if (layout[i].is_header && layout[i].category != ActionCategory::EndTurn) continue;
            int row_y = layout[i].y - sidebar_scroll;
            if (row_y + ROW_H <= SB_CONTENT_TOP || row_y >= SB_CONTENT_BOT) continue;
            int h = layout[i].is_header ? HEADER_H - 2 : ROW_H - 2;
            Rectangle row = { (float)(MAP_OFF + MAP_PX) + 4, (float)row_y, (float)SIDEBAR - 8, (float)h };
            if (CheckCollisionPointRec(mouse, row)) { hovered_action = layout[i].action_idx; break; }
        }

        // Precompute which tile indices the hovered action highlights
        int highlight_tiles[4] = { -1, -1, -1, -1 };
        if (hovered_action >= 0) {
            const Action& ha = actions[hovered_action];
            switch (ha.type) {
                case ActionType::HarvestResource:
                    highlight_tiles[0] = ha.to;  // resource tile
                    highlight_tiles[1] = s.get_city(ha.from).tile_index();  // city tile (ha.from = city_id)
                    break;
                case ActionType::DebugAddPop:
                case ActionType::UpgradeCity:
                    highlight_tiles[0] = s.get_city(ha.from).tile_index();
                    break;
                case ActionType::TrainUnit:
                    highlight_tiles[0] = ha.from;  // city tile
                    break;
                case ActionType::Move:
                case ActionType::Attack:
                    highlight_tiles[0] = ha.from;
                    highlight_tiles[1] = ha.to;
                    break;
                default: break;
            }
        }
        const int cur_msz = s.map_size();

        // Scrubber — click or drag to jump to any replay step
        static bool scrubbing = false;
        constexpr int SCRUB_H = 18;
        const int scrub_y = CONTENT_H - SCRUB_H;
        if (replay_mode) {
            bool over_scrub = mouse.y >= scrub_y;
            if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && over_scrub) scrubbing = true;
            if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON))              scrubbing = false;
            if (scrubbing) {
                float t = (mouse.x) / (float)W;
                t = t < 0 ? 0 : t > 1 ? 1 : t;
                int new_step = (int)(t * ((int)replay_states.size() - 1));
                if (new_step != replay_step) {
                    clear_visuals();
                    replay_step = new_step;
                    s = replay_states[replay_step];
                    refresh_borders();
                    s.legal_actions(actions, action_count);
                }
            }
        }

        // Collect Move/Attack destinations from the currently selected unit (if any).
        int move_dest_tile[256],  move_dest_action[256];
        int attack_dest_tile[64], attack_dest_action[64];
        int move_dest_count = 0, attack_dest_count = 0;
        int click_applied = -1;
        if (selected_tile >= 0) {
            const Tile& st = s.tile_at(selected_tile);
            if (st.has_unit() && s.get_unit(st.unit_id()).owner() == s.current_player()) {
                for (int i = 0; i < action_count; i++) {
                    const Action& a = actions[i];
                    if (!a.affordable || a.from != selected_tile) continue;
                    if (a.type == ActionType::Move && move_dest_count < 256) {
                        move_dest_tile[move_dest_count]   = a.to;
                        move_dest_action[move_dest_count] = i;
                        move_dest_count++;
                    } else if (a.type == ActionType::Attack && attack_dest_count < 64) {
                        attack_dest_tile[attack_dest_count]   = a.to;
                        attack_dest_action[attack_dest_count] = i;
                        attack_dest_count++;
                    }
                }
            }
        }

        // Map tile click: a click on a highlighted destination applies that Move/Attack;
        // otherwise toggle selection (same-tile click deselects).
        {
            int hover_r = -1, hover_c = -1;
            tile_from_screen(mouse.x, mouse.y, cur_msz, hover_r, hover_c);
            if (clicked && !scrubbing && hover_r >= 0 && hover_c >= 0) {
                int tidx = to_index(hover_c, hover_r, cur_msz);  // to_index = (col, row)

                int matched = -1;
                bool was_move = false;
                for (int i = 0; i < move_dest_count; i++)
                    if (move_dest_tile[i] == tidx) { matched = move_dest_action[i]; was_move = true; break; }
                if (matched < 0)
                    for (int i = 0; i < attack_dest_count; i++)
                        if (attack_dest_tile[i] == tidx) { matched = attack_dest_action[i]; break; }

                if (matched >= 0) {
                    click_applied = matched;
                    // Keep selection on the unit so the next click sees fresh
                    // Move/Attack destinations from its new tile.
                    if (was_move) selected_tile = tidx;
                } else {
                    selected_tile = (selected_tile == tidx) ? -1 : tidx;
                }
            }
        }

        auto is_highlighted = [&](int idx) {
            for (int h : highlight_tiles) if (h == idx) return true;
            return false;
        };
        auto is_move_dest = [&](int idx) {
            for (int i = 0; i < move_dest_count; i++) if (move_dest_tile[i] == idx) return true;
            return false;
        };
        auto is_attack_dest = [&](int idx) {
            for (int i = 0; i < attack_dest_count; i++) if (attack_dest_tile[i] == idx) return true;
            return false;
        };

        // Replay step transitions: a single forward step animates the move; any other
        // jump (scrub, rewind, multi-step) cancels in-flight animation and snaps.
        if (replay_mode) {
            if (last_replay_step >= 0 && replay_step == last_replay_step + 1
                && (replay_step - 1) >= 0
                && (replay_step - 1) < (int)replay_actions.size())
            {
                const Action& act = replay_actions[replay_step - 1];
                if (act.type == ActionType::Move
                    && (replay_step - 1) < (int)replay_states.size())
                {
                    start_move_anim(replay_states[replay_step - 1], act);
                } else if (act.type == ActionType::Attack
                    && (replay_step - 1) < (int)replay_states.size()
                    && replay_step < (int)replay_states.size())
                {
                    start_attack_anim(replay_states[replay_step - 1],
                                      replay_states[replay_step], act);
                } else {
                    move_anim.active = false;
                }
            } else if (last_replay_step != replay_step) {
                move_anim.active   = false;
                ranged_anim.active = false;
            }
            last_replay_step = replay_step;
        }

        BeginDrawing();
        ClearBackground(BLACK);

        // --- Map (isometric diamond grid) ---
        // Iteration phases:
        //   1. Ground diamonds (any order — they don't overlap).
        //   2. Per-tile decorations + units, drawn back-to-front in DESCENDING (r+c) so
        //      bottom-of-screen units' sprites overlap the tiles behind them correctly.
        // Most of the per-tile UI (terrain icons, cities, pop bars, borders, climate tint,
        // axis labels, explorer trails) is intentionally stripped while we verify the iso
        // layout. It will be reintroduced once terrain sprites land.
        int vp = (view == ViewMode::P1)      ? 1 :
                 (view == ViewMode::Current) ? s.current_player() : 0;

        // Single back-to-front pass (high r+c first, low r+c last). For each tile
        // we draw terrain → rings → overlays → unit/HP → hover all in one go, so
        // anything on a NEAR tile naturally overdraws anything from a FAR tile —
        // mountains end up covering forests/animals/crops/units sitting behind them.
        const Color FOG_COL   = {  18,  18,  18, 255 };
        const Color WATER_COL = {  30,  80, 130, 255 };
        for (int sum = 2 * (cur_msz - 1); sum >= 0; sum--) {
            for (int r = 0; r < cur_msz; r++) {
                int c = sum - r;
                if (c < 0 || c >= cur_msz) continue;
                int idx = to_index(c, r, cur_msz);
                // While a ranged-attack projectile is in flight, render the target
                // tile from the pre-attack snapshot so the defender stays visible
                // (and at full pre-HP) until the moment of impact. All other tiles
                // render from the live state as usual.
                const GameState& render_state =
                    (ranged_anim.active && idx == ranged_anim.to_tile)
                        ? ranged_anim.pre_state : s;
                const Tile& t = render_state.tile_at(idx);
                bool fogged = (view != ViewMode::Omni) && !s.is_explored(vp, idx);
                Vector2 ctr = tile_center(r, c, cur_msz);

                // Terrain (or fog/water) first inside the tile.
                if (fogged) {
                    draw_diamond(ctr, FOG_COL);
                    continue;  // nothing else renders for fogged tiles
                }
                TerrainType ter = t.terrain();
                if (ter == TerrainType::Water) {
                    draw_diamond(ctr, WATER_COL);
                } else if (ter == TerrainType::Mountain) {
                    draw_terrain_sprite(ctr, TerrainSprite::Mountain);
                } else {
                    draw_terrain_sprite(ctr, TerrainSprite::Grass);
                }

                // Move-destination: cyan movement_ring (inner solid + outer donut).
                // Attack-destination: same composite ring in red.
                if (is_move_dest(idx)) {
                    // Same faint cyan as the ready ring under units that haven't moved.
                    draw_movement_ring(ctr, { 80, 220, 255, 130 });
                } else if (is_attack_dest(idx)) {
                    draw_movement_ring(ctr, { 230, 80, 80, 130 });
                }

                // Tile decorations (forest overlay + resource overlay + captured-
                // city flag) sit between the terrain pass and the unit so a unit
                // standing on the tile correctly occludes them. Hidden under fog.
                // Forest first, then resource (so e.g. Animal lands on top of Forest).
                if (!fogged) {
                    if (t.terrain() == TerrainType::Forest)
                        draw_resource_sprite(ctr, ResourceSprite::Forest);
                    ResourceSprite rs = resource_type_to_sprite(t.resource());
                    if (rs != ResourceSprite::Count)
                        draw_resource_sprite(ctr, rs);
                    if (t.has_city()) {
                        int owner = render_state.get_city(t.city_id()).owner();
                        draw_city_marker(ctr, owner);
                    }
                }

                if (!fogged && t.has_unit()) {
                    const Unit& u = render_state.get_unit(t.unit_id());
                    bool show = (view == ViewMode::Omni)
                             || s.is_explored(vp, idx)
                             || u.owner() == vp;
                    if (show) {
                        float cxf = ctr.x;
                        float cyf = ctr.y;
                        // Animation override: if this unit is currently animating, lerp
                        // its draw position along the recorded path tiles instead of
                        // rendering at its post-apply destination tile.
                        if (move_anim.active && move_anim.unit_id == t.unit_id()) {
                            double elapsed = GetTime() - move_anim.start_time;
                            if (elapsed >= move_anim.duration) {
                                move_anim.active = false;
                            } else if (move_anim.duration > 0.0) {
                                double p = elapsed / move_anim.duration;
                                int ft = move_anim.path_tiles[0];
                                int tt = move_anim.path_tiles[move_anim.lunge ? 1
                                                                              : move_anim.path_count - 1];
                                if (move_anim.lunge) {
                                    // Triangle wave: 0 → 1 → 0 across the animation,
                                    // scaled by ATTACK_LUNGE_FRAC so the attacker stops
                                    // short of the target tile and snaps back.
                                    double tri = 1.0 - fabs(2.0 * p - 1.0);
                                    double offset = tri * (double)ATTACK_LUNGE_FRAC;
                                    Vector2 fp = tile_center(ft / cur_msz, ft % cur_msz, cur_msz);
                                    Vector2 tp = tile_center(tt / cur_msz, tt % cur_msz, cur_msz);
                                    cxf = (float)(fp.x + (tp.x - fp.x) * offset);
                                    cyf = (float)(fp.y + (tp.y - fp.y) * offset);
                                } else {
                                    // Smoothstep ease-in-out (3p² - 2p³) applied across the
                                    // whole path so the unit accelerates from rest, glides
                                    // through waypoints, and decelerates into the destination.
                                    double eased = p * p * (3.0 - 2.0 * p);
                                    int n_hops = move_anim.path_count - 1;
                                    double seg_f = eased * n_hops;
                                    int seg = (int)seg_f;
                                    if (seg >= n_hops) seg = n_hops - 1;
                                    double t_seg = seg_f - seg;
                                    int hft = move_anim.path_tiles[seg];
                                    int htt = move_anim.path_tiles[seg + 1];
                                    Vector2 fp = tile_center(hft / cur_msz, hft % cur_msz, cur_msz);
                                    Vector2 tp = tile_center(htt / cur_msz, htt % cur_msz, cur_msz);
                                    cxf = (float)(fp.x + (tp.x - fp.x) * t_seg);
                                    cyf = (float)(fp.y + (tp.y - fp.y) * t_seg);
                                }
                            }
                        }
                        Color uc = (u.owner() == 0) ? BLUE : RED;
                        bool can_act = false;
                        if (u.owner() == s.current_player()) {
                            if (s.phase() != GameStateType::Idle) {
                                can_act = (u.move_points() > 0 || !u.has_attacked());
                            } else {
                                for (int ai = 0; ai < action_count; ai++) {
                                    const Action& aa = actions[ai];
                                    if ((aa.type == ActionType::Move        && aa.from == idx)
                                     || (aa.type == ActionType::Attack      && aa.from == idx)
                                     || (aa.type == ActionType::CaptureCity && aa.to   == idx)) {
                                        can_act = true; break;
                                    }
                                }
                            }
                        }
                        // "Ready" ring under any current-player unit that still has a legal
                        // Move or Attack from this tile. Scanning the legal-actions list
                        // catches mid-turn cases the "haven't moved yet" check missed —
                        // e.g. a freshly-revealed enemy entering range, or a unit that
                        // moved into a position where it can now attack.
                        bool has_legal_act = false;
                        if (u.owner() == s.current_player()) {
                            for (int ai = 0; ai < action_count; ai++) {
                                const Action& aa = actions[ai];
                                if (aa.from == idx
                                    && (aa.type == ActionType::Move || aa.type == ActionType::Attack)) {
                                    has_legal_act = true; break;
                                }
                            }
                        }
                        if (has_legal_act) {
                            const RingTransformer& rt = READY_RING_TRANSFORMER;
                            float rx = TILE_W * 0.35f * rt.scale;
                            float ry = TILE_H * 0.35f * rt.scale;
                            int rcx = (int)(cxf + rt.x_offset * (float)TILE_W);
                            int rcy = (int)(cyf + rt.y_offset * (float)TILE_H);
                            bool selected = (selected_tile == idx);
                            Color fill = selected ? Color{ 255, 220,  60,  90 }
                                                  : Color{  80, 220, 255,  90 };
                            Color edge = selected ? Color{ 255, 220,  60, 230 }
                                                  : Color{  80, 220, 255, 230 };
                            DrawEllipse(rcx, rcy, rx, ry, color_alpha_mul(fill, rt.opacity));
                            DrawEllipseLines(rcx, rcy, rx, ry, color_alpha_mul(edge, rt.opacity));
                        }

                        // Sprite scales off TILE_W (wider of the two) — keeps figures from
                        // looking squished on the shorter iso tile.
                        const Texture2D& tex = unit_tex[u.owner()][(int)u.type()];
                        if (tex.id != 0) {
                            // Base scale fits the tile width, then the per-unit tweak multiplier.
                            const Transformer& xf = UNIT_TRANSFORMERS[(int)u.type()];
                            float scale = (TILE_W * 0.95f) / (float)tex.height * xf.scale;
                            float w = tex.width  * scale;
                            float h = tex.height * scale;
                            // Auto-centering offsets (from alpha bbox) keep the figure on the tile.
                            float xoff = unit_tex_xoff[u.owner()][(int)u.type()] * scale;
                            float yoff = unit_tex_yoff[u.owner()][(int)u.type()] * scale;
                            // Per-unit manual tweaks, in tile-fractions.
                            float xtweak = xf.x_offset * (float)TILE_W;
                            float ytweak = xf.y_offset * (float)TILE_H;
                            // Facing: a negative source width tells raylib to mirror the texture.
                            // We also invert the x-axis offsets so the figure still centres on the
                            // tile after the mirror.
                            bool flipped = unit_facing_left[t.unit_id()];
                            float src_w = flipped ? -(float)tex.width : (float)tex.width;
                            float xoff_eff   = flipped ? -xoff   : xoff;
                            float xtweak_eff = flipped ? -xtweak : xtweak;
                            DrawTexturePro(tex,
                                { 0, 0, src_w, (float)tex.height },
                                { cxf - w * 0.5f - xoff_eff + xtweak_eff,
                                  cyf - h * 0.5f - yoff + ytweak, w, h },
                                { 0, 0 }, 0.0f, WHITE);
                        } else {
                            DrawCircle((int)cxf, (int)cyf, TILE_W / 7, uc);
                            static const char* unit_icon[] = { "?", "W", "A", "R", "D" };
                            const char* icon = unit_icon[(int)u.type()];
                            DrawText(icon, (int)cxf - MeasureText(icon, 9) / 2, (int)cyf - 4, 9, WHITE);
                        }
                        // HP number above the diamond's top point so it sits clear of the sprite.
                        // Placement governed by HP_TRANSFORMER (offsets in tile-fractions, scale
                        // multiplies the base 11pt font).
                        const char* hp_str = TextFormat("%d", u.hp());
                        int hp_fs = (int)(11.0f * HP_TRANSFORMER.scale + 0.5f);
                        if (hp_fs < 1) hp_fs = 1;
                        int hp_w = MeasureText(hp_str, hp_fs);
                        int hp_x = (int)cxf - hp_w / 2
                                 + (int)(HP_TRANSFORMER.x_offset * TILE_W);
                        int hp_y = (int)(cyf - TILE_H * 0.5f) - 12
                                 + (int)(HP_TRANSFORMER.y_offset * TILE_H);
                        DrawText(hp_str, hp_x + 1, hp_y + 1, hp_fs, BLACK);
                        DrawText(hp_str, hp_x,     hp_y,     hp_fs, WHITE);
                    }
                }

                // Hover highlight on top of everything else.
                if (is_highlighted(idx)) {
                    draw_diamond(ctr, { 255, 255, 255, 40 });
                    draw_diamond_outline(ctr, { 255, 255, 80, 220 }, 3.0f);
                }
            }
        }

        // Tile-coordinate axis labels along the two bottom edges of the diamond grid.
        // Bottom-right edge: tiles with r=0, labeled with their column index 0..N-1.
        // Bottom-left edge:  tiles with c=0, labeled with their row    index 0..N-1.
        // Tile (0,0) sits at the front apex and gets one label per axis (different sides).
        {
            constexpr int AXIS_FONT_SZ = 11;
            const Color AXIS_TEXT   = { 150, 150, 150, 255 };
            const Color AXIS_SHADOW = {  20,  20,  20, 160 };
            for (int c = 0; c < cur_msz; c++) {
                Vector2 ctr = tile_center(0, c, cur_msz);
                char buf[8]; snprintf(buf, sizeof(buf), "%d", c);
                int x = (int)(ctr.x + TILE_W * 0.32f);
                int y = (int)(ctr.y + TILE_H * 0.35f);
                DrawText(buf, x + 1, y + 1, AXIS_FONT_SZ, AXIS_SHADOW);
                DrawText(buf, x,     y,     AXIS_FONT_SZ, AXIS_TEXT);
            }
            for (int r = 0; r < cur_msz; r++) {
                Vector2 ctr = tile_center(r, 0, cur_msz);
                char buf[8]; snprintf(buf, sizeof(buf), "%d", r);
                int tw = MeasureText(buf, AXIS_FONT_SZ);
                int x = (int)(ctr.x - TILE_W * 0.32f) - tw;
                int y = (int)(ctr.y + TILE_H * 0.35f);
                DrawText(buf, x + 1, y + 1, AXIS_FONT_SZ, AXIS_SHADOW);
                DrawText(buf, x,     y,     AXIS_FONT_SZ, AXIS_TEXT);
            }
        }

        // Ranged-attack projectile: a small ball arcs from from_tile → to_tile. The
        // tile-snapshot above keeps the defender at pre-HP during this flight; the
        // moment the projectile lands, we deactivate the anim and the next frame
        // renders the live (post-damage) state.
        if (ranged_anim.active) {
            double elapsed = GetTime() - ranged_anim.start_time;
            if (elapsed >= ranged_anim.duration) {
                ranged_anim.active = false;
            } else if (ranged_anim.duration > 0.0) {
                double p = elapsed / ranged_anim.duration;
                int    ft = ranged_anim.from_tile;
                int    tt = ranged_anim.to_tile;
                Vector2 fp = tile_center(ft / cur_msz, ft % cur_msz, cur_msz);
                Vector2 tp = tile_center(tt / cur_msz, tt % cur_msz, cur_msz);
                float lx = (float)(fp.x + (tp.x - fp.x) * p);
                float ly = (float)(fp.y + (tp.y - fp.y) * p);
                // Parabolic arc — peak at p=0.5. Height scales with travel distance
                // so adjacent shots get a modest hop while longer shots arc higher.
                float dx = tp.x - fp.x, dy = tp.y - fp.y;
                float dist = sqrtf(dx * dx + dy * dy);
                float arc_h = dist * 0.18f;
                float ay = (float)(ly - arc_h * 4.0 * p * (1.0 - p));
                float r  = TILE_W * 0.045f;
                if (r < 2.0f) r = 2.0f;
                DrawCircle((int)lx, (int)ay, r, BLACK);
            }
        }

        // --- Tech tree panel (bottom strip, centred — spans 1/6 to 5/6 of width) ---
        // Equal empty margins on left and right (each 1/6 of W), giving the panel 4/6 of W.
        const int TECH_X       = W / 6;
        const int TECH_X_END   = W - W / 6;
        const int TECH_PANEL_W = TECH_X_END - TECH_X;
        DrawRectangle(TECH_X, CONTENT_H, TECH_PANEL_W, TECH_H, PANEL_BG);
        DrawLine(TECH_X,     CONTENT_H, TECH_X_END, CONTENT_H, PANEL_LINE);
        DrawLine(TECH_X,     CONTENT_H, TECH_X,     H,         PANEL_LINE);
        DrawLine(TECH_X_END, CONTENT_H, TECH_X_END, H,         PANEL_LINE);
        {
            int hover_tech = -1;
            if (hovered_action >= 0 && actions[hovered_action].type == ActionType::ResearchTech)
                hover_tech = actions[hovered_action].param;
            if (view == ViewMode::Omni) {
                int mid_x = TECH_X + TECH_PANEL_W / 2;
                DrawLine(mid_x, CONTENT_H + 4, mid_x, H - 4, { 50, 50, 50, 255 });
                int cur = s.current_player();
                draw_tech_tree(s.get_player(0).techs_mask(), 0,
                               TECH_X, CONTENT_H, mid_x - TECH_X,    TECH_H, "P0",
                               cur == 0 ? hover_tech : -1, tech_zoom);
                draw_tech_tree(s.get_player(1).techs_mask(), 1,
                               mid_x,  CONTENT_H, TECH_X_END - mid_x, TECH_H, "P1",
                               cur == 1 ? hover_tech : -1, tech_zoom);
            } else {
                int pv = (view == ViewMode::P1) ? 1 : (view == ViewMode::Current) ? s.current_player() : 0;
                draw_tech_tree(s.get_player(pv).techs_mask(), pv,
                               TECH_X, CONTENT_H, TECH_PANEL_W, TECH_H, nullptr,
                               hover_tech, tech_zoom);
            }
        }

        // --- Top HUD bar ---
        DrawRectangle(0, 0, W, TOP_HUD, PANEL_BG);
        DrawLine(0, TOP_HUD, W, TOP_HUD, PANEL_LINE);

        // Turn + active player (left side)
        DrawText(TextFormat("Turn %d", s.get_turn()),       PAD + 4, 8,  22, WHITE);
        DrawText(TextFormat("P%d to move", s.current_player()), PAD + 4, 34, 18, LIGHTGRAY);

        // Stars — large, player-coloured (centre). Suffix shows income per turn.
        int spt[2] = {0, 0};
        for (int i = 0; i < s.map_tiles(); i++) {
            const Tile& tt = s.tile_at(i);
            if (!tt.has_city()) continue;
            const City& cc = s.get_city(tt.city_id());
            int o = cc.owner();
            if (o == 0 || o == 1) spt[o] += cc.stars_per_turn();
        }
        const char* p0_str = TextFormat("P0  %d *", s.get_stars(0));
        const char* p1_str = TextFormat("P1  %d *", s.get_stars(1));
        const char* p0_spt = TextFormat("+%d", spt[0]);
        const char* p1_spt = TextFormat("+%d", spt[1]);
        int p0_x = W / 2 - 110, p1_x = W / 2 + 10;
        DrawText(p0_str, p0_x, 6, 28, { 100, 160, 255, 255 });
        DrawText(p1_str, p1_x, 6, 28, { 255, 100, 100, 255 });
        DrawText(p0_spt, p0_x + (MeasureText(p0_str, 28) - MeasureText(p0_spt, 14)) / 2, 36, 14, { 100, 160, 255, 200 });
        DrawText(p1_spt, p1_x + (MeasureText(p1_str, 28) - MeasureText(p1_spt, 14)) / 2, 36, 14, { 255, 100, 100, 200 });

        // View mode badge (right side)
        {
            const char* vname = (view == ViewMode::Omni)    ? "OMNISCIENT" :
                                (view == ViewMode::P0)      ? "FOG: P0"    :
                                (view == ViewMode::P1)      ? "FOG: P1"    : "FOG: CURRENT";
            Color badge_col = (view == ViewMode::P0)      ? COL_P0 :
                              (view == ViewMode::P1)      ? COL_P1 :
                              (view == ViewMode::Current) ? ((s.current_player() == 0) ? COL_P0 : COL_P1)
                                                          : Color{ 90, 90, 90, 255 };
            int font_sz = 18;
            int tw = MeasureText(vname, font_sz);
            int right_edge = MAP_OFF + MAP_PX + SIDEBAR;  // left edge of log panel
            int bx = right_edge - tw - 24, by = 10, bw = tw + 16, bh = 28;
            DrawRectangleRounded({ (float)bx, (float)by, (float)bw, (float)bh }, 0.4f, 6, badge_col);
            DrawText(vname, bx + 8, by + 5, font_sz, WHITE);
            DrawText("[Tab]", right_edge - MeasureText("[Tab]", 13) - 8, by + bh + 4, 13, GRAY);

            if (replay_mode) {
                const char* rtxt = TextFormat("REPLAY  %d / %d", replay_step, (int)replay_states.size() - 1);
                int rtw = MeasureText(rtxt, 16);
                DrawRectangleRounded({ (float)(bx - rtw - 28), (float)by, (float)(rtw + 16), (float)bh }, 0.4f, 6, Color{ 40, 120, 80, 255 });
                DrawText(rtxt, bx - rtw - 20, by + 5, 16, WHITE);
                DrawText("[← →] step   [R] reset", bx - rtw - 20, by + bh + 4, 11, GRAY);
            }
        }

        // --- Sidebar ---
        int SB = MAP_OFF + MAP_PX;  // sidebar left edge
        DrawRectangle(SB, TOP_HUD, SIDEBAR, CONTENT_H - TOP_HUD, SIDEBAR_BG);
        DrawLine(SB, TOP_HUD, SB, CONTENT_H, PANEL_LINE);
        DrawText("LEGAL ACTIONS", SB + 8, TOP_HUD + 10, 16, GRAY);
        DrawLine(SB + 4, TOP_HUD + 30, SB + SIDEBAR - 4, TOP_HUD + 30, { 60, 60, 60, 255 });

        int  applied    = click_applied;  // map-click Move/Attack takes precedence; sidebar can also set this
        bool reset      = false;
        bool regenerate = false;

        // Spacebar = End Turn (only in live play; in replay mode Space steps forward)
        if (!replay_mode && IsKeyPressed(KEY_SPACE)) {
            for (int i = 0; i < action_count; i++) {
                if (actions[i].type == ActionType::EndTurn && actions[i].affordable) {
                    applied = i;
                    break;
                }
            }
        }

        // Replay step controls
        if (replay_mode) {
            auto jump_to = [&](int step) {
                step = step < 0 ? 0 : step > (int)replay_states.size() - 1 ? (int)replay_states.size() - 1 : step;
                if (step != replay_step) {
                    clear_visuals();
                    replay_step = step;
                    s = replay_states[replay_step];
                    refresh_borders();
                    s.legal_actions(actions, action_count);
                }
            };

            static float key_timer = 0.0f;
            static float key_next  = 0.0f;
            static int   key_held  = 0; // -1 left, 0 none, 1 right
            constexpr float HOLD_DELAY  = 0.3f;
            constexpr float HOLD_REPEAT = 0.07f;

            if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_SPACE)) { jump_to(replay_step + 1); key_held = 1;  key_timer = 0.0f; key_next = HOLD_DELAY; }
            else if (IsKeyPressed(KEY_LEFT))                         { jump_to(replay_step - 1); key_held = -1; key_timer = 0.0f; key_next = HOLD_DELAY; }
            else if (key_held != 0) {
                bool still = (key_held == 1 && (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_SPACE)))
                           || (key_held == -1 && IsKeyDown(KEY_LEFT));
                if (still) {
                    key_timer += GetFrameTime();
                    while (key_timer >= key_next) { jump_to(replay_step + key_held); key_next += HOLD_REPEAT; }
                } else { key_held = 0; }
            }
        }

        BeginScissorMode(SB, SB_CONTENT_TOP, SIDEBAR, SB_CONTENT_BOT - SB_CONTENT_TOP);
        for (int i = 0; i < layout_count; i++) {
            const SidebarRow& lr = layout[i];
            int ry = lr.y - sidebar_scroll;

            // Skip rows fully outside the visible area
            int row_h = lr.is_header ? HEADER_H : ROW_H;
            if (ry + row_h <= SB_CONTENT_TOP || ry >= SB_CONTENT_BOT) continue;

            if (lr.is_header && lr.category == ActionCategory::EndTurn) {
                Rectangle row = { (float)SB + 4, (float)ry, (float)SIDEBAR - 8, (float)HEADER_H - 2 };
                bool hov = CheckCollisionPointRec(mouse, row);
                DrawRectangleRec(row, hov ? Color{ 65, 110, 65, 255 } : Color{ 35, 60, 35, 255 });
                DrawText("End Turn", SB + 8, ry + 2, 12, hov ? WHITE : Color{ 160, 200, 160, 255 });
                if (hov && clicked) applied = lr.action_idx;
            } else if (lr.is_header) {
                DrawRectangle(SB + 4, ry, SIDEBAR - 8, HEADER_H - 2, { 25, 25, 25, 255 });
                DrawText(category_name(lr.category), SB + 8, ry + 2, 12, { 160, 160, 160, 255 });
            } else {
                const Action& a   = actions[lr.action_idx];
                bool hovered      = (hovered_action == lr.action_idx);
                bool afford       = a.affordable;
                Rectangle row     = { (float)SB + 4, (float)ry, (float)SIDEBAR - 8, (float)ROW_H - 2 };

                Color bg = afford ? (hovered ? Color{ 65, 90, 140, 255 } : Color{ 50, 50, 50, 255 })
                                  : (hovered ? Color{ 60, 55, 55, 255  } : Color{ 38, 35, 35, 255 });
                DrawRectangleRec(row, bg);

                char label[64]; action_label(a, label, sizeof(label));
                char cost[16];  action_cost(a, cost, sizeof(cost), s.owned_cities(s.current_player()));
                Color text_col = afford ? WHITE : Color{ 100, 100, 100, 255 };
                Color cost_col = afford ? Color{ 255, 210, 60, 255 } : Color{ 100, 90, 50, 255 };
                DrawText(label, SB + 10, ry + 5, 13, text_col);
                if (cost[0]) {
                    int cw = MeasureText(cost, 12);
                    DrawText(cost, SB + SIDEBAR - 10 - cw, ry + 6, 12, cost_col);
                }

                if (hovered && clicked && afford)
                    applied = lr.action_idx;
            }
        }
        EndScissorMode();

        // --- Regenerate button ---
        if (!replay_mode) {
            Rectangle regen_btn = { (float)SB + 4, (float)CONTENT_H - 72, (float)SIDEBAR - 8, 28 };
            bool hovered = CheckCollisionPointRec(mouse, regen_btn);
            DrawRectangleRec(regen_btn, hovered ? Color{ 30, 130, 80, 255 } : Color{ 20, 80, 50, 255 });
            int tw = MeasureText("REGEN MAP", 14);
            DrawText("REGEN MAP", SB + SIDEBAR / 2 - tw / 2, CONTENT_H - 64, 14, WHITE);
            if (hovered && clicked)
                regenerate = true;
        }

        // --- Reset button (bottom of sidebar) ---
        {
            Rectangle reset_btn = { (float)SB + 4, (float)CONTENT_H - 36, (float)SIDEBAR - 8, 28 };
            bool hovered = CheckCollisionPointRec(mouse, reset_btn);
            DrawRectangleRec(reset_btn, hovered ? Color{ 140, 50, 50, 255 } : Color{ 90, 30, 30, 255 });
            DrawText("RESET", SB + SIDEBAR / 2 - 24, CONTENT_H - 30, 18, WHITE);
            if (hovered && clicked)
                reset = true;
        }

        // Apply after the loop so we don't mutate actions[] mid-render
        if (regenerate) {
            clear_visuals();
            initial = new_map();
            TILE_W = iso_grid_width() / initial.map_size();
            TILE_H = (TILE_W * 3) / 5;  // 5:3 ratio matches the grass terrain sprite
            s = initial;
            reset_facing();
            init_colourers();
            s.legal_actions(actions, action_count);
            sidebar_scroll = 0;
            selected_tile  = -1;
            action_log.clear();
            action_log_debug.clear();
            log_scroll = 0;
        } else if (reset) {
            clear_visuals();
            if (replay_mode) {
                replay_step = 0;
                s = replay_states[0];
            } else {
                s = initial;
                action_log.clear();
                action_log_debug.clear();
            }
            reset_facing();
            init_colourers();
            s.legal_actions(actions, action_count);
            sidebar_scroll = 0;
            selected_tile  = -1;
            log_scroll = 0;
        }
        if (applied >= 0 && !replay_mode) {
            action_log.push_back(format_action_str(actions[applied], s.current_player(), s.get_turn(), s.map_size()));
            action_log_debug.push_back(false);

            // Wipe any leftover visual-only state before the new action.
            clear_visuals();

            // If this is the Explorer upgrade, simulate it on a copy first to capture
            // the path the explorer will walk, so we can draw arrows along it.
            const Action& ap = actions[applied];
            if (ap.type == ActionType::UpgradeCity
                && ap.param == (int)CityUpgradeType::L1_EXPLORER) {
                GameState preview = s;
                int path[16]; int plen = 0;
                preview.explore(preview.get_city(ap.from).tile_index(),
                                s.current_player(), path, &plen);
                if (plen > 1) {
                    std::vector<int> t(path, path + plen);
                    explorer_trails.push_back(std::move(t));
                }
            }

            // Update facing for Move/Attack actions: if the unit moves so its screen-x
            // decreases (i.e. (Δcol - Δrow) < 0), it now faces left.
            if (actions[applied].type == ActionType::Move
                || actions[applied].type == ActionType::Attack) {
                const Action& a  = actions[applied];
                int uid = s.tile_at(a.from).unit_id();
                int sz  = s.map_size();
                int fc  = a.from % sz, fr = a.from / sz;
                int tc  = a.to   % sz, tr = a.to   / sz;
                int delta = (tc - tr) - (fc - fr);
                if (uid >= 0 && delta != 0)
                    unit_facing_left[uid] = (delta < 0);
            }

            // Kick off the move/attack animation. For Move we only need pre-state; for
            // Attack we need both pre and post to decide between lunge and kill-advance.
            GameState pre_state = s;
            s = s.apply_action(actions[applied]);
            if (actions[applied].type == ActionType::Move) {
                start_move_anim(pre_state, actions[applied]);
            } else if (actions[applied].type == ActionType::Attack) {
                start_attack_anim(pre_state, s, actions[applied]);
            }
            refresh_borders();
            s.legal_actions(actions, action_count);
            sidebar_scroll = 0;
        }

        // Sync log to current replay step
        if (replay_mode) {
            action_log.assign(replay_log.begin(), replay_log.begin() + std::min(replay_step, (int)replay_log.size()));
            action_log_debug.assign(action_log.size(), false);
        }

        // Pull any new Logger::print entries into the log panel as grey rows.
        while (last_logger_count < Logger::count) {
            int back_idx = Logger::count - 1 - last_logger_count;  // 0 = newest
            const char* msg = Logger::get(back_idx);
            if (msg) {
                action_log.emplace_back(msg);
                action_log_debug.push_back(true);
            }
            last_logger_count++;
        }


        // --- Selected tile outline (info tooltip removed) ---
        if (selected_tile >= 0) {
            int sx, sy;
            to_coords(selected_tile, sx, sy);
            Vector2 sel_ctr = tile_center(sy, sx, cur_msz);
            draw_diamond_outline(sel_ctr, WHITE, 2.0f);
        }

        // --- Action log panel ---
        {
            constexpr int ROW_H   = 17;
            constexpr int LOG_PAD = 6;
            int lx = TECH_W + MAP_PX + SIDEBAR;
            DrawRectangle(lx, 0, LOG_W, CONTENT_H, { 18, 18, 18, 255 });
            DrawLine(lx, 0, lx, CONTENT_H, PANEL_LINE);
            DrawText("ACTION LOG", lx + 8, 8, 14, GRAY);
            DrawLine(lx + 4, 28, lx + LOG_W - 4, 28, { 50, 50, 50, 255 });

            int visible_rows = (CONTENT_H - 28 - LOG_PAD) / ROW_H;
            int total_entries = (int)action_log.size();

            // Auto-scroll to bottom unless user has scrolled up
            int max_scroll = std::max(0, total_entries - visible_rows);
            if (log_scroll > max_scroll) log_scroll = max_scroll;

            // Mouse wheel scroll when over panel
            Rectangle panel_rect = { (float)lx, 28, (float)LOG_W, (float)(CONTENT_H - 28) };
            if (CheckCollisionPointRec(mouse, panel_rect)) {
                log_scroll -= (int)GetMouseWheelMove();
                if (log_scroll < 0) log_scroll = 0;
                if (log_scroll > max_scroll) log_scroll = max_scroll;
            }

            BeginScissorMode(lx, 28 + LOG_PAD, LOG_W, CONTENT_H - 28 - LOG_PAD);
            for (int i = 0; i < visible_rows; i++) {
                int entry_idx = log_scroll + i;
                if (entry_idx >= total_entries) break;
                const std::string& entry = action_log[entry_idx];
                int ey = 28 + LOG_PAD + i * ROW_H;
                if (i % 2 == 0)
                    DrawRectangle(lx + 2, ey, LOG_W - 4, ROW_H - 1, { 25, 25, 25, 255 });
                bool is_debug = (entry_idx < (int)action_log_debug.size())
                              && action_log_debug[entry_idx];
                Color tc = is_debug
                         ? Color{ 160, 160, 160, 255 }
                         : ((entry.size() > 1 && entry[1] == '0') ? COL_P0 : COL_P1);
                DrawText(entry.c_str(), lx + 6, ey + 2, 11, tc);
            }
            EndScissorMode();

            // Scrollbar
            if (total_entries > visible_rows) {
                float bar_h = (float)(CONTENT_H - 28) * visible_rows / total_entries;
                float bar_y = 28 + (float)(CONTENT_H - 28) * log_scroll / total_entries;
                DrawRectangle(lx + LOG_W - 4, (int)bar_y, 3, (int)bar_h, { 90, 90, 90, 255 });
            }
        }

        // Scrubber bar (drawn on top of everything, replay mode only)
        if (replay_mode && !replay_states.empty()) {
            int total = (int)replay_states.size() - 1;
            int fill_w = total > 0 ? (int)((float)replay_step / total * W) : 0;
            int handle_x = fill_w;

            DrawRectangle(0, scrub_y, W, SCRUB_H, { 20, 20, 20, 230 });
            DrawRectangle(0, scrub_y, fill_w, SCRUB_H, { 40, 130, 85, 255 });
            DrawRectangle(handle_x - 2, scrub_y, 4, SCRUB_H, WHITE);

            const char* label = TextFormat("%d / %d", replay_step, total);
            int lw = MeasureText(label, 12);
            DrawText(label, W / 2 - lw / 2, scrub_y + 3, 12, { 200, 200, 200, 255 });
        }

        EndDrawing();
    }

    for (int owner = 0; owner < 2; owner++)
        for (int i = 0; i < (int)UnitType::Count; i++)
            if (unit_tex[owner][i].id != 0) UnloadTexture(unit_tex[owner][i]);
    for (int i = 0; i < (int)TerrainSprite::Count; i++)
        if (terrain_tex[i].id != 0) UnloadTexture(terrain_tex[i]);
    CloseWindow();
    return 0;
}
