#include "raylib.h"
#include "rlgl.h"
#include "mapgen.h"
#include "unit_def.h"
#include "resource_def.h"
#include "building_def.h"
#include "tech_def.h"
#include "game_state.h"
#include "logger.h"
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <utility>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <vector>
#include <string>
#include <unistd.h>
#include <sys/wait.h>

// 5:3 diamond (matches grass sprite 840x507); recomputed each map gen to fill MAP_CANVAS.
static int            TILE_W   = 56;
static int            TILE_H   = 34;

// Global UI font. Loaded after InitWindow; falls back to raylib default if missing.
// Toggle with F at runtime.
static Font g_font           = {};
static bool g_use_custom_font = false;

static inline bool g_font_active() { return g_use_custom_font && g_font.texture.id != 0; }

static inline void DrawTextC(const char* text, int x, int y, int size, Color color) {
    if (g_font_active())
        DrawTextEx(g_font, text, { (float)x, (float)y }, (float)size, 0.0f, color);
    else
        DrawText(text, x, y, size, color);
}

static inline int MeasureTextC(const char* text, int size) {
    if (g_font_active())
        return (int)MeasureTextEx(g_font, text, (float)size, 0.0f).x;
    return MeasureText(text, size);
}
static constexpr int PAD      = 4;
static constexpr int LABEL    = 20;
static constexpr int TOP_HUD  = 64;
static constexpr int SIDEBAR  = 172;
static constexpr int TECH_W   = 210;
static constexpr int MAP_OFF  = TECH_W;
static constexpr int MAP_CANVAS = 674;
static constexpr int MAP_PX     = MAP_CANVAS + PAD * 2 + LABEL;
static constexpr int LOG_W      = 200;
static constexpr int TECH_H     = 18;  // scrubber strip below main content
static constexpr float TECH_PANEL_SIZE = 320.0f;  // square side of the expanded tech tree panel
static constexpr float TECH_ICON_R     = 18.0f;   // radius of the collapsed bottom-left icon
static constexpr int W          = TECH_W + MAP_PX + SIDEBAR + LOG_W;
// CONTENT_H = main content bottom; H = full window incl. bottom tech strip.
static constexpr int CONTENT_H  = TOP_HUD + MAP_CANVAS + PAD * 2 + LABEL;
static constexpr int H          = CONTENT_H + TECH_H;

enum class ViewMode { Omni, P0, P1, Current };

// Reusable colors
static constexpr Color PANEL_BG   = {  22,  22,  22, 255 };
static constexpr Color PANEL_LINE = {  65,  65,  65, 255 };
static constexpr Color SIDEBAR_BG = {  38,  38,  38, 255 };
static constexpr Color COL_P0     = {  80, 140, 220, 255 };
static constexpr Color COL_P1     = { 220,  80,  80, 255 };

// Iso diamond grid. Axes: (0,0)=BOTTOM, (N,N)=TOP, (0,N)=RIGHT, (N,0)=LEFT.
// Tile center: cx = origin_x + (c-r)*TILE_W/2, cy = origin_y - (c+r)*TILE_H/2.
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

// Inverse of tile_center; (-1,-1) if outside. Rounding continuous (r,c) picks the right diamond.
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

// Tile-relative tweak; offsets in tile-fractions (0.10 → 10% of TILE_W), scale multiplies auto-scale.
struct Transformer {
    float x_offset;
    float y_offset;
    float scale;
};

// Per-unit-type sprite tweak.
static const Transformer UNIT_TRANSFORMERS[(int)UnitType::Count] = {
    /* None     */ { 0.00f,  0.00f, 0.75f },
    /* Warrior  */ { -0.03f, -0.04f, 0.70f },
    /* Archer   */ { 0.05f, -0.05f, 0.73f },
    /* Rider    */ { 0.00f, -0.08f, 0.62f },
    /* Defender */ { 0.00f, -0.08f, 0.73f },
    /* Giant    */ { 0.00f, -0.15f, 0.76f },
};

// HP number above the diamond top; scale multiplies 11pt base.
static const Transformer HP_TRANSFORMER = { -0.22f, 0.24f, 0.9f };

// Defense shield(s) shown next to HP when a fortified unit has a defense
// multiplier > 1.0. scale sets the base shield width in pixels; x/y_offset
// nudge the whole shield strip relative to its anchor (right of the HP text).
static const Transformer SHIELD_TRANSFORMER = { -0.11f, 0.02f, 12.0f };
static const Transformer SECOND_SHIELD_TRANSFORMER = { -0.28f, 0.03f, 15.5f };

// Top HUD star icons (both the big stars next to player totals and the small
// stars next to per-player SPT). x/y_offset here are SCREEN PIXELS (unlike the
// tile-fraction Transformers above); scale multiplies the base icon height.
static Transformer TOP_HUD_STAR_TRANSFORMER = { 0.0f, -5.5f, 1.00f };
static Transformer BOTTOM_HUD_STAR_TRANSFORMER = { 0.0f, -2.0f, 1.00f };

// Transformer + opacity multiplier for ring visuals.
struct RingTransformer {
    float x_offset;
    float y_offset;
    float scale;
    float opacity;
};

// "Ready" ring under units that can still act; base radius 35% of tile.
static const RingTransformer READY_RING_TRANSFORMER = { 0.00f, 0.14f, 0.6f, 1.00f };

// Move-dest ring: solid inner ellipse + white donut + outer cyan donut.
// All radii in fractions of TILE_W; squished by TILE_H/TILE_W for iso ground.
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

// Multiplies a colour's alpha by `mul` (clamped to [0,255]).
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

    // Inner ellipse.
    float ir_w = m.inner_solid_r * (float)TILE_W * scale;
    float ir_h = ir_w * aspect;
    DrawEllipse((int)cx, (int)cy, ir_w, ir_h, cyan_col);

    // Donuts: squish matrix vertically since DrawRing is circular.
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
// Anchored to diamond bottom; mountains extend above and rely on back-to-front draw.
enum class TerrainSprite { Grass = 0, Mountain, Count };
struct TerrainAlphaBBox { int min_x, max_x, min_y, max_y; };
static Texture2D       terrain_tex [(int)TerrainSprite::Count] = {};
static TerrainAlphaBBox terrain_bbox[(int)TerrainSprite::Count] = {};

// Per-sprite tweak; bbox width auto-scales to TILE_W with bottom at diamond bottom-centre.
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
// Tile overlays; ~50% of TILE_W. Forest/Village/Lighthouse live here too (not ResourceTypes).
enum class ResourceSprite { Fruit = 0, Crop, Animal, Metal, Forest, Village, Lighthouse, Count };
static Texture2D        resource_tex [(int)ResourceSprite::Count] = {};
static TerrainAlphaBBox resource_bbox[(int)ResourceSprite::Count] = {};

// Offsets in tile-fractions; scale multiplies the 50%-of-TILE_W base.
static Transformer RESOURCE_TRANSFORMERS[(int)ResourceSprite::Count] = {
    /* Fruit      */ { 0.00f, -0.4f,  0.7f  },
    /* Crop       */ { 0.00f, -0.2f,  1.4f  },
    /* Animal     */ { 0.00f, -0.35f, 0.4f  },
    /* Metal      */ { 0.00f, -0.38f, 0.53f },
    /* Forest     */ { 0.02f, -0.1f,  1.5f  },
    /* Village    */ { 0.00f, -0.3f,  0.9f  },
    /* Lighthouse */ { 0.00f, -0.35f,  0.4f  },
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

// --- City sprites --------------------------------------------------------------
// Per-level sprite (lvl1..lvl9). Clamps to lvl9 for higher levels.
static constexpr int CITY_SPRITE_LEVELS = 9;
static Texture2D        city_tex [CITY_SPRITE_LEVELS] = {};
static TerrainAlphaBBox city_bbox[CITY_SPRITE_LEVELS] = {};
// Per-level transformer — tune each city sprite's offset/scale independently.
static Transformer      CITY_SPRITE_TRANSFORMERS[CITY_SPRITE_LEVELS] = {
    /* lvl1 */ { 0.00f, -0.24f, 0.42f },
    /* lvl2 */ { 0.00f, -0.2f, 0.50f },
    /* lvl3 */ { 0.00f, -0.16f, 0.60f },
    /* lvl4 */ { 0.00f, -0.1f, 0.65f },
    /* lvl5 */ { 0.00f, -0.08f, 0.82f },
    /* lvl6 */ { 0.00f, -0.08f, 0.86f },
    /* lvl7 */ { 0.00f, -0.08f, 0.9f },
    /* lvl8 */ { 0.00f, -0.08f, 0.95f },
    /* lvl9 */ { 0.00f, -0.08f, 1.0f },
};

static inline void draw_city_sprite(Vector2 ctr, int level) {
    int idx = level - 1;
    if (idx < 0) idx = 0;
    if (idx >= CITY_SPRITE_LEVELS) idx = CITY_SPRITE_LEVELS - 1;
    const Texture2D& tex = city_tex[idx];
    if (tex.id == 0) return;
    const TerrainAlphaBBox& bb = city_bbox[idx];
    const Transformer&      xf = CITY_SPRITE_TRANSFORMERS[idx];

    int bbox_w = bb.max_x - bb.min_x;
    if (bbox_w <= 0) return;

    float scale = (float)TILE_W * xf.scale / (float)bbox_w;
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

// --- Star icon ----------------------------------------------------------------
// Loaded from sprites/star.png. Used wherever the UI previously printed a '*'
// for stars (top HUD totals, per-city SPT, per-player SPT subtext).
static Texture2D        g_star_tex  = {};
static TerrainAlphaBBox g_star_bbox = {};

// Width of the rendered star icon when its alpha-bbox height equals target_h.
static inline float star_icon_width(float target_h) {
    if (g_star_tex.id == 0) return 0.0f;
    int bbox_h = g_star_bbox.max_y - g_star_bbox.min_y;
    int bbox_w = g_star_bbox.max_x - g_star_bbox.min_x;
    if (bbox_h <= 0 || bbox_w <= 0) return 0.0f;
    return (float)bbox_w * (target_h / (float)bbox_h);
}

// Draws the star centered at (cx, cy) so its alpha-bbox has height target_h.
static inline void draw_star_icon(float cx, float cy, float target_h, Color tint = WHITE) {
    if (g_star_tex.id == 0) return;
    const TerrainAlphaBBox& bb = g_star_bbox;
    int bbox_h = bb.max_y - bb.min_y;
    int bbox_w = bb.max_x - bb.min_x;
    if (bbox_h <= 0 || bbox_w <= 0) return;
    float scale = target_h / (float)bbox_h;
    float dst_w = (float)g_star_tex.width  * scale;
    float dst_h = (float)g_star_tex.height * scale;
    float bb_cx = (bb.min_x + bb.max_x) * 0.5f;
    float bb_cy = (bb.min_y + bb.max_y) * 0.5f;
    float dx = cx - bb_cx * scale;
    float dy = cy - bb_cy * scale;
    DrawTexturePro(g_star_tex,
        { 0, 0, (float)g_star_tex.width, (float)g_star_tex.height },
        { dx, dy, dst_w, dst_h },
        { 0, 0 }, 0.0f, tint);
}

// --- City names ---------------------------------------------------------------
// Tile index -> name via a stable hash; same map regen = same names.
static const char* CITY_NAMES[] = {
    "Epic", "Iris", "Jaiden", "Steven", "Tina", "WinchestrAve", "McartnyAve",
     "Chatswood", "Lindfield", "North Ryde", "Icado", "Dopilus"
};
static inline const char* city_name_for(int tile_index) {
    constexpr int N = (int)(sizeof(CITY_NAMES) / sizeof(CITY_NAMES[0]));
    uint32_t h = (uint32_t)tile_index;
    h = ((h >> 16) ^ h) * 0x45d9f3bu;
    h = ((h >> 16) ^ h) * 0x45d9f3bu;
    h = (h >> 16) ^ h;
    return CITY_NAMES[h % N];
}

// Pop-bar tile-anchor tweak. x/y_offset in tile-fractions (same convention as
// the other Transformer constants), scale multiplies the base bar size.
static Transformer POP_BAR_TRANSFORMER = { 0.00f, 0.30f, 0.6f };

// Bar-length multiplier indexed by city capacity (= level + 1). Tiny capacities
// shrink the bar so a 2-cell level-1 city doesn't look as wide as a 5-cell city.
static inline float pop_bar_length_factor(int capacity) {
    if (capacity <= 2) return 2.0f / 3.0f;
    if (capacity == 3) return 0.8f;
    if (capacity == 4) return 0.9f;
    return 1.0f;
}

// Light-grey population bar. Divided into `capacity` (= level + 1) subcells.
// Filled width = pop / capacity. One black dot per existing unit fills
// successive subcells from the left. All dimensions/positions derive from
// POP_BAR_TRANSFORMER so a single tweak scales every element together.
static inline void draw_pop_bar(Vector2 ctr, int pop, int capacity,
                                int units_owned, int owner) {
    if (capacity <= 0) return;
    const Transformer& xf = POP_BAR_TRANSFORMER;
    const float W = TILE_W * 0.9f * xf.scale * pop_bar_length_factor(capacity);
    const float H = TILE_H * 0.2f * xf.scale;
    const float X = ctr.x + xf.x_offset * TILE_W - W * 0.5f;
    const float Y = ctr.y + xf.y_offset * TILE_H - H * 0.5f;

    const Color BG_COL      = { 210, 210, 210, 230 };
    const Color OUTLINE_COL = {  90,  90,  90, 230 };
    const Color DIVIDER_COL = {  70,  70,  70, 220 };
    (void)owner;
    const Color FILL_COL    = { 60, 170, 255, 240 };

    Rectangle bg = { X, Y, W, H };
    DrawRectangleRec(bg, BG_COL);

    // Fill shares the EXACT same X/Y/H as the bg, only its width shrinks. Both
    // go through DrawRectangleRec with the same Rectangle semantics, so the
    // top/bottom edges align pixel-perfectly with no anti-alias mismatch.
    if (pop > 0) {
        float frac = (float)pop / (float)capacity;
        if (frac > 1.0f) frac = 1.0f;
        Rectangle filled = { X, Y, W * frac, H };
        DrawRectangleRec(filled, FILL_COL);
    }

    const float line_thick = H * 0.10f > 1.0f ? H * 0.10f : 1.0f;
    for (int seg = 1; seg < capacity; seg++) {
        float lx = X + (W * seg) / (float)capacity;
        DrawLineEx({ lx, Y }, { lx, Y + H }, line_thick, DIVIDER_COL);
    }

    int dots = units_owned < capacity ? units_owned : capacity;
    // Integer side + integer top-left so the square rasterizes evenly on every
    // pixel; floats give one extra row/column on some dots and look rectangular.
    int side = (int)(H * 0.30f + 0.5f);
    if (side < 2) side = 2;
    for (int i = 0; i < dots; i++) {
        float dot_cx = X + (W * (i + 0.5f)) / (float)capacity;
        float dot_cy = Y + H * 0.5f;
        int left = (int)(dot_cx - side * 0.5f + 0.5f);
        int top  = (int)(dot_cy - side * 0.5f + 0.5f);
        DrawRectangle(left, top, side, side, BLACK);
    }

    DrawRectangleLinesEx(bg, line_thick, OUTLINE_COL);
}

// Per-city label rendered below the pop bar: blue translucent banner with
// "Name [star]+SPT" in white. Called from a dedicated post-units pass so the
// banner sits in front of everything (including units on the city tile).
static inline void draw_city_label(Vector2 ctr, const char* name, int spt) {
    constexpr int   FS              = 6;
    constexpr float GAP_NAME_STAR   = 2.0f;
    constexpr float GAP_STAR_NUM    = 0.7f;
    constexpr float BANNER_PAD_X    = 4.0f;
    constexpr float BANNER_PAD_Y    = 3.5f;
    const     Color BANNER_COL      = {  30,  90, 200, 170 };
    const     Color TEXT_COL        = WHITE;

    char spt_buf[16];
    snprintf(spt_buf, sizeof(spt_buf), "%d", spt);
    int name_w = MeasureTextC(name,    FS);
    int spt_w  = MeasureTextC(spt_buf, FS);

    float star_h = (float)FS * 0.9f;
    float star_w = star_icon_width(star_h);

    float total_w = (float)name_w + GAP_NAME_STAR + star_w + GAP_STAR_NUM + (float)spt_w;
    float cx = ctr.x;
    // Place the banner just above the pop bar's top edge. Derived from
    // POP_BAR_TRANSFORMER so any future bar tweak shifts the label in step.
    constexpr float LABEL_GAP_TO_BAR = 2.0f;
    const float pop_bar_h     = (float)TILE_H * 0.14f * POP_BAR_TRANSFORMER.scale;
    const float pop_bar_top_y = ctr.y + POP_BAR_TRANSFORMER.y_offset * (float)TILE_H
                              - pop_bar_h * 0.5f;
    const float label_half_h  = (float)FS * 0.5f + BANNER_PAD_Y;
    float cy = pop_bar_top_y - LABEL_GAP_TO_BAR - label_half_h;

    Rectangle banner = {
        cx - total_w * 0.5f - BANNER_PAD_X,
        cy - (float)FS * 0.5f - BANNER_PAD_Y,
        total_w + BANNER_PAD_X * 2.0f,
        (float)FS + BANNER_PAD_Y * 2.0f
    };
    DrawRectangleRec(banner, BANNER_COL);

    float x  = cx - total_w * 0.5f;
    int   ty = (int)(cy - (float)FS * 0.5f);

    DrawTextC(name, (int)x, ty, FS, TEXT_COL);
    x += (float)name_w + GAP_NAME_STAR;

    if (star_w > 0.0f) {
        draw_star_icon(x + star_w * 0.5f, cy, star_h);
        x += star_w + GAP_STAR_NUM;
    }

    DrawTextC(spt_buf, (int)x, ty, FS, TEXT_COL);
}

// Mirrors GameState::get_defense_bonus, but const-friendly so it can be called
// from the render loop. Returns 1.0/1.5/2.0; 2.0 = fortify on tech-defended
// terrain AND city walls.
static inline float defense_bonus_for_render(const GameState& s, const Tile& t) {
    if (!t.has_unit()) return 1.0f;
    const Unit& u = s.get_unit(t.unit_id());
    if (!u.has_ability(ABILITY_FORTIFY)) return 1.0f;
    TerrainType terrain = t.terrain();
    const Player& p = s.get_player(u.owner());
    float total = 1.0f;
    for (TechType tech : p.get_techs()) {
        const auto& terrains = tech_def(tech).defense_terrains;
        if (std::find(terrains.begin(), terrains.end(), terrain) != terrains.end()) {
            total = 1.5f;
            break;
        }
    }
    if (t.has_city()) {
        total += 0.5f;
        if (s.get_city(t.city_id()).has_walls()) total += 0.5f;
    }
    return total;
}

// Pentagon shield outline (flat top, V bottom). `size` = width in pixels;
// height auto-scales. Caller stamps it twice — black at +1/+1, then white on
// top — to get the same drop-shadow look as the HP text.
static inline void draw_shield(float cx, float cy, float size, Color outline) {
    // DrawLineV uses GL_LINES → true 1-pixel strokes (DrawLineEx draws a quad
    // that smears slightly thicker than 1 px). Endpoints are snapped to whole
    // pixels SYMMETRICALLY around the integer-snapped centre — otherwise a
    // float cx that lands between pixels (e.g. 10.3) rounds lx and rx in
    // opposite directions and one leg of the V ends up a pixel longer.
    int icx     = (int)floorf(cx + 0.5f);
    int half_w  = (int)floorf(size * 0.5f + 0.5f);
    int half_h  = (int)floorf(size * 0.6f + 0.5f);     // h = size * 1.2
    // mid_y pushed down so the V is shallower (was size*0.18). Triangle height
    int mid_off = (int)floorf(size * 0.24f + 0.5f);
    int top_y   = (int)floorf(cy + 0.5f) - half_h;
    int mid_y   = (int)floorf(cy + 0.5f) + mid_off;
    int bot_y   = (int)floorf(cy + 0.5f) + half_h;
    int ilx     = icx - half_w;
    int irx     = icx + half_w;
    // Edges and diagonals are drawn so mirrored pairs share the same start
    // direction — GL_LINES rasterization (diamond-exit rule) drops the end
    // pixel, so a `right: mid→apex` paired with `left: apex→mid` ends up one
    // pixel asymmetric. Both diagonals now start at the apex.
    auto P = [](int x, int y) -> Vector2 { return { (float)x, (float)y }; };
    DrawLineV(P(ilx, top_y), P(irx, top_y), outline);   // top edge
    DrawLineV(P(irx, top_y), P(irx, mid_y), outline);   // right edge
    DrawLineV(P(ilx, top_y), P(ilx, mid_y), outline);   // left  edge (same dir as right)
    DrawLineV(P(icx, bot_y), P(irx, mid_y), outline);   // right diag (apex → mid)
    DrawLineV(P(icx, bot_y), P(ilx, mid_y), outline);   // left  diag (apex → mid)
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

// --- City borders --------------------------------------------------------------
// Per-tile per-direction per-owner draw flags rebuilt each frame from
// Tile::border_city_id(). Two friendly cities meeting along an edge cancel each
// other on that edge (same owner, different city); enemy-vs-friendly edges keep
// both stripes and overlap visually. Both sides of each edge set their own flag
// so the near-tile in the back-to-front pass redraws over any terrain occlusion.
//   dir 0: neighbor (r-1, c) — screen lower-right → BR diamond edge
//   dir 1: neighbor (r+1, c) — screen upper-left  → TL diamond edge
//   dir 2: neighbor (r, c-1) — screen lower-left  → BL diamond edge
//   dir 3: neighbor (r, c+1) — screen upper-right → TR diamond edge
// Opposite dir = d ^ 1 (0↔1, 2↔3).
static bool border_draw_flags[MAX_MAP_TILES * 4][2] = {};
static const int BORDER_DR[4] = { -1, +1,  0,  0 };
static const int BORDER_DC[4] = {  0,  0, -1, +1 };

static void compute_border_flags(const GameState& s, const GameState& fog_state,
                                 int vp, bool omni) {
    const int msz  = s.map_size();
    const int mtsz = s.map_tiles();
    for (int i = 0; i < mtsz * 4; i++) {
        border_draw_flags[i][0] = false;
        border_draw_flags[i][1] = false;
    }
    for (int idx = 0; idx < mtsz; idx++) {
        const Tile& t = s.tile_at(idx);
        int cid = t.border_city_id();
        if (cid < 0) continue;
        int owner = s.get_city(cid).owner();
        int r = idx / msz, c = idx % msz;
        bool self_fogged = !omni && !fog_state.is_visible(vp, idx);
        for (int d = 0; d < 4; d++) {
            int nr = r + BORDER_DR[d], nc = c + BORDER_DC[d];
            bool oob = (nr < 0 || nr >= msz || nc < 0 || nc >= msz);
            int n_idx   = oob ? -1 : (nr * msz + nc);
            int n_cid   = -1;
            int n_owner = -1;
            if (!oob) {
                const Tile& nt = s.tile_at(n_idx);
                n_cid = nt.border_city_id();
                if (n_cid >= 0) n_owner = s.get_city(n_cid).owner();
            }
            if (n_cid == cid)     continue;       // same territory
            if (n_owner == owner) continue;       // friendly border cancels
            // Hide edges that are entirely in fog: an edge only shows when at
            // least one of its two tiles is explored (OOB counts as fogged).
            bool n_fogged = oob || (!omni && !fog_state.is_visible(vp, n_idx));
            if (self_fogged && n_fogged) continue;
            // Enemy borders on the same edge: both colours draw. Per-tile
            // render order (in draw_tile_borders) gives the front tile's
            // colour layer priority — its own stripe ends up on top.
            border_draw_flags[idx * 4 + d][owner] = true;
            if (n_idx >= 0)
                border_draw_flags[n_idx * 4 + (d ^ 1)][owner] = true;
        }
    }
}

// Parallelogram stripes along the diamond edge: top/bottom parallel to the
// edge tangent, left/right vertical. Both stripe half-length and gaps between
// stripes are sampled from per-edge uniform distributions; the RNG is seeded
// by the edge endpoints so the pattern stays stable across frames.
// `right_edge`: true for "/" edges (BR + TL), false for "\" edges (BL + TR).
// Right/left orientations get distinct colours to read as facing direction.
static inline void draw_border_stripes(Vector2 a, Vector2 b, int owner, bool right_edge) {
    float ex = b.x - a.x, ey = b.y - a.y;
    float L = sqrtf(ex * ex + ey * ey);
    if (L < 1.0f) return;
    float tx = ex / L, ty = ey / L;
    // Four colours total: rich saturated blue / red for "/" edges, darker
    // toned variants for "\" edges. Owner picks blue (P0) vs red (P1).
    Color col = right_edge
        ? (owner == 0 ? Color{  20,  70, 235, 235 } : Color{ 235,  70,  70, 235 })
        : (owner == 0 ? Color{  15,  40, 130, 235 } : Color{ 150,  35,  35, 235 });
    const float STRIPE_HALF_H   = TILE_H * 0.06f;    // vertical half-thickness
    const float STRIPE_Y_OFFSET = -TILE_H * 0.06f;   // push stripes upward
    const float HL_MIN          = TILE_H * 0.03f;    // half-length min
    const float HL_MAX          = TILE_H * 0.07f;    // half-length max
    const float GAP_MIN         = TILE_H * 0.04f;     // gap range halved
    const float GAP_MAX         = TILE_H * 0.05f;

    // Per-edge deterministic xorshift32 seeded by endpoint coords + owner.
    uint32_t seed = (uint32_t)((int)(a.x * 31.0f) * 73856093u)
                  ^ (uint32_t)((int)(a.y * 31.0f) * 19349663u)
                  ^ (uint32_t)((int)(b.x * 31.0f) * 83492791u)
                  ^ (uint32_t)((int)(b.y * 31.0f) * 50331653u)
                  ^ (uint32_t)(owner * 2654435761u);
    if (seed == 0) seed = 0xA53B7C1Du;
    auto next_u01 = [&]() {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        return (seed & 0xFFFFFFu) / (float)0xFFFFFFu;
    };
    auto uniform = [&](float lo, float hi) { return lo + next_u01() * (hi - lo); };

    // Walk along the edge laying down stripes + gaps. If the next stripe would
    // overflow past L it gets truncated to fit; we still draw a smaller stripe
    // up to the vertex rather than skipping it.
    float s = uniform(GAP_MIN, GAP_MAX) * 0.5f;
    while (s < L) {
        float half_len = uniform(HL_MIN, HL_MAX);
        float full_len = 2.0f * half_len;
        float end = s + full_len;
        if (end > L) end = L;
        float length = end - s;
        if (length < 0.5f) break;
        float center = (s + end) * 0.5f;
        float half_local = length * 0.5f;
        float center_t = center / L;
        float px = a.x + ex * center_t;
        float py = a.y + ey * center_t + STRIPE_Y_OFFSET;
        Vector2 A = { px - tx * half_local, py - ty * half_local - STRIPE_HALF_H };
        Vector2 B = { px - tx * half_local, py - ty * half_local + STRIPE_HALF_H };
        Vector2 C = { px + tx * half_local, py + ty * half_local + STRIPE_HALF_H };
        Vector2 D = { px + tx * half_local, py + ty * half_local - STRIPE_HALF_H };
        DrawTriangle(A, B, C, col);
        DrawTriangle(A, C, D, col);
        s = end + uniform(GAP_MIN, GAP_MAX);
    }
}

// Solid lighter-tone parallelogram spanning the full edge. Used to highlight a
// selected city's territory boundary — same vertical-sided shape as the stripe
// parallelograms, but one continuous block per edge rather than dashed.
static inline void draw_temp_border(Vector2 a, Vector2 b, int owner) {
    // Normalize left-to-right so the A,B,C / A,C,D triangulation is always CCW
    // on screen; raylib's DrawTriangle culls CW. For BL edges (a=vbot, b=vleft)
    // b.x < a.x, which produces CW winding and an invisible parallelogram.
    if (b.x < a.x) { Vector2 t = a; a = b; b = t; }
    float ex = b.x - a.x, ey = b.y - a.y;
    float L = sqrtf(ex * ex + ey * ey);
    if (L < 1.0f) return;
    float tx = ex / L, ty = ey / L;
    Color col = (owner == 0)
        ? Color{ 130, 190, 255, 180 }
        : Color{ 255, 160, 160, 180 };
    const float HALF_H   = TILE_H * 0.06f;
    const float Y_OFFSET = -TILE_H * 0.06f;
    float mid_x = (a.x + b.x) * 0.5f;
    float mid_y = (a.y + b.y) * 0.5f + Y_OFFSET;
    float half_len = L * 0.5f;
    Vector2 A = { mid_x - tx * half_len, mid_y - ty * half_len - HALF_H };
    Vector2 B = { mid_x - tx * half_len, mid_y - ty * half_len + HALF_H };
    Vector2 C = { mid_x + tx * half_len, mid_y + ty * half_len + HALF_H };
    Vector2 D = { mid_x + tx * half_len, mid_y + ty * half_len - HALF_H };
    DrawTriangle(A, B, C, col);
    DrawTriangle(A, C, D, col);
}

// Direction-mask helpers: bit d (= 0..3) is BR/TL/BL/TR respectively.
//   BORDER_DIRS_ALL    — all 4 edges (default).
//   BORDER_DIRS_BACK   — TL + TR only (upper-back edges, occluded by mountain).
//   BORDER_DIRS_FRONT  — BR + BL only (lower-front edges, sit in front of mountain).
static constexpr int BORDER_DIRS_ALL   = 0xF;
static constexpr int BORDER_DIRS_BACK  = (1 << 1) | (1 << 3);
static constexpr int BORDER_DIRS_FRONT = (1 << 0) | (1 << 2);

// `self_owner`: this tile's own border owner (0 or 1), or -1 if the tile has
// no border city. The tile's own colour is drawn LAST so it sits on top;
// combined with the back-to-front pass, this gives the front tile's stripe
// layer priority on edges with conflicting enemy borders.
// `dir_mask` selects which of the 4 directions to render (bit per dir).
static inline void draw_tile_borders(int idx, Vector2 ctr, int self_owner,
                                     int dir_mask = BORDER_DIRS_ALL) {
    Vector2 vtop   = { ctr.x,                  ctr.y - TILE_H * 0.5f };
    Vector2 vright = { ctr.x + TILE_W * 0.5f,  ctr.y                 };
    Vector2 vbot   = { ctr.x,                  ctr.y + TILE_H * 0.5f };
    Vector2 vleft  = { ctr.x - TILE_W * 0.5f,  ctr.y                 };
    Vector2 edges[4][2] = {
        { vbot,  vright },  // dir 0: BR
        { vleft, vtop   },  // dir 1: TL
        { vbot,  vleft  },  // dir 2: BL
        { vtop,  vright },  // dir 3: TR
    };
    int order[2] = { 0, 1 };
    if (self_owner == 0) { order[0] = 1; order[1] = 0; }
    for (int d = 0; d < 4; d++) {
        if (!(dir_mask & (1 << d))) continue;
        // d 0 (BR) and 1 (TL) are the "/" diagonals — "right" edges.
        // d 2 (BL) and 3 (TR) are the "\" diagonals — "left" edges.
        bool right_edge = (d < 2);
        for (int i = 0; i < 2; i++) {
            int p = order[i];
            if (border_draw_flags[idx * 4 + d][p])
                draw_border_stripes(edges[d][0], edges[d][1], p, right_edge);
        }
    }
}

static std::string format_action_str(const Action& a, int player, int turn, int sz) {
    auto coords = [sz](int idx) {
        char buf[12]; snprintf(buf, sizeof(buf), "(%d,%d)", idx / sz, idx % sz); return std::string(buf);
    };
    static const char* unit_names[] = {"?","Warrior","Archer","Rider","Defender","Giant"};
    static const char* tech_names[] = {"Origin","Hunting","Org","Farming","Riding","Climb","Archery","Mining"};
    std::string p = "";
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
        case ActionType::Recover:           return p + "Recover " + coords(a.from);
        case ActionType::TrainUnit:
            return p + "Train " + (a.param > 0 && a.param < (int)UnitType::Count ? unit_names[a.param] : "?") + " @ " + coords(a.from);
        case ActionType::ResearchTech:
            return p + "Tech " + (a.param >= 0 && a.param < 8 ? tech_names[a.param] : "?");
        default: return p + "Action";
    }
}

// Colourer — assigns each city a distinct shade in the owner's colour family.
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

enum class ActionCategory { None, Train, Move, Attack, Capture, Harvest, Recover, Research, Upgrade, Debug, EndTurn };

static ActionCategory action_category(const Action& a) {
    switch (a.type) {
        case ActionType::TrainUnit:       return ActionCategory::Train;
        case ActionType::Move:            return ActionCategory::Move;
        case ActionType::Attack:          return ActionCategory::Attack;
        case ActionType::CaptureCity:     return ActionCategory::Capture;
        case ActionType::HarvestResource: return ActionCategory::Harvest;
        case ActionType::Recover:         return ActionCategory::Recover;
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
        case ActionCategory::Recover: return "Recover";
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
            static const char* unit_names[] = { "?", "Warrior", "Archer", "Rider", "Defender", "Giant" };
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
        case ActionType::Recover:
            to_coords(a.from, fx, fy);
            snprintf(buf, size, "Recover (%d,%d)", fx, fy);
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


// Radial tech tree — three concentric rings around a central Origin.
//   Tier 1: 5 slots on a ring at radius T1_DIST. Angles start at top (90°)
//           and step −72° clockwise: Riding, Organisation, Climbing,
//           Fishing (placeholder, not in engine), Hunting.
//   Tier 2: 10 slots on a ring at radius T2_DIST. Angles start at 18° and
//           step +36° counter-clockwise. Existing tier-2 techs sit on the
//           slot aligned with their parent's ray.
//   Tier 3: 10 slots on a ring at radius T3_DIST, same angles as tier 2.
//           No engine techs yet; all slots render as faded placeholders.
// All distances are in unit-square [0,1] coords (math: x→right, y→up) and
// scale with `size` at draw time, so the layout is resolution-independent.
struct TechNode {
    TechType type;     // TechType::None ⇒ empty placeholder slot
    int      parent;   // index into NODES, or -1 (no edge drawn)
    float    ux, uy;
};

namespace {
    constexpr float T1_DIST = 0.15f;
    constexpr float T2_DIST = 0.30f;
    constexpr float T3_DIST = 0.45f;

    // Hardcoded cos/sin tables (constexpr trig isn't portable).
    // Tier 1 angles: 90°, 18°, −54°, −126°, −198° (top then clockwise)
    constexpr float T1_COS[5] = { 0.00000f,  0.95106f,  0.58779f, -0.58779f, -0.95106f };
    constexpr float T1_SIN[5] = { 1.00000f,  0.30902f, -0.80902f, -0.80902f,  0.30902f };
    // Tier 2 & 3 angles: 18°, 54°, 90°, 126°, 162°, 198°, 234°, 270°, 306°, 342°
    constexpr float TN_COS[10] = {
        0.30902f,  0.80902f,  1.0f, 0.80902f, 0.30902f,
       -0.30902f,  -0.80902f,  -1.0f, -0.80902f, -0.30902f,
    };
    constexpr float TN_SIN[10] = {
        0.95106f,  0.58779f,  0.0f,  -0.58779f,  -0.95106f,
       -0.95106f,  -0.58779f,  0.0f,  0.58779f,  0.95106f,
    };

    inline constexpr float t1_ux(int i) { return 0.5f + T1_DIST * T1_COS[i]; }
    inline constexpr float t1_uy(int i) { return 0.5f + T1_DIST * T1_SIN[i]; }
    inline constexpr float t2_ux(int i) { return 0.5f + T2_DIST * TN_COS[i]; }
    inline constexpr float t2_uy(int i) { return 0.5f + T2_DIST * TN_SIN[i]; }
    inline constexpr float t3_ux(int i) { return 0.5f + T3_DIST * TN_COS[i]; }
    inline constexpr float t3_uy(int i) { return 0.5f + T3_DIST * TN_SIN[i]; }

    constexpr int NODE_COUNT = 26;  // 1 origin + 5 tier-1 + 10 tier-2 + 10 tier-3

    // Stable parent indices into NODES[] for edge lookup.
    constexpr int N_RIDING       = 1;
    constexpr int N_ORGANISATION = 2;
    constexpr int N_CLIMBING     = 3;
    constexpr int N_HUNTING      = 5;
}

static const TechNode NODES[NODE_COUNT] = {
    // Origin (centre)
    { TechType::Origin,        -1,             0.5f,     0.5f     },  // 0

    // Tier 1: top, then clockwise
    { TechType::Riding,         0,             t1_ux(0), t1_uy(0) },  // 1  90°  top
    { TechType::Organisation,   0,             t1_ux(1), t1_uy(1) },  // 2  18°  upper-right
    { TechType::Climbing,       0,             t1_ux(2), t1_uy(2) },  // 3 −54°  lower-right
    { TechType::None,           0,             t1_ux(3), t1_uy(3) },  // 4 −126° Fishing (placeholder)
    { TechType::Hunting,        0,             t1_ux(4), t1_uy(4) },  // 5 −198° upper-left

    // Tier 2: 10-slot ring; bound techs aligned with parent's ray when possible
    { TechType::None,          0,               t2_ux(0), t2_uy(0) }, 
    { TechType::Farming,       N_ORGANISATION,  t2_ux(1), t2_uy(1) }, 
    { TechType::Strategy,      N_ORGANISATION,  t2_ux(2), t2_uy(2) },  
    { TechType::Mining,        N_CLIMBING,      t2_ux(3), t2_uy(3) }, 
    { TechType::None,          -1,              t2_ux(4), t2_uy(4) }, 
    { TechType::None,          -1,              t2_ux(5), t2_uy(5) },  
    { TechType::None,          -1,              t2_ux(6), t2_uy(6) }, 
    { TechType::Archery,        N_HUNTING,      t2_ux(7), t2_uy(7) }, 
    { TechType::None,           -1,              t2_ux(8), t2_uy(8) },
    { TechType::None,          -1,              t2_ux(9), t2_uy(9) },

    // Tier 3: 10-slot ring at T3_DIST, all empty placeholders
    { TechType::None,          -1,              t3_ux(0), t3_uy(0) },  // 16 18°
    { TechType::None,          -1,              t3_ux(1), t3_uy(1) },  // 17 54°
    { TechType::None,          -1,              t3_ux(2), t3_uy(2) },  // 18 90°
    { TechType::None,          -1,              t3_ux(3), t3_uy(3) },  // 19 126°
    { TechType::None,          -1,              t3_ux(4), t3_uy(4) },  // 20 162°
    { TechType::None,          -1,              t3_ux(5), t3_uy(5) },  // 21 198°
    { TechType::None,          -1,              t3_ux(6), t3_uy(6) },  // 22 234°
    { TechType::None,          -1,              t3_ux(7), t3_uy(7) },  // 23 270°
    { TechType::None,          -1,              t3_ux(8), t3_uy(8) },  // 24 306°
    { TechType::None,          -1,              t3_ux(9), t3_uy(9) },  // 25 342°
};

// Draws the tech tree inside a square anchored at (bl_x, bl_y) (screen-space
// bottom-left corner). All sizes (node radius, font, edge thickness) scale
// with `size` so the layout is fully resolution-independent.
static void draw_tech_tree(uint32_t owned_techs, int player_idx,
                           float bl_x, float bl_y, float size,
                           int hover_tech = -1)
{
    const Color owned_col  = (player_idx == 0) ? COL_P0 : COL_P1;
    const Color avail_col  = { 130, 130, 130, 255 };
    const Color locked_col = {  55,  55,  55, 255 };
    const Color edge_col   = {  70,  70,  70, 255 };
    const Color hover_col  = { 255, 220,  60, 255 };

    const float node_r        = size * 0.035f;
    const float placeholder_r = node_r * 0.65f;
    const float edge_thk      = std::max(1.0f, size * 0.010f);
    const int   font_sz       = (int)std::max(8.0f, size * 0.040f);

    auto to_screen = [&](float ux, float uy) -> Vector2 {
        return { bl_x + ux * size, bl_y - uy * size };
    };

    uint32_t avail = available_techs(owned_techs);

    // Edges first so nodes draw on top. Only drawn for bound techs.
    for (int i = 0; i < NODE_COUNT; i++) {
        if (NODES[i].parent < 0) continue;
        if (NODES[i].type == TechType::None) continue;
        Vector2 p1 = to_screen(NODES[NODES[i].parent].ux, NODES[NODES[i].parent].uy);
        Vector2 p2 = to_screen(NODES[i].ux, NODES[i].uy);
        bool  is_hover_edge = (hover_tech == static_cast<int>(NODES[i].type));
        Color ec = is_hover_edge ? hover_col : edge_col;
        float thick = is_hover_edge ? edge_thk * 1.7f : edge_thk;
        DrawLineEx(p1, p2, thick, ec);
    }

    // Nodes
    const Color placeholder_col = { 45, 45, 45, 220 };
    for (int i = 0; i < NODE_COUNT; i++) {
        Vector2 p = to_screen(NODES[i].ux, NODES[i].uy);

        // Empty slots render as small faded dots so the ring structure is visible.
        if (NODES[i].type == TechType::None) {
            DrawCircleV(p, placeholder_r, placeholder_col);
            continue;
        }

        bool is_owned = (owned_techs >> static_cast<int>(NODES[i].type)) & 1;
        bool is_avail = (avail       >> static_cast<int>(NODES[i].type)) & 1;
        bool is_hov   = (hover_tech  == static_cast<int>(NODES[i].type));

        Color fill = is_hov    ? hover_col
                   : is_owned  ? owned_col
                   : is_avail  ? avail_col
                   : locked_col;
        // Outline = larger filled circle behind the fill (DrawCircleLines
        // rounds to int and misaligns from the float-radius fill at small sizes).
        Color outline = is_hov ? WHITE : Color{ 0, 0, 0, 180 };
        float outline_thk = std::max(1.0f, node_r * 0.12f);
        DrawCircleV(p, node_r + outline_thk, outline);
        DrawCircleV(p, node_r, fill);

        if (NODES[i].type != TechType::Origin) {
            const char* name = tech_def(NODES[i].type).name;
            int tw = MeasureTextC(name, font_sz);
            Color tc = (is_owned || is_hov) ? WHITE
                     : is_avail ? Color{200,200,200,255}
                     : Color{100,100,100,255};
            DrawTextC(name, (int)(p.x - tw * 0.5f), (int)(p.y + node_r + 2), font_sz, tc);
        }
    }
}

int main(int argc, char** argv) {
    Logger::debugEnabled = true;

    // Which AI folder to embed. Defaults to the phase-1 heuristic bot; swap
    // with `--ai-dir <path>` to point at another bot directory that exposes
    // a top-level `heuristics.py` with a `state_value(state, player)` fn.
    std::string ai_dir = "ai/mcts-phase1";
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--ai-dir") {
            ai_dir = argv[i + 1];
            break;
        }
    }

    // Map side length for interactive games (`--size <n>`). Defaults to the
    // engine default. Clamped to [2, MAX_MAP_SIZE] — the board arrays are fixed
    // at MAX_MAP_TILES, so a larger size overflows them and aborts.
    int map_sz = cfg::map::DEFAULT_SIZE;
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--size") {
            map_sz = std::atoi(argv[i + 1]);
            if (map_sz < 2)            map_sz = 2;
            if (map_sz > MAX_MAP_SIZE) map_sz = MAX_MAP_SIZE;
            break;
        }
    }

    // ----- Embedded Python interpreter (for calling heuristics.py from UI) -----
    // The scoped_interpreter MUST outlive every py:: object below, so it lives
    // in main()'s scope. JSON is used to ferry the GameState across the
    // boundary: the visualizer's pybind11 doesn't share a type registry with
    // polyshark.cpython-*.so, so passing a GameState directly wouldn't cast.
    namespace py = pybind11;
    std::optional<py::scoped_interpreter> py_guard;
    py::object py_heuristics;
    py::object py_polyshark;
    bool python_ok = false;
    try {
        py_guard.emplace();
        py::module_ sys_mod = py::module_::import("sys");
        py::list path = sys_mod.attr("path");
        // Order matters — later inserts take priority. The AI dir is first
        // so its `heuristics.py` shadows anything else.
        path.attr("insert")(0, "build/bindings");
        path.attr("insert")(0, ai_dir);
        py_polyshark  = py::module_::import("polyshark");
        py_heuristics = py::module_::import("heuristics");
        python_ok = true;
        Logger::print("Python embedded; heuristics loaded from %s", ai_dir.c_str());
    } catch (const std::exception& e) {
        Logger::print("Python init failed (EVAL disabled): %s", e.what());
    }

    auto eval_state_value = [&](const GameState& gs) -> std::optional<double> {
        if (!python_ok) return std::nullopt;
        try {
            std::string js = GameState::serialise(gs).dump();
            py::object state = py_polyshark.attr("GameState").attr("deserialise")(js);
            py::object val   = py_heuristics.attr("state_value")(state, gs.current_player());
            return val.cast<double>();
        } catch (const std::exception& e) {
            Logger::print("EVAL failed: %s", e.what());
            return std::nullopt;
        }
    };

    uint64_t gen_seed = 1;
    int climate[MAX_MAP_TILES] = {};
    auto new_map = [&]() {
        MapGenParams p = MapGenParams::for_biome(BiomeType::Drylands, map_sz);
        p.seed = gen_seed++;
        MapGenResult r = MapGen(p).generate();
        int mtsz = r.state.map_tiles();
        for (int i = 0; i < mtsz; i++) climate[i] = r.climate[i];
        return r.state;
    };
    GameState initial = new_map();
    TILE_W = iso_grid_width() / initial.map_size();
    TILE_H = (TILE_W * 3) / 5;
    GameState s = initial;

    // Per-unit facing (visual only); false = facing right, true = flipped. Reset on regen.
    bool unit_facing_left[MAX_MAP_TILES] = {};
    auto reset_facing = [&]() {
        for (auto& f : unit_facing_left) f = false;
    };

    // Single in-flight move/attack anim. Duration = FIRST_HOP + EXTRA_HOP * (hops - 1).
    //   1 hop → 0.20s, 2 hops → 0.26s, 3 hops → 0.32s.
    // Reused for Attack: kill+advance = 1-hop move, lunge = triangle-wave offset.
    constexpr double MOVE_ANIM_FIRST_HOP_SECS = 0.2;
    constexpr double MOVE_ANIM_EXTRA_HOP_SECS = 0.06;
    constexpr double ATTACK_LUNGE_SECS = 0.2;
    constexpr float  ATTACK_LUNGE_FRAC = 0.35f;
    constexpr double ATTACK_RANGED_SECS = 0.20;
    // pre_state defers fog reveal (and contents in freshly-walked tiles) until anim ends.
    struct MoveAnimation {
        bool   active     = false;
        bool   lunge      = false;
        int    unit_id    = -1;
        int    path_tiles[MAX_MOVE_PATH_STEPS + 1] = {};
        int    path_count = 0;
        double start_time = 0.0;
        double duration   = 0.0;
        GameState pre_state;
    };
    MoveAnimation move_anim;

    // Ranged projectile arcs from→to; pre_state keeps the defender intact until impact.
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
        // New Move cancels any in-flight ranged projectile.
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
        move_anim.pre_state  = pre_state;
        move_anim.active     = true;
    };

    // Three variants: ranged (projectile arc), melee kill+advance (1-hop move),
    // melee lunge (triangle wave). Skip melee if attacker died — no sprite to animate.
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
        move_anim.pre_state = pre_state;
        move_anim.active = true;
    };

    // Action log: text + parallel colour from Logger. Live entries come from
    // Logger::print (apply-action site), replay entries from replay_log below.
    std::vector<std::string> action_log;
    std::vector<LogColor>    action_log_color;
    int last_logger_count = 0;

    auto to_log_color = [](Color c) -> LogColor {
        return LogColor{ c.r, c.g, c.b, c.a };
    };
    auto from_log_color = [](LogColor c) -> Color {
        return Color{ c.r, c.g, c.b, c.a };
    };
    auto player_log_color = [&](int player) -> LogColor {
        return to_log_color(player == 0 ? COL_P0 : COL_P1);
    };
    auto log_turn_header = [&](int turn, int player) {
        Logger::print(player_log_color(player), "Turn %d Player %d", turn, player);
    };

    // EVAL toggle state. When `eval_on` is true the bar is rendered and
    // every applied action triggers a recompute + log. `eval_value` is the
    // cached result from the last compute (in `eval_perspective`'s frame).
    bool   eval_on          = false;
    bool   eval_has_value   = false;
    double eval_value       = 0.0;
    int    eval_perspective = 0;

    auto refresh_eval = [&](bool log_it) {
        if (!eval_on) return;
        auto v = eval_state_value(s);
        if (!v.has_value()) return;
        eval_value       = *v;
        eval_perspective = s.current_player();
        eval_has_value   = true;
        if (log_it) {
            Logger::print(player_log_color(eval_perspective),
                          "Heuristic value (P%d): %+.3f",
                          eval_perspective, eval_value);
        }
    };

    // Explorer-upgrade walk paths (visualizer-only).
    std::vector<std::vector<int>> explorer_trails;
    int log_scroll = 0;

    // Save-as UI: SAVE button toggles into an inline textbox.
    bool save_input_active = false;
    char save_filename[64] = {};
    int  save_filename_len = 0;
    bool save_filename_invalid = false;  // shake/red border on invalid submit

    auto is_valid_filename_char = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
            || (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '_';
    };
    auto is_valid_filename = [&](const char* name, int len) {
        if (len == 0) return false;
        if (name[0] == '.' || name[0] == '-') return false;
        for (int i = 0; i < len; i++) if (!is_valid_filename_char(name[i])) return false;
        return true;
    };
    auto close_save_input = [&]() {
        save_input_active = false;
        save_filename_len = 0;
        save_filename[0]  = 0;
        save_filename_invalid = false;
    };
    auto do_save = [&]() {
        if (!is_valid_filename(save_filename, save_filename_len)) {
            save_filename_invalid = true;
            return;
        }
        namespace fs = std::filesystem;
        fs::path dir = "data/saved_gamestates";
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) {
            Logger::print("Save failed: %s (%s)", dir.string().c_str(), ec.message().c_str());
            save_filename_invalid = true;
            return;
        }
        std::string name(save_filename, save_filename_len);
        if (name.size() < 5 || name.compare(name.size() - 5, 5, ".json") != 0)
            name += ".json";
        fs::path path = dir / name;
        if (fs::exists(path)) {
            Logger::print("Save aborted: %s already exists", path.string().c_str());
            save_filename_invalid = true;
            return;
        }
        std::ofstream f(path);
        if (!f) {
            Logger::print("Save failed: cannot open %s", path.string().c_str());
            save_filename_invalid = true;
            return;
        }
        f << GameState::serialise(s).dump(2);
        if (!f) {
            Logger::print("Save failed while writing %s", path.string().c_str());
            save_filename_invalid = true;
            return;
        }
        Logger::print("Gamestate Saved as %s", save_filename);
        close_save_input();
    };

    // Load-panel state. `load_open` toggles the overlay above the SAVE/LOAD
    // row; `save_files` is rescanned when the panel opens or after a delete.
    bool load_open = false;
    std::vector<std::string> save_files;
    int  load_scroll = 0;

    auto rescan_saves = [&]() {
        save_files.clear();
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::path dir = "data/saved_gamestates";
        if (!fs::exists(dir, ec)) return;
        for (auto& e : fs::directory_iterator(dir, ec)) {
            if (e.is_regular_file(ec) && e.path().extension() == ".json")
                save_files.push_back(e.path().filename().string());
        }
        std::sort(save_files.begin(), save_files.end());
        load_scroll = 0;
    };

    auto clear_visuals = [&]() {
        explorer_trails.clear();
    };

    // Replay mode: --replay <path>
    bool replay_mode = false;
    std::vector<GameState> replay_states;
    std::vector<Action>    replay_actions;  // one Action per applied step (paths populated for Move)
    std::vector<std::string> replay_log;        // one entry per action
    std::vector<LogColor>    replay_log_color;  // parallel colour per replay entry
    std::vector<int>         replay_log_at_step; // how many log entries to show at each replay_step
    std::vector<std::vector<float>> heatmap_steps; // sidecar: [step][r*sz+c]
    std::vector<float>              current_heatmap; // what's displayed now ([sz*sz] or empty)
    bool show_heatmap    = false;
    int  heatmap_channel = -1;   // -1 = Grad-CAM, 0-31 = raw activation plane
    // Grad-CAM coprocess (spawned when --model is passed).
    FILE* cam_write     = nullptr;  // C++ → Python (commands)
    FILE* cam_read      = nullptr;  // Python → C++ (heatmaps)
    pid_t cam_pid       = -1;
    int   cam_hmap_step = -2;       // replay_step for which current_heatmap was last filled
    // Replay metadata needed to describe states to the coprocess.
    uint64_t replay_seed = 0;
    int      replay_sz   = 0;
    int replay_step = 0;
    // Previous frame's replay_step; single +1 advances animate, jumps don't.
    int last_replay_step = -1;
    // Legacy replays lack path_bits — recompute via BFS for the animator.
    auto populate_move_path = [&](Action& a, const GameState& pre) {
        if (a.type != ActionType::Move) return;
        int uid = pre.tile_at(a.from).unit_id();
        if (uid < 0) return;
        int8_t mp_at[MAX_MAP_TILES];
        int    parent[MAX_MAP_TILES];
        reachable_tiles(pre, uid, mp_at, parent);
        encode_path_bits(parent, a.from, a.to, pre.map_size(), &a.path_bits, &a.path_steps);
    };
    // First pass: find --model so we can spawn the coprocess before the window opens.
    std::string cam_ckpt_path;
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--model") { cam_ckpt_path = argv[i + 1]; break; }
    }

    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--replay") {
            replay_mode = true;
            std::ifstream f(argv[i + 1]);
            if (!f) { fprintf(stderr, "Cannot open replay: %s\n", argv[i + 1]); return 1; }
            GameState rs;
            std::string first;
            f >> first;
            auto push_replay_header = [&](int turn, int player) {
                char buf[64];
                snprintf(buf, sizeof(buf), "Turn %d Player %d", turn, player);
                replay_log.emplace_back(buf);
                replay_log_color.push_back(player_log_color(player));
            };
            if (first == "seed") {
                // Format: "seed <N> <sz>" — sz is optional, defaults to 9 for old files.
                std::string seed_line;
                std::getline(f, seed_line);
                std::istringstream iss(seed_line);
                uint64_t seed; iss >> seed;
                int sz = 9;
                iss >> sz;
                replay_seed = seed;
                replay_sz   = sz;
                MapGenParams p = MapGenParams::for_biome(BiomeType::Drylands, sz);
                p.seed = seed;
                rs = MapGen(p).generate().state;
                push_replay_header(rs.get_turn(), rs.current_player());
                // Skip optional outcome line (new format: "outcome <v_p0> <v_p1>").
                {
                    auto pos = f.tellg();
                    std::string tok; f >> tok;
                    if (tok == "outcome") { float v0, v1; f >> v0 >> v1; }
                    else f.seekg(pos);
                }
            } else {
                rs = MapGen(MapGen::drylands_defaults()).generate().state;
                push_replay_header(rs.get_turn(), rs.current_player());
                // first token was already consumed — push it back as action fields
                int t = std::stoi(first), fr, to, pa;
                f >> fr >> to >> pa;
                Action act0 = {(ActionType)t, fr, to, pa, true};
                populate_move_path(act0, rs);
                replay_log.push_back(format_action_str(act0, rs.current_player(), rs.get_turn(), rs.map_size()));
                replay_log_color.push_back(player_log_color(rs.current_player()));
                replay_actions.push_back(act0);
                replay_states.push_back(rs);
                replay_log_at_step.push_back(1);  // state 0: show only turn header
                replay_states.push_back(rs = rs.apply_action(act0));
                if (act0.type == ActionType::EndTurn)
                    push_replay_header(rs.get_turn(), rs.current_player());
                replay_log_at_step.push_back((int)replay_log.size());
            }
            replay_states.push_back(rs);
            replay_log_at_step.push_back((int)replay_log.size());  // step 0
            int t, fr, to, pa;
            while (f >> t >> fr >> to >> pa) {
                // New format appends path_bits path_steps — consume if present.
                { auto pos = f.tellg(); int pb, ps; if (!(f >> pb >> ps)) f.seekg(pos); }
                Action act = {(ActionType)t, fr, to, pa, true};
                populate_move_path(act, rs);
                replay_log.push_back(format_action_str(act, rs.current_player(), rs.get_turn(), rs.map_size()));
                replay_log_color.push_back(player_log_color(rs.current_player()));
                replay_actions.push_back(act);
                replay_states.push_back(rs = rs.apply_action(act));
                if (act.type == ActionType::EndTurn)
                    push_replay_header(rs.get_turn(), rs.current_player());
                replay_log_at_step.push_back((int)replay_log.size());
            }
            initial = s = replay_states[0];
            TILE_W = iso_grid_width() / s.map_size();
            TILE_H = (TILE_W * 3) / 5;
            // Try to load .heatmap sidecar (same path, different extension).
            {
                std::string hp = argv[i + 1];
                auto dot = hp.rfind('.');
                if (dot != std::string::npos) hp = hp.substr(0, dot);
                hp += ".heatmap";
                std::ifstream hf(hp, std::ios::binary);
                if (hf) {
                    int32_t hn, hsz;
                    hf.read(reinterpret_cast<char*>(&hn),  4);
                    hf.read(reinterpret_cast<char*>(&hsz), 4);
                    heatmap_steps.resize(hn, std::vector<float>(hsz * hsz));
                    for (int hi = 0; hi < hn; hi++)
                        hf.read(reinterpret_cast<char*>(heatmap_steps[hi].data()),
                                hsz * hsz * sizeof(float));
                    fprintf(stderr, "heatmap: loaded %d frames from %s\n", hn, hp.c_str());
                }
            }
            break;
        }
    }

    // Spawn Grad-CAM coprocess if --model was supplied.
    if (!cam_ckpt_path.empty()) {
        int pipe_to_py[2], pipe_from_py[2];
        if (pipe(pipe_to_py) == 0 && pipe(pipe_from_py) == 0) {
            cam_pid = fork();
            if (cam_pid == 0) {
                dup2(pipe_to_py[0],   STDIN_FILENO);
                dup2(pipe_from_py[1], STDOUT_FILENO);
                close(pipe_to_py[1]);  close(pipe_from_py[0]);
                close(pipe_to_py[0]);  close(pipe_from_py[1]);
                execlp("python3", "python3", "ai/tdl/gradcam_server.py",
                       "--ckpt", cam_ckpt_path.c_str(), nullptr);
                _exit(1);
            }
            close(pipe_to_py[0]);  close(pipe_from_py[1]);
            cam_write = fdopen(pipe_to_py[1],  "w");
            cam_read  = fdopen(pipe_from_py[0], "r");
            heatmap_steps.clear();  // coprocess supersedes sidecar
            fprintf(stderr, "gradcam: coprocess spawned (pid %d)\n", (int)cam_pid);
        }
    }

    // Helper: fill current_heatmap for replay_step, using coprocess or sidecar.
    auto refresh_heatmap = [&]() {
        if (!show_heatmap) return;
        if (cam_hmap_step == replay_step) return;  // already current
        if (cam_write && cam_read && replay_mode && replay_seed != 0) {
            // Send state description to coprocess.
            int n = replay_step;
            fprintf(cam_write, "%llu %d %d",
                    (unsigned long long)replay_seed, replay_sz, n);
            for (int k = 0; k < n; k++) {
                const Action& a = replay_actions[k];
                fprintf(cam_write, " %d %d %d %d %d %d",
                        (int)a.type, a.from, a.to, a.param,
                        (int)a.path_bits, (int)a.path_steps);
            }
            fprintf(cam_write, " %d\n", heatmap_channel);
            fflush(cam_write);
            int sz2 = replay_sz * replay_sz;
            current_heatmap.resize(sz2);
            for (int k = 0; k < sz2; k++)
                fscanf(cam_read, "%f", &current_heatmap[k]);
        } else if (!heatmap_steps.empty() && replay_step < (int)heatmap_steps.size()) {
            current_heatmap = heatmap_steps[replay_step];
        } else {
            current_heatmap.clear();
        }
        cam_hmap_step = replay_step;
    };

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

    // Initial turn header. Replay mode embeds its own headers in replay_log,
    // so skip here to avoid a duplicate flowing through Logger.
    if (!replay_mode) log_turn_header(s.get_turn(), s.current_player());

    auto refresh_borders = [&]() { init_colourers(); };

    int selected_tile = -1;

    Action actions[256];
    int    action_count   = 0;
    int    sidebar_scroll = 0;
    bool   tech_open      = false;   // sticky toggle from clicking the icon
    s.legal_actions(actions, action_count);

    // do_load / do_delete live here so they can capture the post-setup state
    // (init_colourers, reset_facing, actions[], etc.) by reference. Both keep
    // the game in a consistent state on failure — `s` is only mutated after
    // deserialise has returned a complete state.
    auto do_load = [&](const std::string& filename) {
        namespace fs = std::filesystem;
        fs::path path = fs::path("data/saved_gamestates") / filename;
        std::ifstream f(path);
        if (!f) {
            Logger::print("Load failed: cannot open %s", path.string().c_str());
            return;
        }
        nlohmann::json j;
        try {
            f >> j;
        } catch (const std::exception& ex) {
            Logger::print("Load failed: parse error in %s (%s)",
                          path.string().c_str(), ex.what());
            return;
        }
        GameState loaded;
        try {
            loaded = GameState::deserialise(j);
        } catch (const std::exception& ex) {
            Logger::print("Load failed: deserialise error in %s (%s)",
                          path.string().c_str(), ex.what());
            return;
        }
        s = loaded;
        initial = s;
        TILE_W = iso_grid_width() / s.map_size();
        TILE_H = (TILE_W * 3) / 5;
        reset_facing();
        init_colourers();
        s.legal_actions(actions, action_count);
        sidebar_scroll = 0;
        selected_tile  = -1;
        log_scroll     = 0;
        clear_visuals();
        load_open = false;
        Logger::print("Loaded game from %s", path.string().c_str());
        log_turn_header(s.get_turn(), s.current_player());
        // Reset the heuristic eval bar — the cached value belongs to the
        // previous state. If EVAL is on, refresh against the loaded state.
        eval_has_value = false;
        refresh_eval(true);
    };

    auto do_delete = [&](const std::string& filename) {
        namespace fs = std::filesystem;
        fs::path path = fs::path("data/saved_gamestates") / filename;
        std::error_code ec;
        if (!fs::remove(path, ec) || ec) {
            Logger::print("Delete failed: %s (%s)",
                          path.string().c_str(), ec.message().c_str());
            return;
        }
        Logger::print("Deleted %s", path.string().c_str());
        rescan_saves();
    };

    // HiDPI: render at physical pixel density so Retina doesn't upscale and blur.
    SetConfigFlags(FLAG_WINDOW_HIGHDPI);
    InitWindow(W, H, "Polyshark");
    SetTargetFPS(60);

    // Load San Francisco at a large atlas size; raylib scales down per-draw.
    g_font = LoadFontEx("/System/Library/Fonts/SFNS.ttf", 64, nullptr, 0);
    if (g_font.texture.id != 0)
        SetTextureFilter(g_font.texture, TEXTURE_FILTER_BILINEAR);
    else
        fprintf(stderr, "Failed to load SFNS.ttf; using raylib default font.\n");

    // Unit sprites [owner][UnitType]. Missing files fall back to circle+letter.
    Texture2D unit_tex[2][(int)UnitType::Count] = {};
    // Figure's alpha-bbox center offset from image center (centers padded sprites).
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
    load_sprite(0, UnitType::Giant,    "visualizer/sprites/units/blue/giant_blue.png");
    load_sprite(1, UnitType::Warrior,  "visualizer/sprites/units/red/warrior_red.png");
    load_sprite(1, UnitType::Archer,   "visualizer/sprites/units/red/archer_red.png");
    load_sprite(1, UnitType::Rider,    "visualizer/sprites/units/red/rider_red.png");
    load_sprite(1, UnitType::Defender, "visualizer/sprites/units/red/defender_red.png");
    load_sprite(1, UnitType::Giant,    "visualizer/sprites/units/red/giant_red.png");

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

    // Same loader pattern as terrain.
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
    load_resource(ResourceSprite::Forest,     "visualizer/sprites/terrain/imperius/Imperius_forest.png");
    load_resource(ResourceSprite::Village,    "visualizer/sprites/terrain/village.png");
    load_resource(ResourceSprite::Lighthouse, "visualizer/sprites/terrain/lighthouse.png");

    auto load_city = [&](int level, const char* path) {
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
            city_tex [level - 1] = t;
            city_bbox[level - 1] = { min_x, max_x + 1, min_y, max_y + 1 };
        }
    };
    for (int lvl = 1; lvl <= CITY_SPRITE_LEVELS; lvl++) {
        char path[64];
        snprintf(path, sizeof(path), "visualizer/sprites/cities/lvl%d.png", lvl);
        load_city(lvl, path);
    }

    // Star icon — used by top HUD and per-city SPT labels.
    {
        const char* path = "visualizer/sprites/star.png";
        if (!FileExists(path)) {
            fprintf(stderr, "missing sprite: %s\n", path);
        } else if (Image img = LoadImage(path); img.data != nullptr) {
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
                g_star_tex  = t;
                g_star_bbox = { min_x, max_x + 1, min_y, max_y + 1 };
            }
        }
    }

    while (!WindowShouldClose()) {

        // Global keyboard shortcuts — suppressed while the save textbox or
        // the load panel is open so typing/clicking through them doesn't
        // also fire global actions.
        bool kb_blocked = save_input_active || load_open;
        if (!kb_blocked && IsKeyPressed(KEY_TAB)) {
            view = (view == ViewMode::Omni)    ? ViewMode::P0      :
                   (view == ViewMode::P0)      ? ViewMode::P1      :
                   (view == ViewMode::P1)      ? ViewMode::Current : ViewMode::Omni;
        }

        if (!kb_blocked && IsKeyPressed(KEY_F)) g_use_custom_font = !g_use_custom_font;

        if (!kb_blocked && IsKeyPressed(KEY_P)) GameState::print(s);
      
        if (!save_input_active && IsKeyPressed(KEY_H) &&
                (!heatmap_steps.empty() || cam_write)) {
            show_heatmap = !show_heatmap;
            cam_hmap_step = -2;
            refresh_heatmap();
        }
        if (show_heatmap && cam_write) {
            int delta = IsKeyPressed(KEY_RIGHT_BRACKET) ? 1 :
                        IsKeyPressed(KEY_LEFT_BRACKET)  ? -1 : 0;
            if (delta) {
                heatmap_channel = heatmap_channel + delta;
                if (heatmap_channel > 31) heatmap_channel = -1;
                if (heatmap_channel < -1) heatmap_channel = 31;
                cam_hmap_step = -2;
                refresh_heatmap();
            }
        }

        if (!save_input_active && IsKeyPressed(KEY_P)) GameState::print(s);

        // Build sidebar layout (headers + action rows)
        static SidebarRow layout[512];
        int layout_count = build_sidebar_layout(actions, action_count, layout, 512);

        // Compute hovered action before drawing so map tiles can react to it
        Vector2 mouse   = GetMousePosition();
        bool    clicked = IsMouseButtonPressed(MOUSE_LEFT_BUTTON);

        // When the LOAD panel is open it should behave like a modal: every
        // underlying handler (sidebar actions, map tiles, REGEN/RESET/SAVE,
        // EVAL, CLEAR-log) must see no click this frame, otherwise a click
        // on a panel row also fires the sidebar action drawn behind it.
        // The panel and the LOAD toggle button use `clicked_panel` instead
        // of `clicked` so they remain interactive.
        bool clicked_panel = clicked;
        if (load_open) clicked = false;

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
        const int scrub_y = CONTENT_H;  // in the TECH_H strip, below all content buttons
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
                    refresh_heatmap();
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

        // Map click: applies Move/Attack on a highlighted dest, else toggles selection.
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
                    // Keep selection on the moved unit for chained moves.
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

        // Single forward step animates; jumps/rewinds cancel.
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

        // Fog/view selection — also feeds compute_border_flags so edges that
        // are completely in fog aren't drawn.
        int vp = (view == ViewMode::P1)      ? 1 :
                 (view == ViewMode::Current) ? s.current_player() : 0;
        bool omni = (view == ViewMode::Omni);
        const GameState& border_fog_state =
            move_anim.active ? move_anim.pre_state : s;
        compute_border_flags(s, border_fog_state, vp, omni);

        BeginDrawing();
        ClearBackground(BLACK);

        // --- Map (isometric diamond grid) ---
        // Two back-to-front passes (DESCENDING r+c).
        //   Pass 1 per tile: terrain → borders → rings → overlays → hover.
        //   Pass 2: units + HP. Units always render on top so they're never
        //   occluded by tall sprites (mountains/cities) on tiles in front.

        const Color FOG_COL   = {  18,  18,  18, 255 };
        const Color WATER_COL = {  30,  80, 130, 255 };
        for (int sum = 2 * (cur_msz - 1); sum >= 0; sum--) {
            for (int r = 0; r < cur_msz; r++) {
                int c = sum - r;
                if (c < 0 || c >= cur_msz) continue;
                int idx = to_index(c, r, cur_msz);
                // Ranged in flight: target tile uses pre-attack snapshot until impact.
                const GameState& render_state =
                    (ranged_anim.active && idx == ranged_anim.to_tile)
                        ? ranged_anim.pre_state : s;
                const Tile& t = render_state.tile_at(idx);
                // Defer fog reveal until anim ends — freshly walked tiles stay dark.
                const GameState& fog_state =
                    move_anim.active ? move_anim.pre_state : s;
                bool fogged = (view != ViewMode::Omni) && !fog_state.is_visible(vp, idx);
                Vector2 ctr = tile_center(r, c, cur_msz);

                // Terrain (or fog/water) first inside the tile. Fogged tiles
                // still render their border stripes on top of the fog diamond
                // so territory shape stays readable through fog.
                if (fogged) {
                    draw_diamond(ctr, FOG_COL);
                    int self_border_owner_fog = -1;
                    if (t.border_city_id() >= 0)
                        self_border_owner_fog = render_state.get_city(t.border_city_id()).owner();
                    draw_tile_borders(idx, ctr, self_border_owner_fog);
                    continue;
                }
                TerrainType ter = t.terrain();
                if (ter == TerrainType::Water) {
                    draw_diamond(ctr, WATER_COL);
                } else if (ter != TerrainType::Mountain) {
                    draw_terrain_sprite(ctr, TerrainSprite::Grass);
                }
                // Mountain sprite drawn AFTER borders so it occludes the
                // upper-back borders that pass behind it.

                int self_border_owner = -1;
                if (t.border_city_id() >= 0)
                    self_border_owner = render_state.get_city(t.border_city_id()).owner();
                // For mountain tiles: TL/TR drawn first (mountain occludes them),
                // mountain sprite, then BR/BL drawn on top so the front-facing
                // borders stay visible even when there's no neighbour in front.
                // Non-mountain tiles draw all 4 in one pass.
                if (ter == TerrainType::Mountain) {
                    draw_tile_borders(idx, ctr, self_border_owner, BORDER_DIRS_BACK);
                    draw_terrain_sprite(ctr, TerrainSprite::Mountain);
                    draw_tile_borders(idx, ctr, self_border_owner, BORDER_DIRS_FRONT);
                } else {
                    draw_tile_borders(idx, ctr, self_border_owner);
                }

                // Grad-CAM / activation plane heatmap overlay.
                if (show_heatmap && idx < (int)current_heatmap.size()) {
                    float heat = current_heatmap[idx];
                    if (heat > 0.02f) {
                        uint8_t a = (uint8_t)(heat * 170.0f);
                        uint8_t g = (uint8_t)(200.0f * (1.0f - heat));
                        draw_diamond(ctr, { 255, g, 0, a });
                    }
                }

                // Overlays between terrain and unit so units occlude them.
                if (!fogged) {
                    if (t.terrain() == TerrainType::Forest)
                        draw_resource_sprite(ctr, ResourceSprite::Forest);
                    else if (t.terrain() == TerrainType::Village && !t.has_city())
                        draw_resource_sprite(ctr, ResourceSprite::Village);
                    if (is_lighthouse(idx, cur_msz))
                        draw_resource_sprite(ctr, ResourceSprite::Lighthouse);
                    ResourceSprite rs = resource_type_to_sprite(t.resource());
                    if (rs != ResourceSprite::Count)
                        draw_resource_sprite(ctr, rs);
                    if (t.has_city()) {
                        const City& c = render_state.get_city(t.city_id());
                        draw_city_sprite(ctr, c.level());
                    }
                }

                // Move-dest: cyan ring. Attack-dest: same ring in red.
                // Drawn after city sprite so the ring sits in front of cities.
                if (is_move_dest(idx)) {
                    draw_movement_ring(ctr, { 80, 220, 255, 130 });
                } else if (is_attack_dest(idx)) {
                    draw_movement_ring(ctr, { 230, 80, 80, 130 });
                }

                // Hover highlight on top of terrain/overlays. Units draw in the
                // second pass below, so a unit on a hovered tile still shows.
                if (is_highlighted(idx)) {
                    draw_diamond(ctr, { 255, 255, 255, 40 });
                    draw_diamond_outline(ctr, { 255, 255, 80, 220 }, 3.0f);
                }
            }
        }

        // Pop-bar pass: runs after all terrain so a neighbouring mountain or
        // tall city sprite never overdraws a city's bar. Drawn before units
        // so a unit on a city tile still appears on top.
        for (int i = 0; i < cur_msz * cur_msz; i++) {
            const GameState& render_state_pb =
                (ranged_anim.active && i == ranged_anim.to_tile)
                    ? ranged_anim.pre_state : s;
            const Tile& tt = render_state_pb.tile_at(i);
            if (!tt.has_city()) continue;
            const GameState& fog_state_pb =
                move_anim.active ? move_anim.pre_state : s;
            bool fogged_pb = (view != ViewMode::Omni) && !fog_state_pb.is_visible(vp, i);
            if (fogged_pb) continue;
            int r_pb = i / cur_msz, c_pb = i % cur_msz;
            Vector2 ctr_pb = tile_center(r_pb, c_pb, cur_msz);
            const City& c = render_state_pb.get_city(tt.city_id());
            draw_pop_bar(ctr_pb, c.population(), c.level() + 1,
                         c.units_owned(), c.owner());
        }

        // Pass 2: units + HP. Back-to-front again so units sort against each
        // other, but every unit draws after all terrain/city sprites.
        for (int sum = 2 * (cur_msz - 1); sum >= 0; sum--) {
            for (int r = 0; r < cur_msz; r++) {
                int c = sum - r;
                if (c < 0 || c >= cur_msz) continue;
                int idx = to_index(c, r, cur_msz);
                const GameState& render_state =
                    (ranged_anim.active && idx == ranged_anim.to_tile)
                        ? ranged_anim.pre_state : s;
                const Tile& t = render_state.tile_at(idx);
                if (!t.has_unit()) continue;
                const GameState& fog_state =
                    move_anim.active ? move_anim.pre_state : s;
                bool fogged = (view != ViewMode::Omni) && !fog_state.is_visible(vp, idx);
                if (fogged) continue;
                const Unit& u = render_state.get_unit(t.unit_id());
                bool show = (view == ViewMode::Omni)
                         || s.is_visible(vp, idx)
                         || u.owner() == vp;
                if (!show) continue;
                Vector2 ctr = tile_center(r, c, cur_msz);
                float cxf = ctr.x;
                float cyf = ctr.y;
                // Lerp draw pos along animated path instead of post-apply tile.
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
                            // Triangle wave 0→1→0, scaled by ATTACK_LUNGE_FRAC.
                            double tri = 1.0 - fabs(2.0 * p - 1.0);
                            double offset = tri * (double)ATTACK_LUNGE_FRAC;
                            Vector2 fp = tile_center(ft / cur_msz, ft % cur_msz, cur_msz);
                            Vector2 tp = tile_center(tt / cur_msz, tt % cur_msz, cur_msz);
                            cxf = (float)(fp.x + (tp.x - fp.x) * offset);
                            cyf = (float)(fp.y + (tp.y - fp.y) * offset);
                        } else {
                            // Smoothstep (3p² - 2p³) across the whole path.
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
                // "Ready" ring: scans legal actions to catch mid-turn re-readies.
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

                // Scale off TILE_W so figures don't squish on the shorter iso tile.
                const Texture2D& tex = unit_tex[u.owner()][(int)u.type()];
                if (tex.id != 0) {
                    const Transformer& xf = UNIT_TRANSFORMERS[(int)u.type()];
                    float scale = (TILE_W * 0.95f) / (float)tex.height * xf.scale;
                    float w = tex.width  * scale;
                    float h = tex.height * scale;
                    float xoff = unit_tex_xoff[u.owner()][(int)u.type()] * scale;
                    float yoff = unit_tex_yoff[u.owner()][(int)u.type()] * scale;
                    float xtweak = xf.x_offset * (float)TILE_W;
                    float ytweak = xf.y_offset * (float)TILE_H;
                    // Negative src_w mirrors; invert x-offsets to keep centring after mirror.
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
                    static const char* unit_icon[] = { "?", "W", "A", "R", "D", "G" };
                    const char* icon = unit_icon[(int)u.type()];
                    DrawTextC(icon, (int)cxf - MeasureTextC(icon, 9) / 2, (int)cyf - 4, 9, WHITE);
                }
                // HP number above diamond top; placement via HP_TRANSFORMER.
                const char* hp_str = TextFormat("%d", u.hp());
                int hp_fs = (int)(11.0f * HP_TRANSFORMER.scale + 0.5f);
                if (hp_fs < 1) hp_fs = 1;
                int hp_w = MeasureTextC(hp_str, hp_fs);
                int hp_x = (int)cxf - hp_w / 2
                         + (int)(HP_TRANSFORMER.x_offset * TILE_W);
                int hp_y = (int)(cyf - TILE_H * 0.5f) - 12
                         + (int)(HP_TRANSFORMER.y_offset * TILE_H);
                DrawTextC(hp_str, hp_x + 1, hp_y + 1, hp_fs, BLACK);
                DrawTextC(hp_str, hp_x,     hp_y,     hp_fs, WHITE);

                // Defense shield(s) to the right of HP. count = (bonus-1.0)/0.5,
                // so 1.5x = 1 shield, 2.0x (walls + tech-defended terrain) = 2.
                float def_bonus = defense_bonus_for_render(render_state, t);
                int shields = (int)((def_bonus - 1.0f) / 0.5f + 0.001f);
                if (shields > 0) {
                    // Pen advances horizontally; each shield is positioned by
                    // its own transformer so the 2nd can be tuned independently.
                    float pen_x = (float)(hp_x + hp_w) + 2.0f;
                    for (int si = 0; si < shields; si++) {
                        const Transformer& sxf = (si == 1)
                            ? SECOND_SHIELD_TRANSFORMER
                            : SHIELD_TRANSFORMER;
                        float s_w = sxf.scale;
                        float gap = s_w * 0.25f;
                        float sx  = pen_x + sxf.x_offset * (float)TILE_W + s_w * 0.5f;
                        float sy  = (float)hp_y + (float)hp_fs * 0.5f
                                  + sxf.y_offset * (float)TILE_H;
                        // Text-shadow look: black drop-shadow at +1/+1, white on top.
                        draw_shield(sx + 1.0f, sy + 1.0f, s_w, BLACK);
                        draw_shield(sx,        sy,        s_w, WHITE);
                        pen_x += s_w + gap;
                    }
                }
            }
        }

        // City-label pass: blue banner + name + SPT. Runs AFTER units so it is
        // the only thing layered in front of unit sprites; mirrors the pop-bar
        // pass's fog/anim handling so labels and pop bars stay in sync.
        for (int i = 0; i < cur_msz * cur_msz; i++) {
            const GameState& render_state_cl =
                (ranged_anim.active && i == ranged_anim.to_tile)
                    ? ranged_anim.pre_state : s;
            const Tile& tt = render_state_cl.tile_at(i);
            if (!tt.has_city()) continue;
            const GameState& fog_state_cl =
                move_anim.active ? move_anim.pre_state : s;
            bool fogged_cl = (view != ViewMode::Omni) && !fog_state_cl.is_visible(vp, i);
            if (fogged_cl) continue;
            int r_cl = i / cur_msz, c_cl = i % cur_msz;
            Vector2 ctr_cl = tile_center(r_cl, c_cl, cur_msz);
            const City& c = render_state_cl.get_city(tt.city_id());
            draw_city_label(ctr_cl, city_name_for(c.tile_index()), c.stars_per_turn());
        }

        // Axis labels: bottom-right edge labels columns (r=0), bottom-left labels rows (c=0).
        {
            constexpr int AXIS_FONT_SZ = 11;
            const Color AXIS_TEXT   = { 150, 150, 150, 255 };
            const Color AXIS_SHADOW = {  20,  20,  20, 160 };
            for (int c = 0; c < cur_msz; c++) {
                Vector2 ctr = tile_center(0, c, cur_msz);
                char buf[8]; snprintf(buf, sizeof(buf), "%d", c);
                int x = (int)(ctr.x + TILE_W * 0.32f);
                int y = (int)(ctr.y + TILE_H * 0.35f);
                DrawTextC(buf, x + 1, y + 1, AXIS_FONT_SZ, AXIS_SHADOW);
                DrawTextC(buf, x,     y,     AXIS_FONT_SZ, AXIS_TEXT);
            }
            for (int r = 0; r < cur_msz; r++) {
                Vector2 ctr = tile_center(r, 0, cur_msz);
                char buf[8]; snprintf(buf, sizeof(buf), "%d", r);
                int tw = MeasureTextC(buf, AXIS_FONT_SZ);
                int x = (int)(ctr.x - TILE_W * 0.32f) - tw;
                int y = (int)(ctr.y + TILE_H * 0.35f);
                DrawTextC(buf, x + 1, y + 1, AXIS_FONT_SZ, AXIS_SHADOW);
                DrawTextC(buf, x,     y,     AXIS_FONT_SZ, AXIS_TEXT);
            }
        }

        // Ranged projectile arc; deactivates on land so next frame shows live state.
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
                // Parabolic arc, peak at p=0.5; height scales with travel distance.
                float dx = tp.x - fp.x, dy = tp.y - fp.y;
                float dist = sqrtf(dx * dx + dy * dy);
                float arc_h = dist * 0.18f;
                float ay = (float)(ly - arc_h * 4.0 * p * (1.0 - p));
                float r  = TILE_W * 0.045f;
                if (r < 2.0f) r = 2.0f;
                DrawCircle((int)lx, (int)ay, r, BLACK);
            }
        }

        // Tech tree moved further down the frame so it layers above the
        // heuristic eval bar — see the block just before "Action log panel".

        // --- Top HUD bar ---
        DrawRectangle(0, 0, W, TOP_HUD, PANEL_BG);
        DrawLine(0, TOP_HUD, W, TOP_HUD, PANEL_LINE);

        // Turn + active player (left side)
        DrawTextC(TextFormat("Turn %d", s.get_turn()),       PAD + 4, 8,  22, WHITE);
        DrawTextC(TextFormat("P%d to move", s.current_player()), PAD + 4, 34, 18, LIGHTGRAY);

        // Stars — large, player-coloured (centre). Suffix shows income per turn.
        int spt[2] = {0, 0};
        for (int i = 0; i < s.map_tiles(); i++) {
            const Tile& tt = s.tile_at(i);
            if (!tt.has_city()) continue;
            const City& cc = s.get_city(tt.city_id());
            int o = cc.owner();
            if (o == 0 || o == 1) spt[o] += cc.stars_per_turn();
        }
        const char* p0_str = TextFormat("P0  %d", s.get_stars(0));
        const char* p1_str = TextFormat("P1  %d", s.get_stars(1));
        const char* p0_spt = TextFormat("+%d", spt[0]);
        const char* p1_spt = TextFormat("+%d", spt[1]);
        constexpr int   STAR_GAP   = 28;
        constexpr int   MAP_CENTRE = MAP_OFF + MAP_PX / 2;
        // Star heights/pads in pixels; TOP_HUD_STAR_TRANSFORMER drives the big
        // stars on the main P0/P1 line, BOTTOM_HUD_STAR_TRANSFORMER drives the
        // small stars on the "+income" line below.
        constexpr float STAR_BIG_H_BASE = 22.0f;
        constexpr float STAR_SML_H_BASE = 11.0f;
        constexpr float STAR_PAD_BIG = 8.0f;
        constexpr float STAR_PAD_SML = 3.0f;
        const Transformer& tsx = TOP_HUD_STAR_TRANSFORMER;
        const Transformer& bsx = BOTTOM_HUD_STAR_TRANSFORMER;
        const float STAR_BIG_H = STAR_BIG_H_BASE * tsx.scale;
        const float STAR_SML_H = STAR_SML_H_BASE * bsx.scale;
        const float star_big_w = star_icon_width(STAR_BIG_H);
        const float star_sml_w = star_icon_width(STAR_SML_H);
        const int p0_text_w = MeasureTextC(p0_str, 28);
        const int p1_text_w = MeasureTextC(p1_str, 28);
        const int p0_w = p0_text_w + (int)(STAR_PAD_BIG + star_big_w);
        const int p1_w = p1_text_w + (int)(STAR_PAD_BIG + star_big_w);
        const int block = p0_w + STAR_GAP + p1_w;
        const int p0_x  = MAP_CENTRE - block / 2;
        const int p1_x  = p0_x + p0_w + STAR_GAP;
        DrawTextC(p0_str, p0_x, 6, 28, { 100, 160, 255, 255 });
        DrawTextC(p1_str, p1_x, 6, 28, { 255, 100, 100, 255 });
        const float big_cy = 6.0f + 28.0f * 0.5f + 2.0f + tsx.y_offset;
        draw_star_icon((float)p0_x + (float)p0_text_w + STAR_PAD_BIG + star_big_w * 0.5f + tsx.x_offset,
                       big_cy, STAR_BIG_H);
        draw_star_icon((float)p1_x + (float)p1_text_w + STAR_PAD_BIG + star_big_w * 0.5f + tsx.x_offset,
                       big_cy, STAR_BIG_H);

        // SPT subtext + small star, centred under each player's block.
        const int p0_spt_text_w = MeasureTextC(p0_spt, 14);
        const int p1_spt_text_w = MeasureTextC(p1_spt, 14);
        const float p0_spt_full_w = (float)p0_spt_text_w + STAR_PAD_SML + star_sml_w;
        const float p1_spt_full_w = (float)p1_spt_text_w + STAR_PAD_SML + star_sml_w;
        const float p0_spt_x = (float)p0_x + (float)p0_w * 0.5f - p0_spt_full_w * 0.5f;
        const float p1_spt_x = (float)p1_x + (float)p1_w * 0.5f - p1_spt_full_w * 0.5f;
        DrawTextC(p0_spt, (int)p0_spt_x, 38, 14, { 100, 160, 255, 200 });
        DrawTextC(p1_spt, (int)p1_spt_x, 38, 14, { 255, 100, 100, 200 });
        const float sml_cy = 38.0f + 14.0f * 0.5f + 1.0f + bsx.y_offset;
        draw_star_icon(p0_spt_x + (float)p0_spt_text_w + STAR_PAD_SML + star_sml_w * 0.5f + bsx.x_offset,
                       sml_cy, STAR_SML_H);
        draw_star_icon(p1_spt_x + (float)p1_spt_text_w + STAR_PAD_SML + star_sml_w * 0.5f + bsx.x_offset,
                       sml_cy, STAR_SML_H);

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
            int tw = MeasureTextC(vname, font_sz);
            int right_edge = MAP_OFF + MAP_PX + SIDEBAR;  // left edge of log panel
            int bx = right_edge - tw - 24, by = 10, bw = tw + 16, bh = 28;
            DrawRectangleRounded({ (float)bx, (float)by, (float)bw, (float)bh }, 0.4f, 6, badge_col);
            DrawTextC(vname, bx + 8, by + 5, font_sz, WHITE);
            DrawTextC("[Tab]", right_edge - MeasureTextC("[Tab]", 13) - 8, by + bh + 4, 13, GRAY);

            if (replay_mode) {
                const char* rtxt = TextFormat("REPLAY  %d / %d", replay_step, (int)replay_states.size() - 1);
                int rtw = MeasureTextC(rtxt, 16);
                DrawRectangleRounded({ (float)(bx - rtw - 28), (float)by, (float)(rtw + 16), (float)bh }, 0.4f, 6, Color{ 40, 120, 80, 255 });
                DrawTextC(rtxt, bx - rtw - 20, by + 5, 16, WHITE);
                DrawTextC("[← →] step   [R] reset", bx - rtw - 20, by + bh + 4, 11, GRAY);
            }
        }

        // --- Sidebar ---
        int SB = MAP_OFF + MAP_PX;  // sidebar left edge
        DrawRectangle(SB, TOP_HUD, SIDEBAR, CONTENT_H - TOP_HUD, SIDEBAR_BG);
        DrawLine(SB, TOP_HUD, SB, CONTENT_H, PANEL_LINE);
        DrawTextC("LEGAL ACTIONS", SB + 8, TOP_HUD + 10, 16, GRAY);
        DrawLine(SB + 4, TOP_HUD + 30, SB + SIDEBAR - 4, TOP_HUD + 30, { 60, 60, 60, 255 });

        int  applied    = click_applied;  // map-click Move/Attack takes precedence; sidebar can also set this
        bool reset      = false;
        bool regenerate = false;

        // Spacebar = End Turn (only in live play; in replay mode Space steps forward)
        if (!replay_mode && !save_input_active && !load_open && IsKeyPressed(KEY_SPACE)) {
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
                    refresh_heatmap();
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
                DrawTextC("End Turn", SB + 8, ry + 2, 12, hov ? WHITE : Color{ 160, 200, 160, 255 });
                if (hov && clicked) applied = lr.action_idx;
            } else if (lr.is_header) {
                DrawRectangle(SB + 4, ry, SIDEBAR - 8, HEADER_H - 2, { 25, 25, 25, 255 });
                DrawTextC(category_name(lr.category), SB + 8, ry + 2, 12, { 160, 160, 160, 255 });
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
                DrawTextC(label, SB + 10, ry + 5, 13, text_col);
                if (cost[0]) {
                    int cw = MeasureTextC(cost, 12);
                    DrawTextC(cost, SB + SIDEBAR - 10 - cw, ry + 6, 12, cost_col);
                }

                if (hovered && clicked && afford)
                    applied = lr.action_idx;
            }
        }
        EndScissorMode();

        // --- TOGGLE EVAL button (half-size, bottom-right of the map area).
        //     Toggles the heuristic-value bar at the bottom of the map AND
        //     auto-logging on every applied action. ---
        if (!replay_mode) {
            constexpr int EVAL_W = (SIDEBAR)/2;       // half the old sidebar-width
            constexpr int EVAL_H = 20;                      // half the old height
            int eval_btn_x = iso_grid_right() - 10 - EVAL_W;
            int eval_btn_y = CONTENT_H - 10 - EVAL_H;
            Rectangle eval_btn = { (float)eval_btn_x, (float)eval_btn_y,
                                   (float)EVAL_W, (float)EVAL_H };
            bool eval_enabled  = python_ok && !save_input_active;
            bool eval_hov      = eval_enabled && CheckCollisionPointRec(mouse, eval_btn);
            // ON state is brighter; hover lightens; disabled is dim.
            Color eval_bg = !eval_enabled ? Color{ 35, 35, 35, 255 }
                          : eval_on       ? (eval_hov ? Color{ 130, 90, 180, 255 } : Color{ 100, 70, 150, 255 })
                                          : (eval_hov ? Color{ 90,  60, 130, 255 } : Color{ 60,  40,  90, 255 });
            Color eval_fg = eval_enabled  ? WHITE : Color{ 110, 110, 110, 255 };
            DrawRectangleRec(eval_btn, eval_bg);
            const char* label = "TOGGLE EVAL";
            int etw = MeasureTextC(label, 11);
            DrawTextC(label, eval_btn_x + EVAL_W / 2 - etw / 2,
                      eval_btn_y + (EVAL_H - 11) / 2, 11, eval_fg);
            if (eval_hov && clicked) {
                eval_on = !eval_on;
                if (eval_on) refresh_eval(true);       // compute + log on enable
                else         eval_has_value = false;   // hide stale slider on disable
            }
        }

        // --- Save / Load button row (half-width each, full height, above
        //     regen with the same 8px gap that separates regen and reset).
        //     Clicking SAVE flips the slot into a textbox; clicking LOAD
        //     toggles an overlay panel listing saved games. ---
        if (!replay_mode) {
            constexpr int BTN_H = 28;
            int row_y    = CONTENT_H - 72 - 8 - BTN_H;            // 8px gap above regen
            int half_w   = (SIDEBAR - 12) / 2;                    // 4px outer + 4px center
            int save_x   = SB + 4;
            int load_x   = save_x + half_w + 4;
            Rectangle save_btn      = { (float)save_x, (float)row_y, (float)half_w, (float)BTN_H };
            Rectangle load_btn_rect = { (float)load_x, (float)row_y, (float)half_w, (float)BTN_H };
            if (save_input_active) {
                Color border = save_filename_invalid
                             ? Color{ 200, 60, 60, 255 }
                             : Color{ 120, 120, 120, 255 };
                DrawRectangleRec(save_btn, Color{ 28, 28, 28, 255 });
                DrawRectangleLinesEx(save_btn, 1, border);
                int caret_blink = ((int)(GetTime() * 2.0) & 1);
                std::string shown = std::string(save_filename, save_filename_len);
                if (caret_blink) shown += '|';
                DrawTextC(shown.c_str(), save_x + 4, row_y + (BTN_H - 11) / 2, 11, WHITE);

                // Text input
                int ch = GetCharPressed();
                while (ch > 0) {
                    if (save_filename_len < (int)sizeof(save_filename) - 1
                        && is_valid_filename_char((char)ch)) {
                        save_filename[save_filename_len++] = (char)ch;
                        save_filename[save_filename_len]   = 0;
                        save_filename_invalid = false;
                    }
                    ch = GetCharPressed();
                }
                if (IsKeyPressed(KEY_BACKSPACE) && save_filename_len > 0) {
                    save_filename[--save_filename_len] = 0;
                    save_filename_invalid = false;
                }
                if (IsKeyPressed(KEY_ENTER))  do_save();
                if (IsKeyPressed(KEY_ESCAPE)) close_save_input();
                // Click outside the textbox cancels.
                if (clicked && !CheckCollisionPointRec(mouse, save_btn))
                    close_save_input();
            } else {
                bool hovered = CheckCollisionPointRec(mouse, save_btn);
                DrawRectangleRec(save_btn, hovered ? Color{ 60, 100, 150, 255 } : Color{ 40, 70, 110, 255 });
                int tw = MeasureTextC("SAVE", 14);
                DrawTextC("SAVE", save_x + half_w / 2 - tw / 2, row_y + (BTN_H - 14) / 2, 14, WHITE);
                if (hovered && clicked) save_input_active = true;
            }

            // LOAD button (right half).
            bool load_hovered = CheckCollisionPointRec(mouse, load_btn_rect) && !save_input_active;
            DrawRectangleRec(load_btn_rect, load_hovered ? Color{ 130, 95, 50, 255 } : Color{ 95, 65, 30, 255 });
            int ltw = MeasureTextC("LOAD", 14);
            DrawTextC("LOAD", load_x + half_w / 2 - ltw / 2, row_y + (BTN_H - 14) / 2, 14, WHITE);
            // Use clicked_panel so the LOAD toggle keeps working while the
            // panel is open (the global `clicked` is suppressed in modal mode).
            if (load_hovered && clicked_panel) {
                load_open = !load_open;
                if (load_open) rescan_saves();
            }

            // Load overlay panel (anchored just above the EVAL button so it
            // doesn't cover it).
            if (load_open) {
                constexpr int PANEL_H = 200;
                constexpr int HDR_H   = 22;
                constexpr int ROW_H   = 22;
                int panel_x = SB + 4;
                int panel_w = SIDEBAR - 8;
                // EVAL no longer lives in the sidebar — anchor against the
                // SAVE/LOAD row directly.
                int panel_bottom = row_y - 4;
                int panel_top    = panel_bottom - PANEL_H;
                Rectangle panel = { (float)panel_x, (float)panel_top, (float)panel_w, (float)PANEL_H };
                DrawRectangleRec(panel, Color{ 24, 24, 24, 255 });
                DrawRectangleLinesEx(panel, 1, Color{ 70, 70, 70, 255 });
                DrawTextC("LOAD GAME", panel_x + 6, panel_top + 5, 12, GRAY);
                DrawLine(panel_x + 2, panel_top + HDR_H,
                         panel_x + panel_w - 2, panel_top + HDR_H, Color{ 50, 50, 50, 255 });

                int rows_top    = panel_top + HDR_H;
                int rows_h      = PANEL_H - HDR_H;
                int visible_rows = rows_h / ROW_H;
                int total       = (int)save_files.size();
                int max_scroll  = std::max(0, total - visible_rows);
                if (load_scroll > max_scroll) load_scroll = max_scroll;

                if (CheckCollisionPointRec(mouse, panel))
                    load_scroll = std::clamp(load_scroll - (int)GetMouseWheelMove(), 0, max_scroll);

                BeginScissorMode(panel_x, rows_top, panel_w, rows_h);
                if (total == 0) {
                    DrawTextC("(no saves)", panel_x + 6, rows_top + 5, 11, Color{ 110, 110, 110, 255 });
                }
                for (int i = 0; i < visible_rows && load_scroll + i < total; i++) {
                    int idx = load_scroll + i;
                    const std::string& name = save_files[idx];
                    int ry = rows_top + i * ROW_H;
                    if (i % 2 == 0)
                        DrawRectangle(panel_x + 1, ry, panel_w - 2, ROW_H - 1, Color{ 30, 30, 30, 255 });

                    // Display name without .json suffix.
                    std::string display = name;
                    if (display.size() > 5 && display.compare(display.size() - 5, 5, ".json") == 0)
                        display.resize(display.size() - 5);
                    DrawTextC(display.c_str(), panel_x + 6, ry + (ROW_H - 11) / 2, 11, WHITE);

                    // DELETE (rightmost), then LOAD button to its left.
                    int del_w = 18, ld_w = 38, gap = 4;
                    int del_bx = panel_x + panel_w - 4 - del_w;
                    int ld_bx  = del_bx - gap - ld_w;
                    Rectangle del_btn = { (float)del_bx, (float)(ry + 2), (float)del_w, (float)(ROW_H - 4) };
                    Rectangle ld_btn  = { (float)ld_bx,  (float)(ry + 2), (float)ld_w,  (float)(ROW_H - 4) };

                    bool ld_h  = CheckCollisionPointRec(mouse, ld_btn);
                    DrawRectangleRec(ld_btn, ld_h ? Color{ 60, 120, 80, 255 } : Color{ 30, 80, 50, 255 });
                    int tw = MeasureTextC("LOAD", 11);
                    DrawTextC("LOAD", ld_bx + ld_w / 2 - tw / 2, ry + (ROW_H - 11) / 2, 11, WHITE);

                    bool del_h = CheckCollisionPointRec(mouse, del_btn);
                    DrawRectangleRec(del_btn, del_h ? Color{ 180, 60, 60, 255 } : Color{ 100, 35, 35, 255 });
                    int xw = MeasureTextC("X", 11);
                    DrawTextC("X", del_bx + del_w / 2 - xw / 2, ry + (ROW_H - 11) / 2, 11, WHITE);

                    if (ld_h && clicked_panel)  { do_load(name);   break; }
                    if (del_h && clicked_panel) { do_delete(name); break; }
                }
                EndScissorMode();

                // Esc closes. (Click suppression is now handled by the
                // global modal guard at the start of the frame.)
                if (IsKeyPressed(KEY_ESCAPE)) load_open = false;
            }
        }

        // --- Regenerate button ---
        if (!replay_mode) {
            Rectangle regen_btn = { (float)SB + 4, (float)CONTENT_H - 72, (float)SIDEBAR - 8, 28 };
            bool hovered = CheckCollisionPointRec(mouse, regen_btn) && !save_input_active;
            DrawRectangleRec(regen_btn, hovered ? Color{ 30, 130, 80, 255 } : Color{ 20, 80, 50, 255 });
            int tw = MeasureTextC("REGEN MAP", 14);
            DrawTextC("REGEN MAP", SB + SIDEBAR / 2 - tw / 2, CONTENT_H - 64, 14, WHITE);
            if (hovered && clicked)
                regenerate = true;
        }

        // --- Reset button (bottom of sidebar) ---
        {
            Rectangle reset_btn = { (float)SB + 4, (float)CONTENT_H - 36, (float)SIDEBAR - 8, 28 };
            bool hovered = CheckCollisionPointRec(mouse, reset_btn) && !save_input_active;
            DrawRectangleRec(reset_btn, hovered ? Color{ 140, 50, 50, 255 } : Color{ 90, 30, 30, 255 });
            DrawTextC("RESET", SB + SIDEBAR / 2 - 24, CONTENT_H - 30, 18, WHITE);
            if (hovered && clicked)
                reset = true;
        }

        // Apply after the loop so we don't mutate actions[] mid-render
        if (regenerate) {
            clear_visuals();
            initial = new_map();
            TILE_W = iso_grid_width() / initial.map_size();
            TILE_H = (TILE_W * 3) / 5;
            s = initial;
            reset_facing();
            init_colourers();
            s.legal_actions(actions, action_count);
            sidebar_scroll = 0;
            selected_tile  = -1;
            Logger::print("Regenerated map.");
            log_turn_header(s.get_turn(), s.current_player());
            // Stale cached eval belongs to the previous map — clear it; if
            // EVAL is still on, recompute against the fresh state.
            eval_has_value = false;
            refresh_eval(true);
        } else if (reset) {
            clear_visuals();
            if (replay_mode) {
                replay_step = 0;
                s = replay_states[0];
            } else {
                s = initial;
                Logger::print("Reset game.");
                log_turn_header(s.get_turn(), s.current_player());
            }
            reset_facing();
            init_colourers();
            s.legal_actions(actions, action_count);
            sidebar_scroll = 0;
            selected_tile  = -1;
            log_scroll = 0;
            // Stale eval belongs to the pre-reset state.
            eval_has_value = false;
            refresh_eval(true);
        }
        if (applied >= 0 && !replay_mode) {
            Logger::print(player_log_color(s.current_player()), "%s",
                          format_action_str(actions[applied], s.current_player(),
                                            s.get_turn(), s.map_size()).c_str());

            // Wipe any leftover visual-only state before the new action.
            clear_visuals();

            // Explorer upgrade: simulate on copy first to record the walked path.
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

            // Facing flips left when (Δcol - Δrow) < 0 (screen-x decreasing).
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

            // Move only needs pre-state; Attack needs both to decide lunge vs advance.
            GameState pre_state = s;
            s = s.apply_action(actions[applied]);
            if (actions[applied].type == ActionType::Move) {
                start_move_anim(pre_state, actions[applied]);
            } else if (actions[applied].type == ActionType::Attack) {
                start_attack_anim(pre_state, s, actions[applied]);
            } else if (actions[applied].type == ActionType::EndTurn) {
                log_turn_header(s.get_turn(), s.current_player());
            }
            refresh_borders();
            // Auto-refresh the heuristic bar after every applied action.
            refresh_eval(true);
            s.legal_actions(actions, action_count);
            sidebar_scroll = 0;
        }

        // Sync log to current replay step
        if (replay_mode) {
            int n = (replay_step < (int)replay_log_at_step.size())
                    ? replay_log_at_step[replay_step]
                    : (int)replay_log.size();
            action_log.assign(replay_log.begin(), replay_log.begin() + n);
            action_log_color.assign(replay_log_color.begin(), replay_log_color.begin() + n);
        }

        // Pull any new Logger::print entries into the log panel with their
        // recorded colour (live action rows go through Logger::print too).
        while (last_logger_count < Logger::count) {
            int back_idx = Logger::count - 1 - last_logger_count;  // 0 = newest
            const char* msg = Logger::get(back_idx);
            if (msg) {
                action_log.emplace_back(msg);
                action_log_color.push_back(Logger::get_color(back_idx));
            }
            last_logger_count++;
        }


        // --- Selected tile outline ---
        if (selected_tile >= 0) {
            int sx, sy;
            to_coords(selected_tile, sx, sy);
            Vector2 sel_ctr = tile_center(sy, sx, cur_msz);
            draw_diamond_outline(sel_ctr, WHITE, 2.0f);
        }

        // --- Selected city territory highlight ---
        // Solid-filled temp borders along every outer-boundary edge of the
        // selected city's territory; disambiguates overlapping city borders.
        if (selected_tile >= 0 && s.tile_at(selected_tile).has_city()) {
            int sel_city = s.tile_at(selected_tile).city_id();
            int sel_owner = s.get_city(sel_city).owner();
            int msz_h = s.map_size();
            int mtsz_h = s.map_tiles();
            for (int i = 0; i < mtsz_h; i++) {
                if (s.tile_at(i).border_city_id() != sel_city) continue;
                int rr = i / msz_h, cc = i % msz_h;
                Vector2 ctr2 = tile_center(rr, cc, msz_h);
                Vector2 vtop2   = { ctr2.x,                  ctr2.y - TILE_H * 0.5f };
                Vector2 vright2 = { ctr2.x + TILE_W * 0.5f,  ctr2.y                 };
                Vector2 vbot2   = { ctr2.x,                  ctr2.y + TILE_H * 0.5f };
                Vector2 vleft2  = { ctr2.x - TILE_W * 0.5f,  ctr2.y                 };
                Vector2 edges2[4][2] = {
                    { vbot2,  vright2 },
                    { vleft2, vtop2   },
                    { vbot2,  vleft2  },
                    { vtop2,  vright2 },
                };
                for (int d = 0; d < 4; d++) {
                    int nr2 = rr + BORDER_DR[d], nc2 = cc + BORDER_DC[d];
                    int n_cid2 = -1;
                    if (nr2 >= 0 && nr2 < msz_h && nc2 >= 0 && nc2 < msz_h)
                        n_cid2 = s.tile_at(nr2 * msz_h + nc2).border_city_id();
                    if (n_cid2 == sel_city) continue;
                    draw_temp_border(edges2[d][0], edges2[d][1], sel_owner);
                }
            }
        }

        // --- Heuristic eval bar (bottom of map, shown only when EVAL is on).
        //     Drawn before the tech tree so the tech panel layers on top. ---
        if (eval_on && !replay_mode) {
            // Centre the bar on the iso grid's actual centre (the iso grid is
            // drawn around `iso_grid_right() / 2`, which doesn't coincide with
            // the middle of the MAP_PX region because of the LABEL gutter).
            const int bar_margin = 30;
            const int bar_w      = MAP_CANVAS - bar_margin * 2;
            const int bar_cx     = iso_grid_right() / 2;  // value 0
            const int bar_x      = bar_cx - bar_w / 2;
            const int bar_h      = 4;
            const int bar_y      = CONTENT_H - 28;

            Color minor_tick = { 130, 130, 130, 255 };
            Color zero_tick  = { 230, 230, 230, 255 };
            Color bar_fill   = { 40,  40,  40, 230 };
            Color bar_border = { 90,  90,  90, 255 };

            DrawRectangle(bar_x, bar_y, bar_w, bar_h, bar_fill);
            DrawRectangleLines(bar_x, bar_y, bar_w, bar_h, bar_border);

            // Ticks at every 10-unit step from -100 to +100. Big at 0, small elsewhere.
            for (int i = -10; i <= 10; i++) {
                int  tx = bar_x + (int)(((i + 10) / 20.0f) * bar_w);
                int  th = (i == 0) ? 16 : 6;
                int  ty = bar_y + bar_h / 2 - th / 2;
                Color tc = (i == 0) ? zero_tick : minor_tick;
                DrawRectangle(tx, ty, 1, th, tc);
            }

            // End labels and "0" label.
            DrawTextC("0",    bar_cx - 3,        bar_y + bar_h + 6, 10, zero_tick);
            DrawTextC("-100", bar_x - 26,        bar_y - 4,         10, Color{ 200, 90,  90, 255 });
            DrawTextC("+100", bar_x + bar_w + 4, bar_y - 4,         10, Color{ 90,  200, 90, 255 });

            // Slider — vertical bar through the strip plus a small triangle
            // pointing down to it for legibility at the exact position.
            if (eval_has_value) {
                double clamped = eval_value < -100.0 ? -100.0
                               : eval_value >  100.0 ?  100.0 : eval_value;
                int sx = bar_x + (int)((clamped + 100.0) / 200.0 * bar_w);
                Color slider_col = (eval_perspective == 0) ? COL_P0 : COL_P1;

                // Vertical marker bar — 3px wide, extends 8px above and below
                // the scale so it stands out against the tick marks.
                int marker_h = bar_h + 16;
                DrawRectangle(sx - 1, bar_y - 8, 3, marker_h, slider_col);

                // Triangle pointer above the marker.
                Vector2 a = { (float)(sx - 5), (float)(bar_y - 14) };
                Vector2 b = { (float)(sx + 5), (float)(bar_y - 14) };
                Vector2 c = { (float)sx,       (float)(bar_y - 8)  };
                DrawTriangle(a, b, c, slider_col);

                char buf[24];
                snprintf(buf, sizeof(buf), "%+.3f", eval_value);
                int lw = MeasureTextC(buf, 10);
                DrawTextC(buf, sx - lw / 2, bar_y - 28, 10, slider_col);
            }
        }

        // --- Collapsible tech tree (bottom-left). Hover or click to expand.
        //     Drawn LATE in the frame so its expanded panel visually covers
        //     the heuristic eval bar (and any other floating UI) below it. ---
        {
            const float icon_r     = TECH_ICON_R;
            const float icon_cx    = PAD + icon_r + 4.0f;
            const float icon_cy    = H   - PAD - icon_r - 4.0f;
            const float panel_size = TECH_PANEL_SIZE;

            // Panel sits above-and-right of the icon, sharing its bottom-left corner.
            const float panel_left   = icon_cx - icon_r;
            const float panel_bottom = icon_cy + icon_r;
            const float panel_top    = panel_bottom - panel_size;
            Rectangle panel_rect = { panel_left, panel_top, panel_size, panel_size };

            int hover_tech = -1;
            if (hovered_action >= 0 && actions[hovered_action].type == ActionType::ResearchTech)
                hover_tech = actions[hovered_action].param;

            bool icon_hovered  = CheckCollisionPointCircle(mouse, { icon_cx, icon_cy }, icon_r);
            bool panel_hovered = (tech_open || icon_hovered) && CheckCollisionPointRec(mouse, panel_rect);
            bool expanded      = tech_open || icon_hovered || panel_hovered;

            if (icon_hovered && clicked) {
                tech_open = !tech_open;
                clicked = false;  // consume so nothing else reacts this frame
            }

            // Expanded panel — drawn first so the icon overlays its corner.
            if (expanded) {
                DrawRectangleRec(panel_rect, PANEL_BG);
                DrawRectangleLinesEx(panel_rect, 1, PANEL_LINE);

                // Tech tree's bottom-left is just inside the panel; size insets to fit.
                float pad_in = panel_size * 0.10f;
                float tree_bl_x = panel_left + pad_in;
                float tree_bl_y = panel_bottom - pad_in;
                float tree_size = panel_size - 2 * pad_in;

                int pv = (view == ViewMode::Omni)    ? s.current_player()
                       : (view == ViewMode::P1)      ? 1
                       : (view == ViewMode::Current) ? s.current_player()
                                                     : 0;
                draw_tech_tree(s.get_player(pv).techs_mask(), pv,
                               tree_bl_x, tree_bl_y, tree_size, hover_tech);

                // "P0 TECH" / "P1 TECH" header
                Color hc = (pv == 0) ? COL_P0 : COL_P1;
                const char* hdr = (pv == 0) ? "P0 TECH" : "P1 TECH";
                int hsz = (int)std::max(10.0f, panel_size * 0.055f);
                DrawTextC(hdr, (int)(panel_left + 8), (int)(panel_top + 6), hsz, hc);
            }

            // Icon — small collapsed circle with three radial hint-lines.
            Color ic_fill = expanded ? PANEL_BG : Color{ 30, 30, 30, 255 };
            Color ic_line = expanded ? ((s.current_player() == 0) ? COL_P0 : COL_P1)
                                     : Color{ 130, 130, 130, 255 };
            DrawCircleV({ icon_cx, icon_cy }, icon_r, ic_fill);
            DrawCircleLines((int)icon_cx, (int)icon_cy, (int)icon_r, ic_line);
            // Three short rays as a visual hint of the radial layout
            float hint_r = icon_r * 0.65f;
            for (int k = 0; k < 3; k++) {
                float a = (k + 1) / 4.0f * (3.14159265f / 2.0f);  // 22.5°, 45°, 67.5°
                DrawLineEx({ icon_cx - icon_r * 0.15f, icon_cy + icon_r * 0.15f },
                           { icon_cx - icon_r * 0.15f + cosf(a) * hint_r,
                             icon_cy + icon_r * 0.15f - sinf(a) * hint_r },
                           1.5f, ic_line);
            }
        }

        // --- Action log panel ---
        {
            constexpr int ROW_H     = 17;
            constexpr int LOG_PAD   = 6;
            constexpr int CLEAR_H   = 24;
            constexpr int CLEAR_PAD = 4;
            int lx = TECH_W + MAP_PX + SIDEBAR;
            DrawRectangle(lx, 0, LOG_W, CONTENT_H, { 18, 18, 18, 255 });
            DrawLine(lx, 0, lx, CONTENT_H, PANEL_LINE);
            DrawTextC("LOG", lx + 8, 8, 14, GRAY);
            DrawLine(lx + 4, 28, lx + LOG_W - 4, 28, { 50, 50, 50, 255 });

            int log_bottom = CONTENT_H - CLEAR_H - CLEAR_PAD * 2;
            int rows_h     = log_bottom - 28 - LOG_PAD;
            int visible_rows = rows_h / ROW_H;
            int total_entries = (int)action_log.size();

            // Auto-scroll to bottom unless user has scrolled up
            int max_scroll = std::max(0, total_entries - visible_rows);
            if (log_scroll > max_scroll) log_scroll = max_scroll;

            // Mouse wheel scroll when over panel (rows area only)
            Rectangle rows_rect = { (float)lx, 28, (float)LOG_W, (float)(log_bottom - 28) };
            if (CheckCollisionPointRec(mouse, rows_rect)) {
                log_scroll -= (int)GetMouseWheelMove();
                if (log_scroll < 0) log_scroll = 0;
                if (log_scroll > max_scroll) log_scroll = max_scroll;
            }

            BeginScissorMode(lx, 28 + LOG_PAD, LOG_W, rows_h);
            for (int i = 0; i < visible_rows; i++) {
                int entry_idx = log_scroll + i;
                if (entry_idx >= total_entries) break;
                const std::string& entry = action_log[entry_idx];
                int ey = 28 + LOG_PAD + i * ROW_H;
                if (i % 2 == 0)
                    DrawRectangle(lx + 2, ey, LOG_W - 4, ROW_H - 1, { 25, 25, 25, 255 });
                LogColor lc = (entry_idx < (int)action_log_color.size())
                            ? action_log_color[entry_idx]
                            : LOG_COLOR_GREY;
                DrawTextC(entry.c_str(), lx + 6, ey + 2, 11, from_log_color(lc));
            }
            EndScissorMode();

            // Scrollbar
            if (total_entries > visible_rows) {
                float bar_h = (float)(log_bottom - 28) * visible_rows / total_entries;
                float bar_y = 28 + (float)(log_bottom - 28) * log_scroll / total_entries;
                DrawRectangle(lx + LOG_W - 4, (int)bar_y, 3, (int)bar_h, { 90, 90, 90, 255 });
            }

            // CLEAR button at the bottom of the panel. Disabled in replay mode
            // (replay_log overwrites action_log every frame, so clearing it
            // would just snap back).
            DrawLine(lx + 4, log_bottom, lx + LOG_W - 4, log_bottom, { 50, 50, 50, 255 });
            Rectangle clr_btn = { (float)(lx + CLEAR_PAD),
                                  (float)(log_bottom + CLEAR_PAD),
                                  (float)(LOG_W - CLEAR_PAD * 2),
                                  (float)CLEAR_H };
            bool clr_hovered = !replay_mode && CheckCollisionPointRec(mouse, clr_btn);
            Color clr_bg = replay_mode  ? Color{ 30, 30, 30, 255 }
                         : clr_hovered  ? Color{ 70, 70, 70, 255 }
                                        : Color{ 45, 45, 45, 255 };
            DrawRectangleRec(clr_btn, clr_bg);
            int cw = MeasureTextC("CLEAR", 12);
            Color clr_fg = replay_mode ? Color{ 110, 110, 110, 255 }
                                       : Color{ 200, 200, 200, 255 };
            DrawTextC("CLEAR", lx + LOG_W / 2 - cw / 2, log_bottom + CLEAR_PAD + 6, 12, clr_fg);
            if (clr_hovered && clicked) {
                action_log.clear();
                action_log_color.clear();
                // Leave Logger's circular buffer intact (preserves debug
                // history) but skip past it so old entries don't refill.
                last_logger_count = Logger::count;
                log_scroll = 0;
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
            int lw = MeasureTextC(label, 12);
            DrawTextC(label, W / 2 - lw / 2, scrub_y + 3, 12, { 200, 200, 200, 255 });
        }

        // Heatmap mode indicator (top-left of board, only when active).
        if (show_heatmap && !current_heatmap.empty()) {
            const char* mode = (heatmap_channel == -1)
                ? "GRAD-CAM  [ / ] to cycle channels"
                : TextFormat("CH %02d / 31  [ / ] to cycle", heatmap_channel);
            DrawRectangle(MAP_OFF + PAD, TOP_HUD + 2, MeasureTextC(mode, 11) + 8, 16, { 0, 0, 0, 160 });
            DrawTextC(mode, MAP_OFF + PAD + 4, TOP_HUD + 4, 11, { 255, 200, 80, 255 });
        }

        EndDrawing();
    }

    for (int owner = 0; owner < 2; owner++)
        for (int i = 0; i < (int)UnitType::Count; i++)
            if (unit_tex[owner][i].id != 0) UnloadTexture(unit_tex[owner][i]);
    for (int i = 0; i < (int)TerrainSprite::Count; i++)
        if (terrain_tex[i].id != 0) UnloadTexture(terrain_tex[i]);
    for (int i = 0; i < CITY_SPRITE_LEVELS; i++)
        if (city_tex[i].id != 0) UnloadTexture(city_tex[i]);
    if (g_star_tex.id != 0) UnloadTexture(g_star_tex);
    if (g_font.texture.id != 0) UnloadFont(g_font);
    CloseWindow();
    return 0;
}
