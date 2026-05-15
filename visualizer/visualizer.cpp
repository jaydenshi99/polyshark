#include "raylib.h"
#include "init.h"
#include "mapgen.h"
#include "unit_def.h"
#include "resource_def.h"
#include "building_def.h"
#include "tech_def.h"
#include "game_state.h"
#include <utility>
#include <cstdio>
#include <fstream>
#include <vector>
#include <string>

static int            TILE     = 56;   // recomputed on each map gen; do not make constexpr
static constexpr int PAD      = 4;
static constexpr int LABEL    = 20;   // space for row/col index labels
static constexpr int TOP_HUD  = 64;   // info bar across the top
static constexpr int SIDEBAR  = 230;
static constexpr int TECH_W   = 210;  // tech tree panel on the left
static constexpr int MAP_OFF  = TECH_W;  // map x-offset (tech panel sits left of it)
// Fixed canvas: constant pixel budget. TILE scales to fill this regardless of actual map_size.
static constexpr int MAP_CANVAS = 616;
static constexpr int MAP_PX     = MAP_CANVAS + PAD * 2 + LABEL;
static constexpr int W          = TECH_W + MAP_PX + SIDEBAR;
static constexpr int H          = TOP_HUD + MAP_CANVAS + PAD * 2 + LABEL;

enum class ViewMode { Omni, P0, P1, Current };

// Reusable colors
static constexpr Color PANEL_BG   = {  22,  22,  22, 255 };
static constexpr Color PANEL_LINE = {  65,  65,  65, 255 };
static constexpr Color SIDEBAR_BG = {  38,  38,  38, 255 };
static constexpr Color COL_P0     = {  80, 140, 220, 255 };
static constexpr Color COL_P1     = { 220,  80,  80, 255 };

// Tile pixel origin
static inline int tile_px(int x) { return MAP_OFF + PAD + LABEL + x * TILE; }
static inline int tile_py(int y) { return TOP_HUD + PAD + LABEL + y * TILE; }

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


static void action_cost(const Action& a, char* buf, int size) {
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
                tech_def(static_cast<TechType>(a.param)).cost);
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
            snprintf(buf, size, "End Turn");
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
            static const char* unit_names[] = { "?", "Warrior", "Archer", "Rider" };
            const char* uname = (a.param > 0 && a.param < 4) ? unit_names[a.param] : "?";
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
// Terrain icons — each takes the tile's top-left pixel (px, py)
// ---------------------------------------------------------------------------
static void draw_village_icon(int px, int py) {
    int hx = px + TILE/2 - 10, hy = py + TILE/2 - 4;
    DrawRectangle(hx, hy + 8, 20, 12, { 200, 160, 90, 255 });
    DrawRectangleLines(hx, hy + 8, 20, 12, BLACK);
    DrawRectangle(hx + 7, hy + 13, 6, 7, { 80, 50, 20, 255 });
    Vector2 rl = { (float)(hx - 2),  (float)(hy + 9) };
    Vector2 rr = { (float)(hx + 22), (float)(hy + 9) };
    Vector2 rt = { (float)(hx + 10), (float)(hy)     };
    DrawTriangle(rl, rr, rt, { 160, 60, 30, 255 });
    DrawTriangleLines(rl, rr, rt, BLACK);
}

static void draw_mountain_icon(int px, int py) {
    float cx = (float)(px + TILE/2), cy = (float)(py + TILE/2);
    Vector2 C  = { cx,                     cy                     };
    Vector2 TL = { (float)px,              (float)py              };
    Vector2 TR = { (float)(px + TILE - 1), (float)py              };
    Vector2 BL = { (float)px,              (float)(py + TILE - 1) };
    Vector2 BR = { (float)(px + TILE - 1), (float)(py + TILE - 1) };
    DrawTriangle(TL, C,  TR, { 210, 200, 180, 255 });
    DrawTriangle(TR, C,  BR, { 150, 135, 110, 255 });
    DrawTriangle(BR, C,  BL, {  70,  62,  50, 255 });
    DrawTriangle(BL, C,  TL, { 105,  93,  75, 255 });
    Color ridge = { 40, 35, 28, 200 };
    DrawLine((int)cx, (int)cy, px,            py,             ridge);
    DrawLine((int)cx, (int)cy, px + TILE - 1, py,             ridge);
    DrawLine((int)cx, (int)cy, px,            py + TILE - 1,  ridge);
    DrawLine((int)cx, (int)cy, px + TILE - 1, py + TILE - 1,  ridge);
}

static void draw_forest_icon(int px, int py) {
    Color trunk = { 100, 65, 25, 255 };
    Color leaf  = {  34, 110, 34, 255 };
    auto draw_tree = [&](int tx, int ty) {
        DrawRectangle(tx + 3, ty + 14, 4, 8, trunk);
        DrawRectangleLines(tx + 3, ty + 14, 4, 8, BLACK);
        Vector2 tl = { (float)tx,        (float)(ty + 15) };
        Vector2 tr = { (float)(tx + 10), (float)(ty + 15) };
        Vector2 tt = { (float)(tx + 5),  (float)ty        };
        DrawTriangle(tl, tr, tt, leaf);
        DrawTriangleLines(tl, tr, tt, BLACK);
    };
    draw_tree(px + 2,          py + 4);
    draw_tree(px + TILE/2 - 5, py + TILE/2);
    draw_tree(px + TILE - 14,  py + 4);
}

// ---------------------------------------------------------------------------
// Resource icons — each takes the tile's top-left pixel (px, py)
// ---------------------------------------------------------------------------
static void draw_fruit_icon(int px, int py) {
    Color red   = { 210,  40,  40, 255 };
    Color brown = { 100,  60,  20, 255 };
    int cx = px + TILE/2, cy = py + TILE/2;
    int acx[3] = { cx,     cx - 9, cx + 9 };
    int acy[3] = { cy - 8, cy + 5, cy + 5 };
    for (int i = 0; i < 3; i++) {
        DrawRectangle(acx[i] - 1, acy[i] - 6, 2, 3, brown);
        DrawCircle(acx[i] - 1, acy[i], 3, red);
        DrawCircle(acx[i] + 1, acy[i], 3, red);
        DrawCircleLines(acx[i] - 1, acy[i], 3, BLACK);
        DrawCircleLines(acx[i] + 1, acy[i], 3, BLACK);
    }
}

static void draw_crop_icon(int px, int py) {
    Color g = { 80, 190, 60, 255 };
    int left = px + 4, top = py + 4;
    int cols = 3, rows = 3;
    int cw = (TILE - 8) / cols, rh = (TILE - 8) / rows;
    int hw = 2, h = 9;  // half-width and height of each blade
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            int bx = left + c * cw + cw / 2;
            int by = top  + r * rh + rh;
            Vector2 br = { (float)(bx + hw), (float)by       };
            Vector2 tip = { (float)bx,       (float)(by - h) };
            Vector2 bl = { (float)(bx - hw), (float)by       };
            DrawTriangle(br, tip, bl, g);
            DrawTriangleLines(br, tip, bl, BLACK);
        }
}

static void draw_farm_icon(int px, int py) {
    // Wheat field rows
    Color soil  = { 160, 110,  50, 255 };
    Color wheat = { 220, 185,  50, 255 };
    Color head  = { 240, 210,  80, 255 };
    int left = px + 4, top = py + 6, w = TILE - 9;
    DrawRectangle(left, top, w, TILE - 12, soil);
    DrawRectangleLines(left, top, w, TILE - 12, BLACK);
    int rows = 3;
    int row_h = (TILE - 12) / rows;
    for (int r = 0; r < rows; r++) {
        int ry = top + r * row_h + 2;
        int cols = 5;
        int col_w = w / cols;
        for (int c = 0; c < cols; c++) {
            int sx = left + c * col_w + col_w/2;
            DrawRectangle(sx - 1, ry + 2, 2, row_h - 4, wheat);
            DrawRectangle(sx - 2, ry,     4, 3,          head);
        }
    }
}

static void draw_animal_icon(int px, int py) {
    Color ac = { 140, 80, 30, 255 };
    int ax = px + TILE/2 - 11, ay = py + TILE/2 - 5;
    DrawRectangle(ax,      ay,     15, 7, ac);
    DrawRectangle(ax + 15, ay - 3,  7, 7, ac);
    DrawRectangle(ax + 1,  ay + 7,  3, 6, ac);
    DrawRectangle(ax + 5,  ay + 7,  3, 6, ac);
    DrawRectangle(ax + 10, ay + 7,  3, 6, ac);
    DrawRectangle(ax + 15, ay + 7,  3, 6, ac);
    DrawRectangleLines(ax,      ay,     15, 7, BLACK);
    DrawRectangleLines(ax + 15, ay - 3,  7, 7, BLACK);
    DrawRectangleLines(ax + 1,  ay + 7,  3, 6, BLACK);
    DrawRectangleLines(ax + 5,  ay + 7,  3, 6, BLACK);
    DrawRectangleLines(ax + 10, ay + 7,  3, 6, BLACK);
    DrawRectangleLines(ax + 15, ay + 7,  3, 6, BLACK);
}

static void draw_metal_icon(int px, int py) {
    Color gold = { 255, 210, 30, 255 };
    int cx = px + TILE/2, cy = py + TILE/2;
    int d = TILE / 4;
    int xs[3] = { cx - d/2 - 4, cx - d - 4, cx + d - 5 };
    int ys[3] = { cy - d - 4,   cy + d/2 - 4, cy + d/2 - 4 };
    for (int i = 0; i < 3; i++) {
        DrawRectangle(xs[i], ys[i], 9, 9, gold);
        DrawRectangleLines(xs[i], ys[i], 9, 9, BLACK);
    }
}

// ---------------------------------------------------------------------------
// Building / city decoration icons
// ---------------------------------------------------------------------------
static void draw_castle_icon(int px, int py) {
    Color stone = { 170, 160, 145, 255 };
    Color dark  = {  60,  55,  48, 255 };
    int cx = px + TILE/2;
    int base_y = py + TILE/2 - 4;  // shifted up

    // Left turret
    DrawRectangle(cx - 12, base_y - 8,  7, 8, stone);
    DrawRectangleLines(cx - 12, base_y - 8, 7, 8, dark);
    DrawRectangle(cx - 12, base_y - 11, 2, 3, stone);
    DrawRectangle(cx - 9,  base_y - 11, 2, 3, stone);
    DrawRectangle(cx - 6,  base_y - 11, 2, 3, stone);
    DrawRectangleLines(cx - 12, base_y - 11, 2, 3, dark);
    DrawRectangleLines(cx - 9,  base_y - 11, 2, 3, dark);
    DrawRectangleLines(cx - 6,  base_y - 11, 2, 3, dark);

    // Right turret
    DrawRectangle(cx + 5, base_y - 8,  7, 8, stone);
    DrawRectangleLines(cx + 5, base_y - 8, 7, 8, dark);
    DrawRectangle(cx + 5,  base_y - 11, 2, 3, stone);
    DrawRectangle(cx + 8,  base_y - 11, 2, 3, stone);
    DrawRectangle(cx + 11, base_y - 11, 2, 3, stone);
    DrawRectangleLines(cx + 5,  base_y - 11, 2, 3, dark);
    DrawRectangleLines(cx + 8,  base_y - 11, 2, 3, dark);
    DrawRectangleLines(cx + 11, base_y - 11, 2, 3, dark);

    // Central tower
    DrawRectangle(cx - 5, base_y - 13, 10, 13, stone);
    DrawRectangleLines(cx - 5, base_y - 13, 10, 13, dark);
    DrawRectangle(cx - 4, base_y - 16, 2, 4, stone);
    DrawRectangle(cx - 1, base_y - 16, 2, 4, stone);
    DrawRectangle(cx + 2, base_y - 16, 2, 4, stone);
    DrawRectangleLines(cx - 4, base_y - 16, 2, 4, dark);
    DrawRectangleLines(cx - 1, base_y - 16, 2, 4, dark);
    DrawRectangleLines(cx + 2, base_y - 16, 2, 4, dark);
    DrawRectangle(cx - 2, base_y - 6, 4, 6, dark);  // gate

    // Base wall
    DrawRectangle(cx - 12, base_y, 24, 3, stone);
    DrawRectangleLines(cx - 12, base_y, 24, 3, dark);
}

static void draw_small_castle_icon(int px, int py) {
    Color stone = { 170, 160, 145, 255 };
    Color dark  = {  60,  55,  48, 255 };
    int cx = px + TILE/2;
    int base_y = py + TILE/2 + 2;

    // Single squat tower
    DrawRectangle(cx - 5, base_y - 7, 10, 7, stone);
    DrawRectangleLines(cx - 5, base_y - 7, 10, 7, dark);
    // Battlements
    DrawRectangle(cx - 5, base_y - 10, 3, 3, stone);
    DrawRectangle(cx - 1, base_y - 10, 3, 3, stone);
    DrawRectangle(cx + 3, base_y - 10, 3, 3, stone);
    DrawRectangleLines(cx - 5, base_y - 10, 3, 3, dark);
    DrawRectangleLines(cx - 1, base_y - 10, 3, 3, dark);
    DrawRectangleLines(cx + 3, base_y - 10, 3, 3, dark);
    // Gate
    DrawRectangle(cx - 2, base_y - 4, 4, 4, dark);
    // Base
    DrawRectangle(cx - 7, base_y, 14, 2, stone);
    DrawRectangleLines(cx - 7, base_y, 14, 2, dark);
}

static void draw_mine_icon(int px, int py) {
    Color dark = {  35,  28,  20, 255 };
    Color wood = { 120,  80,  35, 255 };
    int mx = px + TILE/2 - 8, my = py + TILE/2 - 6;
    DrawRectangle(mx,      my + 4, 16, 12, dark);
    DrawRectangleLines(mx, my + 4, 16, 12, BLACK);
    DrawRectangle(mx,      my + 4, 3, 12, wood);
    DrawRectangle(mx + 13, my + 4, 3, 12, wood);
    DrawRectangle(mx,      my + 2, 16,  3, wood);
    DrawRectangleLines(mx,      my + 4,  3, 12, BLACK);
    DrawRectangleLines(mx + 13, my + 4,  3, 12, BLACK);
    DrawRectangleLines(mx,      my + 2, 16,  3, BLACK);
}

static void draw_shield_icon(int px, int py) {
    int sx = px + TILE - 15, sy = py + TILE - 20;
    Vector2 tl = { (float)sx,     (float)sy     };
    Vector2 tr = { (float)(sx+7), (float)sy     };
    Vector2 r  = { (float)(sx+7), (float)(sy+5) };
    Vector2 b  = { (float)(sx+3), (float)(sy+9) };
    Vector2 l  = { (float)sx,     (float)(sy+5) };
    DrawTriangle(tl, l, r,  { 160, 160, 160, 255 });
    DrawTriangle(l,  b, r,  { 160, 160, 160, 255 });
    DrawTriangle(tl, r, tr, { 160, 160, 160, 255 });
    Vector2 pts[] = { tl, tr, r, b, l, tl };
    DrawLineStrip(pts, 6, BLACK);
}

static void draw_workshop_icon(int px, int py) {
    int wx = px + 5, wy = py + TILE - 21;
    Vector2 wbl = { (float)wx,      (float)(wy + 10) };
    Vector2 wbr = { (float)(wx+10), (float)(wy + 10) };
    Vector2 wt  = { (float)(wx+5),  (float)wy        };
    DrawTriangle(wbl, wbr, wt, WHITE);
    DrawTriangleLines(wbl, wbr, wt, BLACK);
}

static void draw_pop_bar(int px, int py, int pop, int needed, int pending_pop) {
    int bx = px + 2, by = py + TILE - 9, bw = TILE - 5, bh = 6;
    DrawRectangle(bx, by, bw, bh, { 20, 20, 20, 200 });
    int filled = (bw * pop) / needed;
    if (filled > 0)
        DrawRectangle(bx, by, filled, bh, { 255, 210, 60, 230 });
    if (pending_pop > 0) {
        int ghost_end = (bw * (pop + pending_pop)) / needed;
        int ghost_w   = ghost_end - filled;
        if (ghost_w > 0)
            DrawRectangle(bx + filled, by, ghost_w, bh, { 255, 210, 60, 100 });
    }
    for (int seg = 1; seg < needed; seg++) {
        int lx = bx + (bw * seg) / needed;
        DrawLine(lx, by, lx, by + bh, { 0, 0, 0, 180 });
    }
    DrawRectangleLines(bx, by, bw, bh, { 80, 80, 80, 200 });
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
// scale: zoom factor (1.0 = default); >1 = zoomed in, nodes closer together
static void draw_tech_tree(uint32_t owned_techs, int player_idx,
                           int y_top, int y_bot, const char* label,
                           int hover_tech = -1, float scale = 1.0f)
{
    const Color owned_col  = (player_idx == 0) ? COL_P0 : COL_P1;
    const Color avail_col  = { 130, 130, 130, 255 };
    const Color locked_col = {  55,  55,  55, 255 };
    const Color edge_col   = {  70,  70,  70, 255 };
    const Color hover_col  = { 255, 220,  60, 255 };

    constexpr int   TIERS   = 3;
    constexpr int   FONT_SZ = 8;
    const float     NODE_R  = 9.0f * scale;

    int area_h   = y_bot - y_top;
    int header_h = (label != nullptr) ? 13 : 0;
    int centre_y = y_top + area_h / 2;

    // Base tier spacing: 55px, scaled. Centred vertically in the region.
    float step    = 55.0f * scale;
    float tree_h  = step * (TIERS - 1);
    float base_y  = (float)centre_y - tree_h * 0.5f + (float)header_h * 0.5f;

    auto tier_y = [&](int tier) -> float { return base_y + tier * step; };

    // Node x: spread across TECH_W, scaled toward/away from centre
    auto node_x = [&](float slot, int tier) -> float {
        if (tier == 0) return TECH_W * 0.5f;
        float margin = 25.0f / scale;
        float step_x = (TECH_W - margin * 2) / 3.0f;
        float x = margin + slot * step_x;
        float centre = TECH_W * 0.5f;
        return centre + (x - centre) * scale;
    };

    uint32_t avail = available_techs(owned_techs);

    if (label) {
        int lw = MeasureText(label, FONT_SZ + 1);
        DrawText(label, TECH_W / 2 - lw / 2, y_top + 2, FONT_SZ + 1, owned_col);
    }

    // Edges first
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
    uint64_t gen_seed = 1;
    int climate[MAX_MAP_TILES] = {};
    auto new_map = [&]() {
        MapGenParams p = MapGen::drylands_defaults();
        p.seed = gen_seed++;
        MapGenResult r = MapGen(p).generate();
        int mtsz = r.state.map_tiles();
        for (int i = 0; i < mtsz; i++) climate[i] = r.climate[i];
        return r.state;
    };
    GameState initial = new_map();
    TILE = MAP_CANVAS / initial.map_size();
    GameState s = initial;

    // Replay mode: --replay <path>
    bool replay_mode = false;
    std::vector<GameState> replay_states;
    int replay_step = 0;
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
                rs = make_game();
                // first token was already consumed — push it back as action fields
                int t = std::stoi(first), fr, to, pa;
                f >> fr >> to >> pa;
                replay_states.push_back(rs);
                replay_states.push_back(rs = rs.apply_action({(ActionType)t, fr, to, pa, true}));
            }
            replay_states.push_back(rs);
            int t, fr, to, pa;
            while (f >> t >> fr >> to >> pa)
                replay_states.push_back(rs = rs.apply_action({(ActionType)t, fr, to, pa, true}));
            initial = s = replay_states[0];
            TILE = MAP_CANVAS / s.map_size();
            break;
        }
    }

    ViewMode  view = ViewMode::Omni;

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

    InitWindow(W, H, "Polyshark");
    SetTargetFPS(60);

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

        // Scroll wheel zooms tech tree when cursor is over the panel
        if (mouse.x < TECH_W) {
            float wheel = GetMouseWheelMove();
            if (wheel != 0.0f) {
                tech_zoom *= (wheel > 0 ? 1.15f : 1.0f / 1.15f);
                if (tech_zoom < 0.4f) tech_zoom = 0.4f;
                if (tech_zoom > 3.0f) tech_zoom = 3.0f;
            }
        }

        // Scroll wheel scrolls the sidebar actions list
        int SB_CONTENT_TOP = SIDEBAR_TOP;
        int SB_CONTENT_BOT = H - 80;  // above regen + reset buttons
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
        const int scrub_y = H - SCRUB_H;
        if (replay_mode) {
            bool over_scrub = mouse.y >= scrub_y;
            if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && over_scrub) scrubbing = true;
            if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON))              scrubbing = false;
            if (scrubbing) {
                float t = (mouse.x) / (float)W;
                t = t < 0 ? 0 : t > 1 ? 1 : t;
                int new_step = (int)(t * ((int)replay_states.size() - 1));
                if (new_step != replay_step) {
                    replay_step = new_step;
                    s = replay_states[replay_step];
                    refresh_borders();
                    s.legal_actions(actions, action_count);
                }
            }
        }

        // Map tile selection — click toggles; clicking same tile deselects
        {
            int tx = ((int)mouse.x - (MAP_OFF + PAD + LABEL)) / TILE;
            int ty = ((int)mouse.y - (TOP_HUD + PAD + LABEL)) / TILE;
            if (clicked && !scrubbing && tx >= 0 && tx < cur_msz && ty >= 0 && ty < cur_msz) {
                int tidx = to_index(tx, ty, cur_msz);
                selected_tile = (selected_tile == tidx) ? -1 : tidx;
            }
        }

        auto is_highlighted = [&](int idx) {
            for (int h : highlight_tiles) if (h == idx) return true;
            return false;
        };

        BeginDrawing();
        ClearBackground({ 30, 30, 30, 255 });

        // Border owner per tile: read directly from stored border_city_id.
        // -1 = unclaimed; otherwise look up city owner for player colour.
        int border_owner[MAX_MAP_TILES];
        for (int i = 0; i < s.map_tiles(); i++) {
            int cid = s.tile_at(i).border_city_id();
            border_owner[i] = (cid >= 0) ? s.get_city(cid).owner() : -1;
        }

        // --- Row / col indices ---
        for (int x = 0; x < cur_msz; x++)
            DrawText(TextFormat("%d", x), tile_px(x) + TILE/2 - 4, TOP_HUD + PAD + 2, 14, GRAY);
        for (int y = 0; y < cur_msz; y++)
            DrawText(TextFormat("%d", y), MAP_OFF + PAD + 2, tile_py(y) + TILE/2 - 7, 14, GRAY);

        // --- Map ---
        int vp = (view == ViewMode::P1)      ? 1 :
                 (view == ViewMode::Current) ? s.current_player() : 0;

        for (int y = 0; y < cur_msz; y++) {
            for (int x = 0; x < cur_msz; x++) {
                const Tile& t = s.tile_at(to_index(x, y));
                int px  = tile_px(x);
                int py  = tile_py(y);
                int idx = to_index(x, y);

                bool fogged = (view != ViewMode::Omni) && !s.is_explored(vp, idx);
                bool dimmed = false;  // explored tiles stay permanently visible

                // Fogged tiles: solid black, no info
                if (fogged) {
                    DrawRectangle(px, py, TILE - 1, TILE - 1, { 18, 18, 18, 255 });
                } else {
                    TerrainType ter = t.terrain();
                    // Village and Forest use field as base
                    Color base = (ter == TerrainType::Village || ter == TerrainType::Forest)
                                 ? terrain_color(TerrainType::Field)
                                 : terrain_color(ter);
                    DrawRectangle(px, py, TILE - 1, TILE - 1, base);

                    // Terrain detail icons
                    if      (ter == TerrainType::Village)  draw_village_icon(px, py);
                    else if (ter == TerrainType::Mountain) draw_mountain_icon(px, py);
                    else if (ter == TerrainType::Forest)   draw_forest_icon(px, py);

                    // Climate tint
                    Color ctint = (climate[idx] == 0)
                        ? Color{ 255, 170, 50,  28 }   // warm amber — P0 tribe
                        : Color{  50, 180, 200, 28 };  // cool teal  — P1 tribe
                    DrawRectangle(px, py, TILE - 1, TILE - 1, ctint);

                    // Territory colour overlay
                    Color tc = BLANK;
                    for (int p = 0; p < 2 && tc.a == 0; p++)
                        tc = colourers[p].tile_colour(idx, s);
                    if (tc.a > 0)
                        DrawRectangle(px, py, TILE - 1, TILE - 1, tc);
                }

                // Resources only when fully visible
                if (!fogged && !dimmed && t.resource() != ResourceType::None) {
                    switch (t.resource()) {
                        case ResourceType::Fruit:  draw_fruit_icon(px, py);  break;
                        case ResourceType::Crop:   draw_crop_icon(px, py);   break;
                        case ResourceType::Animal: draw_animal_icon(px, py); break;
                        case ResourceType::Metal:  draw_metal_icon(px, py);  break;
                        default: break;
                    }
                }

                // Buildings
                if (!fogged && t.has_building()) {
                    if      (t.building() == BuildingType::Mine) draw_mine_icon(px, py);
                    else if (t.building() == BuildingType::Farm) draw_farm_icon(px, py);
                }

                // City ring + population bar when explored
                if (!fogged && t.has_city()) {
                    DrawRectangleLinesEx(
                        { (float)px + 2, (float)py + 2, (float)TILE - 5, (float)TILE - 5 },
                        3, WHITE);

                    const City& city = s.get_city(t.city_id());

                    // City ID bottom-left, stars bottom-right — same y, above the pop bar
                    int label_y = py + TILE - 22;
                    DrawText(TextFormat("%d", t.city_id()), px + 4, label_y, 11, WHITE);

                    const char* spt_str = TextFormat("%d*", city.stars_per_turn());
                    int spt_w = MeasureText(spt_str, 11);
                    DrawText(spt_str, px + TILE - 5 - spt_w, label_y, 11, { 255, 230, 50, 255 });

                    if (city.is_capital())   draw_castle_icon(px, py);
                    else if (city.owner() >= 0) draw_small_castle_icon(px, py);
                    if (city.has_walls())    draw_shield_icon(px, py);
                    if (city.has_workshop()) draw_workshop_icon(px, py);

                    int pending_pop = 0;
                    if (hovered_action >= 0) {
                        const Action& ha = actions[hovered_action];
                        if ((ha.type == ActionType::HarvestResource || ha.type == ActionType::DebugAddPop)
                            && ha.from == t.city_id()) {
                            pending_pop = (ha.type == ActionType::HarvestResource)
                                ? resource_def(static_cast<ResourceType>(ha.param)).pop_reward
                                : ha.param;
                        }
                    }
                    draw_pop_bar(px, py, city.population(), city.level() + 1, pending_pop);
                }

                // Units — visible tiles or own units
                if (t.has_unit()) {
                    const Unit& u = s.get_unit(t.unit_id());
                    bool show = (view == ViewMode::Omni)
                             || s.is_visible(vp, idx)
                             || u.owner() == vp;
                    if (show) {
                        int cx = px + TILE / 2;
                        int cy = py + TILE / 2;
                        Color uc = (u.owner() == 0) ? BLUE : RED;
                        DrawCircle(cx, cy, TILE / 5, uc);
                        static const char* unit_icon[] = { "?", "W", "A", "R" };
                        const char* icon = unit_icon[(int)u.type()];
                        DrawText(icon, cx - MeasureText(icon, 10) / 2, cy - 5, 10, WHITE);
                        DrawText(TextFormat("%d", u.hp()), px + 4, py + 4, 11, WHITE);
                    }
                }

                // Hover highlight overlay
                if (is_highlighted(idx)) {
                    DrawRectangle(px, py, TILE - 1, TILE - 1, { 255, 255, 255, 40 });
                    DrawRectangleLinesEx({ (float)px, (float)py, (float)TILE-1, (float)TILE-1 },
                                         3, { 255, 255, 80, 220 });

                }

                DrawRectangleLines(px, py, TILE, TILE, { 0, 0, 0, 60 });
            }
        }

        // Border edge lines — draw on tile edges that are on the territory boundary
        for (int y = 0; y < cur_msz; y++) {
            for (int x = 0; x < cur_msz; x++) {
                int idx   = to_index(x, y, cur_msz);
                int owner = border_owner[idx];
                if (owner == -1) continue;
                if ((view != ViewMode::Omni) && !s.is_explored(vp, idx)) continue;  // vp already tracks current player in Current mode

                int px = tile_px(x);
                int py = tile_py(y);
                Color ec = (owner == 0) ? Color{ 80, 150, 255, 200 }
                                        : Color{ 255, 80,  80, 200 };

                // Draw a line on each edge where the adjacent tile is outside this border
                constexpr float BORDER_THICK = 3.0f;
                auto outside = [&](int nx, int ny) {
                    return !in_bounds(nx, ny) || border_owner[to_index(nx, ny)] != owner;
                };
                if (outside(x, y-1)) DrawLineEx({(float)px,        (float)py        }, {(float)(px+TILE-1), (float)py        }, BORDER_THICK, ec);
                if (outside(x, y+1)) DrawLineEx({(float)px,        (float)(py+TILE-1)}, {(float)(px+TILE-1),(float)(py+TILE-1)}, BORDER_THICK, ec);
                if (outside(x-1, y)) DrawLineEx({(float)px,        (float)py        }, {(float)px,          (float)(py+TILE-1)}, BORDER_THICK, ec);
                if (outside(x+1, y)) DrawLineEx({(float)(px+TILE-1),(float)py        }, {(float)(px+TILE-1),(float)(py+TILE-1)}, BORDER_THICK, ec);
            }
        }

        // --- Tech tree panel (left) ---
        DrawRectangle(0, TOP_HUD, TECH_W, H - TOP_HUD, PANEL_BG);
        DrawLine(TECH_W - 1, TOP_HUD, TECH_W - 1, H, PANEL_LINE);
        {
            int hover_tech = -1;
            if (hovered_action >= 0 && actions[hovered_action].type == ActionType::ResearchTech)
                hover_tech = actions[hovered_action].param;
            if (view == ViewMode::Omni) {
                int mid_y = TOP_HUD + (H - TOP_HUD) / 2;
                DrawLine(4, mid_y, TECH_W - 5, mid_y, { 50, 50, 50, 255 });
                int cur = s.current_player();
                draw_tech_tree(s.get_player(0).get_techs(), 0, TOP_HUD, mid_y, "P0", cur == 0 ? hover_tech : -1, tech_zoom);
                draw_tech_tree(s.get_player(1).get_techs(), 1, mid_y,  H,      "P1", cur == 1 ? hover_tech : -1, tech_zoom);
            } else {
                int pv = (view == ViewMode::P1) ? 1 : (view == ViewMode::Current) ? s.current_player() : 0;
                draw_tech_tree(s.get_player(pv).get_techs(), pv, TOP_HUD, H, nullptr, hover_tech, tech_zoom);
            }
        }

        // --- Top HUD bar ---
        DrawRectangle(0, 0, W, TOP_HUD, PANEL_BG);
        DrawLine(0, TOP_HUD, W, TOP_HUD, PANEL_LINE);

        // Turn + active player (left side)
        DrawText(TextFormat("Turn %d", s.get_turn()),       PAD + 4, 8,  22, WHITE);
        DrawText(TextFormat("P%d to move", s.current_player()), PAD + 4, 34, 18, LIGHTGRAY);

        // Stars — large, player-coloured (centre)
        DrawText(TextFormat("P0  %d *", s.get_stars(0)), W / 2 - 110, 10, 28, { 100, 160, 255, 255 });
        DrawText(TextFormat("P1  %d *", s.get_stars(1)), W / 2 + 10,  10, 28, { 255, 100, 100, 255 });

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
        DrawRectangle(SB, TOP_HUD, SIDEBAR, H - TOP_HUD, SIDEBAR_BG);
        DrawLine(SB, TOP_HUD, SB, H, PANEL_LINE);
        DrawText("LEGAL ACTIONS", SB + 8, TOP_HUD + 10, 16, GRAY);
        DrawLine(SB + 4, TOP_HUD + 30, SB + SIDEBAR - 4, TOP_HUD + 30, { 60, 60, 60, 255 });

        int  applied    = -1;
        bool reset      = false;
        bool regenerate = false;
        // Replay step controls
        if (replay_mode) {
            auto jump_to = [&](int step) {
                step = step < 0 ? 0 : step > (int)replay_states.size() - 1 ? (int)replay_states.size() - 1 : step;
                if (step != replay_step) {
                    replay_step = step;
                    s = replay_states[replay_step];
                    refresh_borders();
                    s.legal_actions(actions, action_count);
                }
            };

            static float key_timer = 0.0f;
            static int   key_held  = 0; // -1 left, 0 none, 1 right
            constexpr float HOLD_DELAY  = 0.3f;
            constexpr float HOLD_REPEAT = 0.07f;

            if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_SPACE)) { jump_to(replay_step + 1); key_held = 1;  key_timer = 0.0f; }
            else if (IsKeyPressed(KEY_LEFT))                         { jump_to(replay_step - 1); key_held = -1; key_timer = 0.0f; }
            else if (key_held != 0) {
                bool still = (key_held == 1 && (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_SPACE)))
                           || (key_held == -1 && IsKeyDown(KEY_LEFT));
                if (still) {
                    key_timer += GetFrameTime();
                    float threshold = (key_timer < HOLD_DELAY + HOLD_REPEAT) ? HOLD_DELAY : HOLD_REPEAT;
                    if (key_timer >= threshold) { jump_to(replay_step + key_held); key_timer -= threshold; }
                } else { key_held = 0; key_timer = 0.0f; }
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
                char cost[16];  action_cost(a, cost, sizeof(cost));
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
            Rectangle regen_btn = { (float)SB + 4, (float)H - 72, (float)SIDEBAR - 8, 28 };
            bool hovered = CheckCollisionPointRec(mouse, regen_btn);
            DrawRectangleRec(regen_btn, hovered ? Color{ 30, 130, 80, 255 } : Color{ 20, 80, 50, 255 });
            int tw = MeasureText("REGEN MAP", 14);
            DrawText("REGEN MAP", SB + SIDEBAR / 2 - tw / 2, H - 64, 14, WHITE);
            if (hovered && clicked)
                regenerate = true;
        }

        // --- Reset button (bottom of sidebar) ---
        {
            Rectangle reset_btn = { (float)SB + 4, (float)H - 36, (float)SIDEBAR - 8, 28 };
            bool hovered = CheckCollisionPointRec(mouse, reset_btn);
            DrawRectangleRec(reset_btn, hovered ? Color{ 140, 50, 50, 255 } : Color{ 90, 30, 30, 255 });
            DrawText("RESET", SB + SIDEBAR / 2 - 24, H - 30, 18, WHITE);
            if (hovered && clicked)
                reset = true;
        }

        // Apply after the loop so we don't mutate actions[] mid-render
        if (regenerate) {
            initial = new_map();
            TILE = MAP_CANVAS / initial.map_size();
            s = initial;
            init_colourers();
            s.legal_actions(actions, action_count);
            sidebar_scroll = 0;
            selected_tile  = -1;
        } else if (reset) {
            if (replay_mode) {
                replay_step = 0;
                s = replay_states[0];
            } else {
                s = initial;
            }
            init_colourers();
            s.legal_actions(actions, action_count);
            sidebar_scroll = 0;
            selected_tile  = -1;
        }
        if (applied >= 0 && !replay_mode) {
            s = s.apply_action(actions[applied]);
            refresh_borders();
            s.legal_actions(actions, action_count);
            sidebar_scroll = 0;
        }


        // --- Selected tile info box ---
        if (selected_tile >= 0) {
            const Tile& ht = s.tile_at(selected_tile);
            int cid = ht.border_city_id();

            char line1[48], line2[48];
            if (cid >= 0) {
                const City& bc = s.get_city(cid);
                snprintf(line1, sizeof(line1), "City %d  P%d  L%d", cid, bc.owner(), bc.level());
            } else {
                snprintf(line1, sizeof(line1), "Unclaimed");
            }
            if (ht.has_unit()) {
                const Unit& u = s.get_unit(ht.unit_id());
                static const char* unames[] = { "?", "Warrior", "Archer", "Rider" };
                snprintf(line2, sizeof(line2), "%s (P%d %dHP)",
                    unames[(int)u.type()], u.owner(), u.hp());
            } else {
                line2[0] = '\0';
            }

            int tw1 = MeasureText(line1, 13);
            int tw2 = line2[0] ? MeasureText(line2, 12) : 0;
            int bw  = (tw1 > tw2 ? tw1 : tw2) + 12;
            int bh  = line2[0] ? 36 : 20;

            // Anchor just below-right of the selected tile, clamped to window
            int sx, sy;
            to_coords(selected_tile, sx, sy);
            int bx = tile_px(sx) + TILE + 2;
            int by = tile_py(sy);
            if (bx + bw > W) bx = tile_px(sx) - bw - 2;
            if (by + bh > H) by = H - bh - 2;

            DrawRectangle(bx, by, bw, bh, { 20, 20, 20, 220 });
            DrawRectangleLines(bx, by, bw, bh, { 80, 80, 80, 200 });
            Color c1 = (cid >= 0)
                ? (s.get_city(cid).owner() == 0 ? COL_P0 : COL_P1)
                : Color{ 160, 160, 160, 255 };
            DrawText(line1, bx + 6, by + 4, 13, c1);
            if (line2[0])
                DrawText(line2, bx + 6, by + 20, 12, LIGHTGRAY);

            // Highlight the selected tile with a distinct outline
            DrawRectangleLinesEx(
                { (float)tile_px(sx), (float)tile_py(sy), (float)TILE - 1, (float)TILE - 1 },
                2, WHITE);
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

    CloseWindow();
    return 0;
}
