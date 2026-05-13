#include "raylib.h"
#include "init.h"
#include "unit_def.h"
#include "action.h"
#include <utility>
#include <cstdio>

static constexpr int TILE    = 56;
static constexpr int PAD     = 4;
static constexpr int LABEL   = 20;   // space for row/col index labels
static constexpr int HUD     = 44;
static constexpr int SIDEBAR = 230;
static constexpr int MAP_PX  = MAP_SIZE * TILE + PAD * 2 + LABEL;
static constexpr int W       = MAP_PX + SIDEBAR;
static constexpr int H       = MAP_SIZE * TILE + PAD * 2 + LABEL + HUD;

enum class ViewMode { Omni, P0, P1 };

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

static Color resource_color(ResourceType r) {
    switch (r) {
        case ResourceType::Fruit:  return { 220,  60,  60, 255 };
        case ResourceType::Game:   return { 140,  80,  30, 255 };
        case ResourceType::Metal:  return { 180, 180, 200, 255 };
        default:                   return BLANK;
    }
}

static const char* resource_label(ResourceType r) {
    switch (r) {
        case ResourceType::Fruit:  return "Fr";
        case ResourceType::Game:   return "Gm";
        case ResourceType::Metal:  return "Me";
        default:                   return nullptr;
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
            snprintf(buf, size, "Move (%d,%d) -> (%d,%d)", fx, fy, tx, ty);
            break;
        case ActionType::Attack:
            to_coords(a.from, fx, fy); to_coords(a.to, tx, ty);
            snprintf(buf, size, "Attack (%d,%d) -> (%d,%d)", fx, fy, tx, ty);
            break;
        case ActionType::TrainUnit:
            to_coords(a.from, fx, fy);
            snprintf(buf, size, "Train unit at (%d,%d)", fx, fy);
            break;
        case ActionType::ResearchTech:
            snprintf(buf, size, "Research tech %d", a.param);
            break;
        case ActionType::CaptureCity:
            to_coords(a.to, tx, ty);
            snprintf(buf, size, "Capture city at (%d,%d)", tx, ty);
            break;
        case ActionType::BuildImprovement:
            to_coords(a.to, tx, ty);
            snprintf(buf, size, "Build at (%d,%d)", tx, ty);
            break;
        case ActionType::HarvestResource: {
            to_coords(a.to, tx, ty);
            const char* rname = (a.param == (int)ResourceType::Fruit) ? "Fruit" : "Game";
            snprintf(buf, size, "Harvest %s (%d,%d)", rname, tx, ty);
            break;
        }
        default:
            snprintf(buf, size, "Unknown");
            break;
    }
}

int main() {
    GameState s = make_game();
    ViewMode  view = ViewMode::Omni;

    Action actions[256];
    int    action_count = 0;
    legal_actions(s, actions, action_count);

    InitWindow(W, H, "Polyshark");
    SetTargetFPS(60);

    while (!WindowShouldClose()) {

        if (IsKeyPressed(KEY_TAB)) {
            view = (view == ViewMode::Omni) ? ViewMode::P0  :
                   (view == ViewMode::P0)   ? ViewMode::P1  : ViewMode::Omni;
        }

        BeginDrawing();
        ClearBackground({ 30, 30, 30, 255 });

        // Precompute border ownership for every tile (-1 = no city)
        int border_owner[MAP_TILES];
        for (int i = 0; i < MAP_TILES; i++) border_owner[i] = -1;
        for (int i = 0; i < MAP_TILES; i++) {
            const Tile& ct = s.tile_at(i);
            if (!ct.has_city()) continue;
            const City& city = s.get_city(ct.city_id());
            int ccx, ccy;
            to_coords(i, ccx, ccy);
            int r = city.border_radius();
            for (int dy = -r; dy <= r; dy++)
                for (int dx = -r; dx <= r; dx++)
                    if (in_bounds(ccx + dx, ccy + dy))
                        border_owner[to_index(ccx + dx, ccy + dy)] = city.owner();
        }

        // --- Row / col indices ---
        for (int x = 0; x < MAP_SIZE; x++)
            DrawText(TextFormat("%d", x),
                     PAD + LABEL + x * TILE + TILE / 2 - 4, PAD + 2, 14, GRAY);
        for (int y = 0; y < MAP_SIZE; y++)
            DrawText(TextFormat("%d", y),
                     PAD + 2, PAD + LABEL + y * TILE + TILE / 2 - 7, 14, GRAY);

        // --- Map ---
        int vp = (view == ViewMode::P1) ? 1 : 0;

        for (int y = 0; y < MAP_SIZE; y++) {
            for (int x = 0; x < MAP_SIZE; x++) {
                const Tile& t = s.tile_at(to_index(x, y));
                int px  = PAD + LABEL + x * TILE;
                int py  = PAD + LABEL + y * TILE;
                int idx = to_index(x, y);

                bool fogged = (view != ViewMode::Omni) && !s.is_explored(vp, idx);
                bool dimmed = (view != ViewMode::Omni) && s.is_explored(vp, idx)
                                                       && !s.is_visible(vp, idx);

                // Fogged tiles: solid black, no info
                if (fogged) {
                    DrawRectangle(px, py, TILE - 1, TILE - 1, { 18, 18, 18, 255 });
                } else {
                    DrawRectangle(px, py, TILE - 1, TILE - 1, terrain_color(t.terrain()));
                }

                // Resources only when fully visible
                if (!fogged && !dimmed && t.resource() != ResourceType::None) {
                    DrawRectangle(px + 2, py + TILE - 12, 10, 10,
                                  resource_color(t.resource()));
                    DrawText(resource_label(t.resource()), px + 3, py + TILE - 12, 8, WHITE);
                }

                // City ring when explored
                if (!fogged && t.has_city()) {
                    DrawRectangleLinesEx(
                        { (float)px + 2, (float)py + 2, (float)TILE - 5, (float)TILE - 5 },
                        3, WHITE);
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
                        DrawCircle(cx, cy, TILE / 4, uc);
                        DrawText("W", cx - 4, cy - 5, 10, WHITE);
                    }
                }

                // Explored-but-not-visible dimming
                if (dimmed)
                    DrawRectangle(px, py, TILE - 1, TILE - 1, { 0, 0, 0, 120 });

                DrawRectangleLines(px, py, TILE, TILE, { 0, 0, 0, 60 });
            }
        }

        // Border edge lines — draw on tile edges that are on the territory boundary
        for (int y = 0; y < MAP_SIZE; y++) {
            for (int x = 0; x < MAP_SIZE; x++) {
                int idx   = to_index(x, y);
                int owner = border_owner[idx];
                if (owner == -1) continue;
                if ((view != ViewMode::Omni) && !s.is_explored(vp, idx)) continue;

                int px = PAD + LABEL + x * TILE;
                int py = PAD + LABEL + y * TILE;
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

        // --- Sidebar ---
        DrawRectangle(MAP_PX, 0, SIDEBAR, H, { 38, 38, 38, 255 });
        DrawLine(MAP_PX, 0, MAP_PX, H, { 65, 65, 65, 255 });
        DrawText("LEGAL ACTIONS", MAP_PX + 8, 10, 14, GRAY);
        DrawLine(MAP_PX + 4, 28, MAP_PX + SIDEBAR - 4, 28, { 60, 60, 60, 255 });

        Vector2 mouse   = GetMousePosition();
        bool    clicked = IsMouseButtonPressed(MOUSE_LEFT_BUTTON);
        int     applied = -1;

        for (int i = 0; i < action_count; i++) {
            int iy       = 36 + i * 28;
            Rectangle row = { (float)MAP_PX + 4, (float)iy, (float)SIDEBAR - 8, 24 };
            bool hovered  = CheckCollisionPointRec(mouse, row);

            DrawRectangleRec(row, hovered ? Color{ 65, 90, 140, 255 }
                                          : Color{ 50, 50, 50,  255 });

            char label[64];
            action_label(actions[i], label, sizeof(label));
            DrawText(label, MAP_PX + 10, iy + 5, 13, WHITE);

            if (hovered && clicked)
                applied = i;
        }

        // Apply after the loop so we don't mutate actions[] mid-render
        if (applied >= 0) {
            s = apply_action(std::move(s), actions[applied]);
            legal_actions(s, actions, action_count);
        }

        // --- HUD ---
        int hy = H - HUD + 8;
        const char* vname = (view == ViewMode::Omni) ? "Omniscient" :
                            (view == ViewMode::P0)   ? "Player 0"   : "Player 1";
        DrawText(TextFormat("View: %-12s [Tab]", vname), PAD, hy, 17, LIGHTGRAY);
        DrawText(TextFormat("Turn %d  |  P%d to move  |  Stars:  P0=%d  P1=%d",
                 s.get_turn(), s.current_player(), s.get_stars(0), s.get_stars(1)),
                 PAD, hy + 20, 16, LIGHTGRAY);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
