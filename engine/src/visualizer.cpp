#include "raylib.h"
#include "init.h"
#include "unit_def.h"

static constexpr int TILE  = 56;
static constexpr int PAD   = 12;
static constexpr int W     = MAP_SIZE * TILE + PAD * 2;
static constexpr int H     = MAP_SIZE * TILE + PAD * 2;

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

int main() {
    GameState s = make_game();

    InitWindow(W, H, "Polyshark");
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground({ 30, 30, 30, 255 });

        for (int y = 0; y < MAP_SIZE; y++) {
            for (int x = 0; x < MAP_SIZE; x++) {
                const Tile& t = s.tile_at(to_index(x, y));
                int px = PAD + x * TILE;
                int py = PAD + y * TILE;

                // Terrain base
                DrawRectangle(px, py, TILE - 1, TILE - 1, terrain_color(t.terrain()));

                // Resource dot + label (bottom-left corner)
                if (t.resource() != ResourceType::None) {
                    DrawRectangle(px + 2, py + TILE - 12, 10, 10,
                                  resource_color(t.resource()));
                    DrawText(resource_label(t.resource()), px + 3, py + TILE - 12, 8, WHITE);
                }

                // City ring
                if (t.has_city()) {
                    DrawRectangleLinesEx({ (float)px + 2, (float)py + 2,
                                          (float)TILE - 5, (float)TILE - 5 }, 3, WHITE);
                }

                // Unit circle — blue=P0, red=P1
                if (t.has_unit()) {
                    const Unit& u = s.get_unit(t.unit_id());
                    int cx = px + TILE / 2;
                    int cy = py + TILE / 2;
                    Color uc = (u.owner() == 0) ? BLUE : RED;
                    DrawCircle(cx, cy, TILE / 4, uc);
                    DrawText("W", cx - 4, cy - 5, 10, WHITE);
                }

                // Grid line
                DrawRectangleLines(px, py, TILE, TILE, { 0, 0, 0, 60 });
            }
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
