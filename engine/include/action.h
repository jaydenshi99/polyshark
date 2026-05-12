#pragma once

enum class ActionType {
    Move,
    Attack,
    TrainUnit,
    BuildImprovement,
    ResearchTech,
    CaptureCity,
    EndTurn,
};

struct Action {
    ActionType type;
    int from;   // tile index (where relevant)
    int to;     // tile index (where relevant)
    int param;  // unit type, tech id, improvement type, etc.
};
