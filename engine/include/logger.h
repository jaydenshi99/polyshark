#pragma once

#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <cstring>

// Engine-pure RGBA byte colour. Layout matches raylib's `Color` so the
// visualizer can construct one directly from its existing colour constants.
struct LogColor {
    uint8_t r, g, b, a;
};

inline constexpr LogColor LOG_COLOR_GREY = { 160, 160, 160, 255 };

class Logger {
public:
    static constexpr int MAX_ENTRIES = 512;
    static constexpr int ENTRY_LEN   = 128;

    static bool debugEnabled;

    struct Entry {
        char     text[ENTRY_LEN];
        LogColor color;
    };

    // Circular log buffer
    static Entry entries[MAX_ENTRIES];
    static int   count;   // total entries ever written (not clamped)

    static void print(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        vprint_impl(LOG_COLOR_GREY, fmt, args);
        va_end(args);
    }

    static void print(LogColor color, const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        vprint_impl(color, fmt, args);
        va_end(args);
    }

    static void clear() { count = 0; }

    // Returns the i-th most recent entry (0 = newest).
    // Returns nullptr if i >= available entries.
    static const char* get(int i) {
        int slot = slot_for(i);
        return slot < 0 ? nullptr : entries[slot].text;
    }

    static LogColor get_color(int i) {
        int slot = slot_for(i);
        return slot < 0 ? LOG_COLOR_GREY : entries[slot].color;
    }

    static int size() { return count < MAX_ENTRIES ? count : MAX_ENTRIES; }

private:
    static int slot_for(int i) {
        int available = count < MAX_ENTRIES ? count : MAX_ENTRIES;
        if (i < 0 || i >= available) return -1;
        return ((count - 1 - i) % MAX_ENTRIES + MAX_ENTRIES) % MAX_ENTRIES;
    }

    static void vprint_impl(LogColor color, const char* fmt, va_list args) {
        if (!debugEnabled) return;
        Entry& e = entries[count % MAX_ENTRIES];
        vsnprintf(e.text, ENTRY_LEN, fmt, args);
        e.color = color;
        count++;
        printf("%s\n", e.text);
    }
};

inline bool          Logger::debugEnabled              = false;
inline int           Logger::count                     = 0;
inline Logger::Entry Logger::entries[Logger::MAX_ENTRIES] = {};
