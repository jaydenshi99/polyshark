#pragma once

#include <cstdio>
#include <cstdarg>
#include <cstring>

class Logger {
public:
    static constexpr int MAX_ENTRIES = 512;
    static constexpr int ENTRY_LEN   = 128;

    static bool debugEnabled;

    // Circular log buffer
    static char entries[MAX_ENTRIES][ENTRY_LEN];
    static int  count;   // total entries ever written (not clamped)

    static void print(const char* fmt, ...) {
        if (!debugEnabled) return;
        char buf[ENTRY_LEN];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, ENTRY_LEN, fmt, args);
        va_end(args);
        // Write into circular slot
        int slot = count % MAX_ENTRIES;
        strncpy(entries[slot], buf, ENTRY_LEN - 1);
        entries[slot][ENTRY_LEN - 1] = '\0';
        count++;
        printf("%s\n", buf);
    }

    static void clear() { count = 0; }

    // Returns the i-th most recent entry (0 = newest).
    // Returns nullptr if i >= available entries.
    static const char* get(int i) {
        int available = count < MAX_ENTRIES ? count : MAX_ENTRIES;
        if (i >= available) return nullptr;
        int slot = ((count - 1 - i) % MAX_ENTRIES + MAX_ENTRIES) % MAX_ENTRIES;
        return entries[slot];
    }

    static int size() { return count < MAX_ENTRIES ? count : MAX_ENTRIES; }
};

inline bool Logger::debugEnabled = true;
inline int  Logger::count        = 0;
inline char Logger::entries[Logger::MAX_ENTRIES][Logger::ENTRY_LEN] = {};
