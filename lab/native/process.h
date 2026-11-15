// A bot subprocess speaking newline-delimited JSON over pipes (POSIX).
#pragma once

#include <signal.h>
#include <sys/types.h>

#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace lab {

// Set from a SIGTERM handler; exchange() then returns Closed promptly.
extern volatile sig_atomic_t g_terminate;

class BotProcess {
public:
    enum class Read { Line, Timeout, Closed };

    ~BotProcess();

    // Starts argv in its own process group; stderr (and non-JSON stdout) go to log_path.
    bool start(const std::vector<std::string>& argv, const std::string& cwd,
               const std::map<std::string, std::string>& env, const std::string& log_path,
               size_t log_cap, std::string& error);

    // Queues `line` (newline appended) and pumps I/O until a JSON line arrives,
    // the deadline passes or the process closes stdout. Non-JSON stdout lines
    // are logged and skipped.
    Read exchange(const std::string& line, int timeout_ms, std::string& reply);
    Read receive(int timeout_ms, std::string& reply) { return exchange(std::string(), timeout_ms, reply); }

    void log(const std::string& text);
    void shutdown(const std::string& farewell, int grace_ms);
    bool running() const { return pid_ > 0; }
    int  exit_status() const { return status_; }

private:
    void drain_stderr();
    void close_fd(int& fd);

    pid_t       pid_     = -1;
    int         in_fd_   = -1;
    int         out_fd_  = -1;
    int         err_fd_  = -1;
    int         status_  = -1;
    std::string out_buf_;
    std::FILE*  log_     = nullptr;
    size_t      logged_  = 0;
    size_t      log_cap_ = 0;
    bool        truncated_ = false;
};

}  // namespace lab
