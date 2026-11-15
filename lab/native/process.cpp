#include "process.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace lab {

volatile sig_atomic_t g_terminate = 0;

using Clock = std::chrono::steady_clock;

static int64_t ms_until(Clock::time_point deadline) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
}

BotProcess::~BotProcess() { shutdown(std::string(), 0); }

void BotProcess::close_fd(int& fd) {
    if (fd >= 0) ::close(fd);
    fd = -1;
}

bool BotProcess::start(const std::vector<std::string>& argv, const std::string& cwd,
                       const std::map<std::string, std::string>& env, const std::string& log_path,
                       size_t log_cap, std::string& error) {
    if (argv.empty()) { error = "empty command"; return false; }
    log_cap_ = log_cap;
    if (!log_path.empty()) log_ = std::fopen(log_path.c_str(), "w");

    int in_p[2], out_p[2], err_p[2];
    if (pipe(in_p) || pipe(out_p) || pipe(err_p)) { error = std::strerror(errno); return false; }

    // Everything the child needs is prepared before fork (no allocation after it).
    std::vector<char*> args;
    for (const auto& a : argv) args.push_back(const_cast<char*>(a.c_str()));
    args.push_back(nullptr);
    std::vector<std::string> env_kv;
    for (const auto& kv : env) env_kv.push_back(kv.first + "=" + kv.second);

    pid_t pid = fork();
    if (pid < 0) { error = std::strerror(errno); return false; }
    if (pid == 0) {
        setpgid(0, 0);
        dup2(in_p[0], STDIN_FILENO);
        dup2(out_p[1], STDOUT_FILENO);
        dup2(err_p[1], STDERR_FILENO);
        for (int fd : { in_p[0], in_p[1], out_p[0], out_p[1], err_p[0], err_p[1] }) ::close(fd);
        if (!cwd.empty() && chdir(cwd.c_str()) != 0) {
            std::fprintf(stderr, "lab: cannot enter %s: %s\n", cwd.c_str(), std::strerror(errno));
            _exit(127);
        }
        for (auto& kv : env_kv) putenv(const_cast<char*>(kv.c_str()));
        signal(SIGPIPE, SIG_DFL);
        execvp(args[0], args.data());
        std::fprintf(stderr, "lab: cannot run %s: %s\n", args[0], std::strerror(errno));
        _exit(127);
    }
    setpgid(pid, pid);
    pid_ = pid;
    ::close(in_p[0]);
    ::close(out_p[1]);
    ::close(err_p[1]);
    in_fd_ = in_p[1];
    out_fd_ = out_p[0];
    err_fd_ = err_p[0];
    for (int fd : { in_fd_, out_fd_, err_fd_ }) {
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
        fcntl(fd, F_SETFD, FD_CLOEXEC);
    }
    return true;
}

void BotProcess::log(const std::string& text) {
    if (!log_ || truncated_) return;
    if (logged_ + text.size() > log_cap_) {
        size_t room = log_cap_ > logged_ ? log_cap_ - logged_ : 0;
        std::fwrite(text.data(), 1, room, log_);
        std::fputs("\n[lab: log truncated]\n", log_);
        truncated_ = true;
    } else {
        std::fwrite(text.data(), 1, text.size(), log_);
    }
    logged_ += text.size();
    std::fflush(log_);
}

void BotProcess::drain_stderr() {
    if (err_fd_ < 0) return;
    char buf[8192];
    for (;;) {
        ssize_t n = ::read(err_fd_, buf, sizeof buf);
        if (n > 0) { log(std::string(buf, (size_t)n)); continue; }
        if (n == 0) close_fd(err_fd_);
        return;
    }
}

BotProcess::Read BotProcess::exchange(const std::string& line, int timeout_ms, std::string& reply) {
    std::string pending = line.empty() ? std::string() : line + "\n";
    size_t written = 0;
    auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    bool out_eof = out_fd_ < 0;

    for (;;) {
        size_t nl;
        while ((nl = out_buf_.find('\n')) != std::string::npos) {
            std::string got = out_buf_.substr(0, nl);
            out_buf_.erase(0, nl + 1);
            if (!got.empty() && got.back() == '\r') got.pop_back();
            size_t first = got.find_first_not_of(" \t");
            if (first != std::string::npos && got[first] == '{') { reply = got; return Read::Line; }
            if (!got.empty()) log("[stdout] " + got + "\n");
        }
        if (out_eof || g_terminate) return Read::Closed;
        int64_t left = ms_until(deadline);
        if (left <= 0) return Read::Timeout;
        if (left > 100) left = 100;  // wake regularly to notice g_terminate

        pollfd fds[3];
        int n = 0, in_i = -1, out_i = -1, err_i = -1;
        if (written < pending.size() && in_fd_ >= 0) { in_i = n; fds[n++] = { in_fd_, POLLOUT, 0 }; }
        out_i = n; fds[n++] = { out_fd_, POLLIN, 0 };
        if (err_fd_ >= 0) { err_i = n; fds[n++] = { err_fd_, POLLIN, 0 }; }
        int rc = poll(fds, (nfds_t)n, (int)left);
        if (rc < 0) {
            if (errno == EINTR) continue;
            return Read::Closed;
        }
        if (in_i >= 0 && fds[in_i].revents) {
            ssize_t w = ::write(in_fd_, pending.data() + written, pending.size() - written);
            if (w > 0) written += (size_t)w;
            else if (w < 0 && errno != EAGAIN && errno != EINTR) { close_fd(in_fd_); written = pending.size(); }
        }
        if (err_i >= 0 && fds[err_i].revents) drain_stderr();
        if (fds[out_i].revents) {
            char buf[65536];
            for (;;) {
                ssize_t r = ::read(out_fd_, buf, sizeof buf);
                if (r > 0) { out_buf_.append(buf, (size_t)r); continue; }
                if (r == 0) out_eof = true;
                break;
            }
        }
    }
}

void BotProcess::shutdown(const std::string& farewell, int grace_ms) {
    if (pid_ <= 0) {
        if (log_) { std::fclose(log_); log_ = nullptr; }
        return;
    }
    if (!farewell.empty() && in_fd_ >= 0) {
        std::string msg = farewell + "\n";
        ssize_t w = ::write(in_fd_, msg.data(), msg.size());
        (void)w;
    }
    close_fd(in_fd_);
    auto deadline = Clock::now() + std::chrono::milliseconds(grace_ms);
    int status = 0;
    pid_t done = 0;
    for (;;) {
        done = waitpid(pid_, &status, WNOHANG);
        if (done == pid_ || done < 0) break;
        // Keep the pipes drained so a chatty bot can still exit.
        char buf[8192];
        if (out_fd_ >= 0) while (::read(out_fd_, buf, sizeof buf) > 0) {}
        drain_stderr();
        if (ms_until(deadline) <= 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    kill(-pid_, SIGKILL);
    if (done != pid_ && done >= 0) waitpid(pid_, &status, 0);
    status_ = status;
    drain_stderr();
    close_fd(out_fd_);
    close_fd(err_fd_);
    pid_ = -1;
    if (log_) { std::fclose(log_); log_ = nullptr; }
}

}  // namespace lab
