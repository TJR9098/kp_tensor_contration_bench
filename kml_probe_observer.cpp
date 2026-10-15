// Optional OMPT diagnostics; counters are enabled only around a diagnostic call.
#include <omp-tools.h>
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <execinfo.h>
#include <unistd.h>

namespace {
std::atomic<int> enabled{0}, initialized{0}, parallel_events{0};
std::atomic<int> requested_team{0}, actual_team{0}, implicit_tasks{0};
std::atomic<int> parallel_callback_status{0}, implicit_callback_status{0};
void maximum(std::atomic<int>& value, int next) {
    int old = value.load();
    while (old < next && !value.compare_exchange_weak(old, next)) {}
}
void parallel_begin(ompt_data_t*, const ompt_frame_t*, ompt_data_t*,
                    unsigned requested, int, const void*) {
    if (!enabled.load()) return;
    ++parallel_events;
    maximum(requested_team, static_cast<int>(requested));
}
void implicit_task(ompt_scope_endpoint_t endpoint, ompt_data_t*, ompt_data_t*,
                   unsigned actual, unsigned, int) {
    if (!enabled.load() || endpoint != ompt_scope_begin) return;
    ++implicit_tasks;
    maximum(actual_team, static_cast<int>(actual));
}
int initialize(ompt_function_lookup_t lookup, int, ompt_data_t*) {
    auto set = reinterpret_cast<ompt_set_callback_t>(lookup("ompt_set_callback"));
    if (!set) return 0;
    parallel_callback_status = set(ompt_callback_parallel_begin,
        reinterpret_cast<ompt_callback_t>(parallel_begin));
    implicit_callback_status = set(ompt_callback_implicit_task,
        reinterpret_cast<ompt_callback_t>(implicit_task));
    initialized = 1;
    return 1;
}
void finalize(ompt_data_t*) {}
// Only used in isolated crash diagnostics, never in measured runs.
void fault(int signal) {
    const char message[] = "KML probe native backtrace (diagnostic process):\n";
    write(STDERR_FILENO, message, sizeof(message) - 1);
    void* frames[64];
    int count = backtrace(frames, 64);
    backtrace_symbols_fd(frames, count, STDERR_FILENO);
    _exit(128 + signal);
}
}
extern "C" {
ompt_start_tool_result_t* ompt_start_tool(unsigned, const char*) {
    static ompt_start_tool_result_t result{initialize, finalize, {0}};
    return &result;
}
void kml_probe_begin() {
    enabled = 0;
    parallel_events = 0; requested_team = 0; actual_team = 0; implicit_tasks = 0;
    enabled = 1;
}
void kml_probe_end() { enabled = 0; }
int kml_probe_get(int field) {
    switch (field) {
    case 0: return initialized;
    case 1: return parallel_events;
    case 2: return requested_team;
    case 3: return actual_team;
    case 4: return implicit_tasks;
    case 5: return parallel_callback_status;
    case 6: return implicit_callback_status;
    default: return -1;
    }
}
void kml_probe_fault_handler() {
    std::signal(SIGSEGV, fault);
    std::signal(SIGILL, fault);
}
}
