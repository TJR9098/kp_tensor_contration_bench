// Diagnostic-only BLAS boundary and OMPT observer. Never loaded in timed runs.
#include <omp-tools.h>
#include <atomic>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
std::atomic<int> enabled{0}, ready{0}, calls{0}, outer_calls{0};
std::atomic<int> local_min{INT_MAX}, local_max{0}, active_level_max{0};
std::atomic<int> tid_slots[256]{};
std::atomic<int> regions[4]{}, max_team[4]{}, implicit_tasks[4]{};
std::atomic<int> callback_parallel{0}, callback_implicit{0};
void maximum(std::atomic<int>& dst, int v) {
    int old = dst.load();
    while (old < v && !dst.compare_exchange_weak(old, v)) {}
}
void minimum(std::atomic<int>& dst, int v) {
    int old = dst.load();
    while (old > v && !dst.compare_exchange_weak(old, v)) {}
}
void parallel_begin(ompt_data_t*, const ompt_frame_t*, ompt_data_t* data,
                    unsigned, int, const void* code) {
    int origin = 3;
    Dl_info info{};
    if (code && dladdr(code, &info) && info.dli_fname) {
        if (std::strstr(info.dli_fname, "libkblas")) origin = 1;
        else if (std::strstr(info.dli_fname, "libtorch_cpu")) origin = 2;
    }
    if (data) data->value = static_cast<unsigned>(origin);
    if (enabled.load()) ++regions[origin];
}
void implicit_task(ompt_scope_endpoint_t endpoint, ompt_data_t* parallel,
                   ompt_data_t*, unsigned actual, unsigned, int) {
    if (!enabled.load() || endpoint != ompt_scope_begin) return;
    const auto origin = parallel ? parallel->value : 3;
    if (origin > 3) return;
    ++implicit_tasks[origin];
    maximum(max_team[origin], actual);
}
int initialize(ompt_function_lookup_t lookup, int, ompt_data_t*) {
    auto set = reinterpret_cast<ompt_set_callback_t>(lookup("ompt_set_callback"));
    if (!set) return 0;
    callback_parallel = set(ompt_callback_parallel_begin, reinterpret_cast<ompt_callback_t>(parallel_begin));
    callback_implicit = set(ompt_callback_implicit_task, reinterpret_cast<ompt_callback_t>(implicit_task));
    ready = 1;
    return 1;
}
void finalize(ompt_data_t*) {}
}
extern "C" {
ompt_start_tool_result_t* ompt_start_tool(unsigned, const char*) {
    static ompt_start_tool_result_t result{initialize, finalize, {0}};
    return &result;
}
void kml_batch_observer_begin() {
    enabled = 0; calls = 0; outer_calls = 0; local_min = INT_MAX; local_max = 0; active_level_max = 0;
    for (auto& x : tid_slots) x = 0;
    for (int i=0;i<4;++i) { regions[i]=0; max_team[i]=0; implicit_tasks[i]=0; }
    enabled = 1;
}
void kml_batch_observer_end() { enabled = 0; }
int kml_batch_observer_get(int field) {
    switch(field) {
    case 0: return ready;
    case 1: return calls;
    case 2: { int n=0; for (auto& x:tid_slots) if(x.load()) ++n; return n; }
    case 3: return outer_calls;
    case 4: return local_min.load()==INT_MAX ? -1 : local_min.load();
    case 5: return local_max;
    case 6: return active_level_max;
    case 7: return regions[1];
    case 8: return max_team[1];
    case 9: return implicit_tasks[1];
    case 10: return regions[2];
    case 11: return max_team[2];
    case 12: return regions[3];
    case 13: return max_team[3];
    case 14: return callback_parallel;
    case 15: return callback_implicit;
    default: return -1;
    }
}
void sgemm_(const char* ta, const char* tb, const int* m, const int* n, const int* k,
            const float* alpha, const float* a, const int* lda, const float* b,
            const int* ldb, const float* beta, float* c, const int* ldc) {
    using gemm_t = void (*)(const char*,const char*,const int*,const int*,const int*,
                           const float*,const float*,const int*,const float*,const int*,
                           const float*,float*,const int*);
    static auto next = reinterpret_cast<gemm_t>(dlsym(RTLD_NEXT, "sgemm_"));
    static auto get_local = reinterpret_cast<int(*)()>(dlsym(RTLD_DEFAULT, "BlasGetNumThreadsLocal"));
    static auto in_parallel = reinterpret_cast<int(*)()>(dlsym(RTLD_DEFAULT, "omp_in_parallel"));
    static auto get_level = reinterpret_cast<int(*)()>(dlsym(RTLD_DEFAULT, "omp_get_active_level"));
    if (!next) std::abort();
    if (enabled.load()) {
        ++calls;
        const int tid = static_cast<int>(syscall(SYS_gettid));
        for (auto& slot:tid_slots) {
            int old = slot.load();
            if (old==tid) break;
            if (old==0 && slot.compare_exchange_strong(old,tid)) break;
            if (old==tid) break;
        }
        if (in_parallel && in_parallel()) ++outer_calls;
        if (get_local) { const int nlocal=get_local(); minimum(local_min,nlocal); maximum(local_max,nlocal); }
        if (get_level) maximum(active_level_max,get_level());
    }
    next(ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}
}
