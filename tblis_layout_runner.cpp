#include <tblis.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <openssl/sha.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Row = std::unordered_map<std::string, std::string>;

constexpr int kWarmup = 10;
constexpr int kRepeat = 50;
constexpr std::uint64_t kDiagnosticTargetNs = 250000000ULL;

std::vector<std::string> split(const std::string& text, const char delimiter) {
    std::vector<std::string> result;
    std::string item;
    std::istringstream stream(text);
    while (std::getline(stream, item, delimiter)) {
        if (!item.empty() && item.back() == '\r') item.pop_back();
        result.push_back(item);
    }
    if (!text.empty() && text.back() == delimiter) result.emplace_back();
    return result;
}

Row read_case(const fs::path& path, const std::string& wanted) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open manifest: " + path.string());
    std::string line;
    std::getline(input, line);
    const auto header = split(line, '\t');
    Row found;
    int matches = 0;
    while (std::getline(input, line)) {
        const auto values = split(line, '\t');
        if (values.size() != header.size()) throw std::runtime_error("malformed manifest row");
        Row row;
        for (std::size_t i = 0; i < header.size(); ++i) row[header[i]] = values[i];
        if (row.at("case_id") == wanted) {
            found = std::move(row);
            ++matches;
        }
    }
    if (matches != 1) throw std::runtime_error("case lookup count is not one");
    return found;
}

std::vector<tblis::len_type> parse_shape(const std::string& text) {
    std::vector<tblis::len_type> output;
    for (const auto& value : split(text, ',')) output.push_back(std::stoll(value));
    return output;
}

template <typename T>
std::size_t product(const std::vector<T>& values) {
    std::size_t result = 1;
    for (const auto value : values) result *= static_cast<std::size_t>(value);
    return result;
}

std::vector<tblis::stride_type> contiguous_strides(
    const std::vector<tblis::len_type>& shape) {
    std::vector<tblis::stride_type> strides(shape.size(), 1);
    for (std::size_t i = shape.size(); i-- > 1;) strides[i - 1] = strides[i] * shape[i];
    return strides;
}

float deterministic_value(const std::size_t index, const std::uint64_t salt) {
    const std::uint64_t bits = (index * 1103515245ULL + salt) & 0xFFFFULL;
    return static_cast<float>(bits) / 32768.0F - 1.0F;
}

std::string sha256(const std::vector<float>& values) {
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256(reinterpret_cast<const unsigned char*>(values.data()),
           values.size() * sizeof(float), digest.data());
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const auto byte : digest) output << std::setw(2) << static_cast<unsigned>(byte);
    return output.str();
}

struct Operand {
    std::string labels;
    std::string physical_order;
    std::vector<tblis::len_type> shape;
    std::vector<tblis::stride_type> strides;
    std::vector<float> storage;
    std::string logical_sha256;
};

Operand make_operand(const Row& item, const bool is_a) {
    const std::string labels = is_a ? "abkl" : "klcd";
    const std::string physical_order = item.at(is_a ? "a_physical_order" : "b_physical_order");
    const auto logical_shape = parse_shape(item.at(is_a ? "a_shape" : "b_shape"));
    if (labels.size() != logical_shape.size() || physical_order.size() != logical_shape.size())
        throw std::runtime_error("operand rank mismatch");
    std::map<char, tblis::len_type> size;
    for (std::size_t i = 0; i < labels.size(); ++i) size[labels[i]] = logical_shape[i];
    std::vector<tblis::len_type> backing_shape;
    for (const char label : physical_order) backing_shape.push_back(size.at(label));
    char padded_label = '\0';
    int padding_step = 1;
    const auto padding = split(item.at("padding"), ':');
    if (padding.size() == 3 && padding[0] == (is_a ? "A" : "B")) {
        padded_label = padding[1].at(0);
        padding_step = std::stoi(padding[2]);
        backing_shape[physical_order.find(padded_label)] *= padding_step;
    }
    const auto backing_strides = contiguous_strides(backing_shape);
    std::map<char, tblis::stride_type> stride;
    for (std::size_t i = 0; i < physical_order.size(); ++i) {
        stride[physical_order[i]] = backing_strides[i] *
            (physical_order[i] == padded_label ? padding_step : 1);
    }
    std::vector<tblis::stride_type> logical_strides;
    for (const char label : labels) logical_strides.push_back(stride.at(label));
    std::vector<float> storage(product(backing_shape), 0.0F);
    std::vector<float> canonical(product(logical_shape));
    std::vector<tblis::len_type> coordinates(logical_shape.size());
    const std::uint64_t salt = is_a ? 12345 : 54321;
    for (std::size_t logical = 0; logical < canonical.size(); ++logical) {
        std::size_t value = logical;
        std::size_t offset = 0;
        for (std::size_t i = logical_shape.size(); i-- > 0;) {
            coordinates[i] = static_cast<tblis::len_type>(value % logical_shape[i]);
            value /= static_cast<std::size_t>(logical_shape[i]);
        }
        for (std::size_t i = 0; i < coordinates.size(); ++i)
            offset += static_cast<std::size_t>(coordinates[i] * logical_strides[i]);
        canonical[logical] = deterministic_value(logical, salt);
        storage[offset] = canonical[logical];
    }
    return {
        labels, physical_order, logical_shape, logical_strides, std::move(storage),
        sha256(canonical),
    };
}

std::vector<tblis::label_type> labels(const std::string& text) {
    std::vector<tblis::label_type> output;
    for (const char value : text) output.push_back(static_cast<tblis::label_type>(value));
    return output;
}

std::uint64_t elapsed_ns(const Clock::time_point begin) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
}

std::uint64_t process_cpu_ns() {
    timespec value{};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0)
        throw std::runtime_error("clock_gettime failed");
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL + value.tv_nsec;
}

std::string affinity_text(const pid_t tid = 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(tid, sizeof(set), &set) != 0)
        throw std::runtime_error("sched_getaffinity failed");
    std::ostringstream output;
    bool first = true;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &set)) continue;
        if (!first) output << ',';
        output << cpu;
        first = false;
    }
    return output.str();
}

std::vector<int> affinity_cpus(const int threads) {
    if (threads == 1) return {3};
    if (threads == 2) return {3, 2};
    if (threads == 4) return {3, 2, 1, 0};
    if (threads == 8) return {3, 2, 1, 0, 4, 5, 6, 7};
    if (threads == 32) {
        std::vector<int> cpus = {3, 2, 1, 0};
        for (int cpu = 4; cpu < 32; ++cpu) cpus.push_back(cpu);
        return cpus;
    }
    throw std::runtime_error("unsupported thread count");
}

void set_affinity(const int threads) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (const int cpu : affinity_cpus(threads)) CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0)
        throw std::runtime_error("sched_setaffinity failed");
}

struct Record {
    std::uint64_t total_ns = 0;
    std::uint64_t cpu_ns = 0;
    std::uint64_t transform_ns = 0;
    std::uint64_t packing_ns = 0;
    std::uint64_t kernel_ns = 0;
    std::uint64_t restore_ns = 0;
};

Record execute(const Operand& a, const Operand& b, std::vector<float>& c,
               const std::vector<tblis::len_type>& c_shape,
               const std::vector<tblis::stride_type>& c_strides,
               const std::vector<tblis::label_type>& a_labels,
               const std::vector<tblis::label_type>& b_labels,
               const std::vector<tblis::label_type>& c_labels) {
    tblis_profile_reset();
    const auto cpu_begin = process_cpu_ns();
    const auto total_begin = Clock::now();
    auto a_shape = a.shape;
    auto b_shape = b.shape;
    auto mutable_a_strides = a.strides;
    auto mutable_b_strides = b.strides;
    auto mutable_c_shape = c_shape;
    auto mutable_c_strides = c_strides;
    tblis::tblis_tensor a_tensor{}, b_tensor{}, c_tensor{};
    tblis::tblis_init_tensor_scaled_s(
        &a_tensor, 1.0F, static_cast<int>(a_shape.size()), a_shape.data(),
        const_cast<float*>(a.storage.data()), mutable_a_strides.data());
    tblis::tblis_init_tensor_scaled_s(
        &b_tensor, 1.0F, static_cast<int>(b_shape.size()), b_shape.data(),
        const_cast<float*>(b.storage.data()), mutable_b_strides.data());
    tblis::tblis_init_tensor_scaled_s(
        &c_tensor, 0.0F, static_cast<int>(mutable_c_shape.size()),
        mutable_c_shape.data(), c.data(), mutable_c_strides.data());
    tblis::tblis_tensor_mult(nullptr, nullptr, &a_tensor, a_labels.data(),
                             &b_tensor, b_labels.data(), &c_tensor, c_labels.data());
    Record record;
    record.total_ns = elapsed_ns(total_begin);
    record.cpu_ns = process_cpu_ns() - cpu_begin;
    tblis_profile profile{};
    tblis_profile_get(&profile);
    record.transform_ns = profile.transform_ns;
    record.packing_ns = profile.packing_ns;
    record.kernel_ns = profile.kernel_ns;
    record.restore_ns = profile.restore_ns;
    return record;
}

std::vector<double> read_reference(const fs::path& path, const std::size_t elements) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open reference: " + path.string());
    std::vector<double> reference(elements);
    input.read(reinterpret_cast<char*>(reference.data()),
               static_cast<std::streamsize>(elements * sizeof(double)));
    if (!input || input.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("reference size mismatch: " + path.string());
    return reference;
}

std::pair<double, double> validate(const std::vector<float>& actual,
                                   const std::vector<double>& reference) {
    if (actual.size() != reference.size()) throw std::runtime_error("output size mismatch");
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error("non-finite output");
        const double error = std::abs(static_cast<double>(actual[i]) - reference[i]);
        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, error / std::max(std::abs(reference[i]), 1.0));
    }
    return {max_abs, max_rel};
}

struct TaskInfo {
    std::uint64_t ticks = 0;
    std::uint64_t runtime_ns = 0;
    std::string name;
    std::string affinity;
};

std::map<pid_t, TaskInfo> task_snapshot() {
    std::map<pid_t, TaskInfo> result;
    for (const auto& entry : fs::directory_iterator("/proc/self/task")) {
        const std::string name = entry.path().filename().string();
        if (!std::all_of(name.begin(), name.end(), ::isdigit)) continue;
        const pid_t tid = static_cast<pid_t>(std::stoi(name));
        std::ifstream stat(entry.path() / "stat");
        std::string line;
        if (!std::getline(stat, line)) continue;
        const auto close = line.rfind(')');
        if (close == std::string::npos) continue;
        const auto fields = split(line.substr(close + 2), ' ');
        if (fields.size() < 13) continue;
        TaskInfo info;
        info.ticks = std::stoull(fields[11]) + std::stoull(fields[12]);
        std::ifstream schedstat(entry.path() / "schedstat");
        if (!(schedstat >> info.runtime_ns)) continue;
        info.name = line.substr(line.find('(') + 1, close - line.find('(') - 1);
        try { info.affinity = affinity_text(tid); }
        catch (const std::exception&) { continue; }
        result[tid] = std::move(info);
    }
    return result;
}

struct WorkerDiagnostic {
    int iterations = 0;
    std::uint64_t wall_ns = 0;
    std::uint64_t cpu_ns = 0;
    std::vector<std::pair<pid_t, TaskInfo>> active;
    std::size_t runtime_advancing_workers = 0;
    std::size_t all_tasks = 0;
};

WorkerDiagnostic diagnose_workers(
    const std::uint64_t median_ns, const Operand& a, const Operand& b,
    std::vector<float>& c, const std::vector<tblis::len_type>& c_shape,
    const std::vector<tblis::stride_type>& c_strides,
    const std::vector<tblis::label_type>& a_labels,
    const std::vector<tblis::label_type>& b_labels,
    const std::vector<tblis::label_type>& c_labels) {
    WorkerDiagnostic result;
    result.iterations = static_cast<int>(std::min<std::uint64_t>(
        20000, std::max<std::uint64_t>(1, (kDiagnosticTargetNs + median_ns - 1) / median_ns)));
    const auto before = task_snapshot();
    std::atomic<bool> stop_monitor{false};
    std::atomic<pid_t> monitor_tid{0};
    std::map<pid_t, TaskInfo> peak_tasks = before;
    std::map<pid_t, TaskInfo> peak_active;
    std::thread monitor([&] {
        const auto tid = static_cast<pid_t>(syscall(SYS_gettid));
        monitor_tid.store(tid, std::memory_order_release);
        while (!stop_monitor.load(std::memory_order_acquire)) {
            auto snapshot = task_snapshot();
            snapshot.erase(tid);
            std::map<pid_t, TaskInfo> active;
            for (const auto& [task_tid, info] : snapshot) {
                const auto previous = before.find(task_tid);
                const std::uint64_t previous_runtime =
                    previous == before.end() ? 0 : previous->second.runtime_ns;
                if (info.runtime_ns > previous_runtime) active.emplace(task_tid, info);
            }
            if (snapshot.size() > peak_tasks.size()) peak_tasks = snapshot;
            if (active.size() > peak_active.size()) peak_active = std::move(active);
            std::this_thread::sleep_for(std::chrono::microseconds(250));
        }
    });
    while (monitor_tid.load(std::memory_order_acquire) == 0) std::this_thread::yield();
    const auto cpu_begin = process_cpu_ns();
    const auto wall_begin = Clock::now();
    for (int i = 0; i < result.iterations; ++i)
        execute(a, b, c, c_shape, c_strides, a_labels, b_labels, c_labels);
    result.wall_ns = elapsed_ns(wall_begin);
    result.cpu_ns = process_cpu_ns() - cpu_begin;
    stop_monitor.store(true, std::memory_order_release);
    monitor.join();
    auto after = task_snapshot();
    std::map<pid_t, TaskInfo> final_active;
    for (const auto& [tid, info] : after) {
        const auto previous = before.find(tid);
        const std::uint64_t previous_runtime =
            previous == before.end() ? 0 : previous->second.runtime_ns;
        if (info.runtime_ns > previous_runtime) final_active.emplace(tid, info);
    }
    if (final_active.size() > peak_active.size()) peak_active = std::move(final_active);
    if (after.size() > peak_tasks.size()) peak_tasks = std::move(after);
    result.runtime_advancing_workers = peak_active.size();
    result.all_tasks = peak_tasks.size();
    for (auto& [tid, info] : peak_tasks) result.active.emplace_back(tid, std::move(info));
    return result;
}

std::uint64_t integer_median(std::vector<std::uint64_t> values) {
    std::sort(values.begin(), values.end());
    return (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2;
}

template <typename T>
std::string vector_json(const std::vector<T>& values) {
    std::ostringstream output;
    output << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) output << ',';
        output << values[i];
    }
    output << ']';
    return output.str();
}

struct BatchedOperand {
    std::string labels;
    std::string physical_order;
    std::vector<tblis::len_type> shape;
    std::vector<tblis::stride_type> strides;
    std::vector<float> storage;
    std::string logical_sha256;
    std::size_t batch = 1;
    std::size_t batch_stride = 0;
};

std::string digest_text(const std::array<unsigned char, SHA256_DIGEST_LENGTH>& digest) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const auto byte : digest) output << std::setw(2) << static_cast<unsigned>(byte);
    return output.str();
}

BatchedOperand make_batched_operand(const Row& item, const bool is_a) {
    const std::string labels_text = is_a ? "abkl" : "klcd";
    const std::string physical_order = item.at(
        is_a ? "a_physical_order" : "b_physical_order");
    const auto logical_shape = parse_shape(item.at(is_a ? "a_shape" : "b_shape"));
    const std::size_t batch = std::stoull(item.at("batch"));
    if (labels_text.size() != logical_shape.size() ||
        physical_order.size() != logical_shape.size())
        throw std::runtime_error("batched operand rank mismatch");
    std::map<char, tblis::len_type> size;
    for (std::size_t i = 0; i < labels_text.size(); ++i)
        size[labels_text[i]] = logical_shape[i];
    std::vector<tblis::len_type> physical_shape;
    for (const char label : physical_order) physical_shape.push_back(size.at(label));
    const auto physical_strides = contiguous_strides(physical_shape);
    std::map<char, tblis::stride_type> stride;
    for (std::size_t i = 0; i < physical_order.size(); ++i)
        stride[physical_order[i]] = physical_strides[i];
    std::vector<tblis::stride_type> logical_strides;
    for (const char label : labels_text) logical_strides.push_back(stride.at(label));
    const std::size_t batch_stride = product(physical_shape);
    const std::size_t logical_elements = product(logical_shape);
    std::vector<float> storage(batch * batch_stride, 0.0F);
    std::vector<tblis::len_type> coordinates(logical_shape.size());
    const std::uint64_t salt = is_a ? 12345 : 54321;
    SHA256_CTX context;
    SHA256_Init(&context);
    for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (std::size_t logical = 0; logical < logical_elements; ++logical) {
            std::size_t value = logical;
            std::size_t offset = batch_index * batch_stride;
            for (std::size_t i = logical_shape.size(); i-- > 0;) {
                coordinates[i] = static_cast<tblis::len_type>(value % logical_shape[i]);
                value /= static_cast<std::size_t>(logical_shape[i]);
            }
            for (std::size_t i = 0; i < coordinates.size(); ++i)
                offset += static_cast<std::size_t>(coordinates[i] * logical_strides[i]);
            const float generated = deterministic_value(
                batch_index * logical_elements + logical, salt);
            storage[offset] = generated;
            SHA256_Update(&context, &generated, sizeof(generated));
        }
    }
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256_Final(digest.data(), &context);
    return {
        labels_text, physical_order, logical_shape, logical_strides,
        std::move(storage), digest_text(digest), batch, batch_stride,
    };
}

void pin_batch_worker(const int requested_threads, const int worker_count,
                      const int worker_index) {
    const auto cpus = affinity_cpus(requested_threads);
    if (worker_count > static_cast<int>(cpus.size()))
        throw std::runtime_error("batch worker count exceeds thread budget");
    const int cpu = cpus[worker_index];
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0)
        throw std::runtime_error("batch worker sched_setaffinity failed");
}

void batched_mult_one(const BatchedOperand& a, const BatchedOperand& b,
                      std::vector<float>& c, const std::size_t batch_index,
                      const std::vector<tblis::len_type>& c_shape,
                      const std::vector<tblis::stride_type>& c_strides,
                      const std::vector<tblis::label_type>& a_labels,
                      const std::vector<tblis::label_type>& b_labels,
                      const std::vector<tblis::label_type>& c_labels) {
    auto a_shape = a.shape;
    auto b_shape = b.shape;
    auto a_strides = a.strides;
    auto b_strides = b.strides;
    auto mutable_c_shape = c_shape;
    auto mutable_c_strides = c_strides;
    const std::size_t c_stride = product(c_shape);
    tblis::tblis_tensor a_tensor{}, b_tensor{}, c_tensor{};
    tblis::tblis_init_tensor_scaled_s(
        &a_tensor, 1.0F, static_cast<int>(a_shape.size()), a_shape.data(),
        const_cast<float*>(a.storage.data()) + batch_index * a.batch_stride,
        a_strides.data());
    tblis::tblis_init_tensor_scaled_s(
        &b_tensor, 1.0F, static_cast<int>(b_shape.size()), b_shape.data(),
        const_cast<float*>(b.storage.data()) + batch_index * b.batch_stride,
        b_strides.data());
    tblis::tblis_init_tensor_scaled_s(
        &c_tensor, 0.0F, static_cast<int>(mutable_c_shape.size()),
        mutable_c_shape.data(), c.data() + batch_index * c_stride,
        mutable_c_strides.data());
    tblis::tblis_tensor_mult(nullptr, nullptr, &a_tensor, a_labels.data(),
                             &b_tensor, b_labels.data(), &c_tensor, c_labels.data());
}

Record execute_batched_intra(
    const BatchedOperand& a, const BatchedOperand& b, std::vector<float>& c,
    const std::vector<tblis::len_type>& c_shape,
    const std::vector<tblis::stride_type>& c_strides,
    const std::vector<tblis::label_type>& a_labels,
    const std::vector<tblis::label_type>& b_labels,
    const std::vector<tblis::label_type>& c_labels) {
    Record record;
    const auto cpu_begin = process_cpu_ns();
    const auto total_begin = Clock::now();
    for (std::size_t batch_index = 0; batch_index < a.batch; ++batch_index) {
        tblis_profile_reset();
        batched_mult_one(
            a, b, c, batch_index, c_shape, c_strides, a_labels, b_labels, c_labels);
        tblis_profile profile{};
        tblis_profile_get(&profile);
        record.transform_ns += profile.transform_ns;
        record.packing_ns += profile.packing_ns;
        record.kernel_ns += profile.kernel_ns;
        record.restore_ns += profile.restore_ns;
    }
    record.total_ns = elapsed_ns(total_begin);
    record.cpu_ns = process_cpu_ns() - cpu_begin;
    return record;
}

class InterBatchExecutor {
public:
    InterBatchExecutor(
        const int requested_threads, const BatchedOperand& a, const BatchedOperand& b,
        std::vector<float>& c, const std::vector<tblis::len_type>& c_shape,
        const std::vector<tblis::stride_type>& c_strides,
        const std::vector<tblis::label_type>& a_labels,
        const std::vector<tblis::label_type>& b_labels,
        const std::vector<tblis::label_type>& c_labels)
        : requested_threads_(requested_threads),
          worker_count_(std::min<int>(requested_threads, static_cast<int>(a.batch))),
          a_(a), b_(b), c_(c), c_shape_(c_shape), c_strides_(c_strides),
          a_labels_(a_labels), b_labels_(b_labels), c_labels_(c_labels),
          tids_(worker_count_, 0) {
        for (int worker = 0; worker < worker_count_; ++worker)
            workers_.emplace_back([this, worker] { worker_loop(worker); });
        while (ready_.load(std::memory_order_acquire) != worker_count_)
            std::this_thread::yield();
    }

    ~InterBatchExecutor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        start_.notify_all();
        for (auto& worker : workers_) worker.join();
    }

    Record run() {
        const auto cpu_begin = process_cpu_ns();
        const auto total_begin = Clock::now();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            completed_ = 0;
            ++generation_;
            start_.notify_all();
            done_.wait(lock, [this] { return completed_ == worker_count_; });
        }
        Record record;
        record.total_ns = elapsed_ns(total_begin);
        record.cpu_ns = process_cpu_ns() - cpu_begin;
        return record;
    }

    int worker_count() const { return worker_count_; }
    const std::vector<pid_t>& tids() const { return tids_; }

private:
    void worker_loop(const int worker_index) {
        pin_batch_worker(requested_threads_, worker_count_, worker_index);
        tids_[worker_index] = static_cast<pid_t>(syscall(SYS_gettid));
        ready_.fetch_add(1, std::memory_order_release);
        std::uint64_t seen_generation = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            start_.wait(lock, [this, seen_generation] {
                return stopping_ || generation_ > seen_generation;
            });
            if (stopping_) return;
            seen_generation = generation_;
            lock.unlock();
            for (std::size_t batch_index = worker_index;
                 batch_index < a_.batch; batch_index += worker_count_) {
                batched_mult_one(
                    a_, b_, c_, batch_index, c_shape_, c_strides_,
                    a_labels_, b_labels_, c_labels_);
            }
            lock.lock();
            if (++completed_ == worker_count_) done_.notify_one();
        }
    }

    int requested_threads_;
    int worker_count_;
    const BatchedOperand& a_;
    const BatchedOperand& b_;
    std::vector<float>& c_;
    const std::vector<tblis::len_type>& c_shape_;
    const std::vector<tblis::stride_type>& c_strides_;
    const std::vector<tblis::label_type>& a_labels_;
    const std::vector<tblis::label_type>& b_labels_;
    const std::vector<tblis::label_type>& c_labels_;
    std::vector<std::thread> workers_;
    std::vector<pid_t> tids_;
    std::atomic<int> ready_{0};
    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable done_;
    std::uint64_t generation_ = 0;
    int completed_ = 0;
    bool stopping_ = false;
};

WorkerDiagnostic diagnose_operation(
    const std::uint64_t median_ns, const std::function<Record()>& operation) {
    WorkerDiagnostic result;
    result.iterations = static_cast<int>(std::min<std::uint64_t>(
        20000, std::max<std::uint64_t>(
            1, (kDiagnosticTargetNs + median_ns - 1) / median_ns)));
    const auto before = task_snapshot();
    std::atomic<bool> stop_monitor{false};
    std::atomic<pid_t> monitor_tid{0};
    std::map<pid_t, TaskInfo> peak_tasks = before;
    std::map<pid_t, TaskInfo> peak_active;
    std::thread monitor([&] {
        const auto tid = static_cast<pid_t>(syscall(SYS_gettid));
        monitor_tid.store(tid, std::memory_order_release);
        while (!stop_monitor.load(std::memory_order_acquire)) {
            auto snapshot = task_snapshot();
            snapshot.erase(tid);
            std::map<pid_t, TaskInfo> active;
            for (const auto& [task_tid, info] : snapshot) {
                const auto previous = before.find(task_tid);
                const std::uint64_t previous_runtime =
                    previous == before.end() ? 0 : previous->second.runtime_ns;
                if (info.runtime_ns > previous_runtime) active.emplace(task_tid, info);
            }
            if (snapshot.size() > peak_tasks.size()) peak_tasks = snapshot;
            if (active.size() > peak_active.size()) peak_active = std::move(active);
            std::this_thread::sleep_for(std::chrono::microseconds(250));
        }
    });
    while (monitor_tid.load(std::memory_order_acquire) == 0) std::this_thread::yield();
    const auto cpu_begin = process_cpu_ns();
    const auto wall_begin = Clock::now();
    for (int iteration = 0; iteration < result.iterations; ++iteration) operation();
    result.wall_ns = elapsed_ns(wall_begin);
    result.cpu_ns = process_cpu_ns() - cpu_begin;
    stop_monitor.store(true, std::memory_order_release);
    monitor.join();
    auto after = task_snapshot();
    for (const auto& [tid, info] : after) {
        const auto previous = before.find(tid);
        const std::uint64_t previous_runtime =
            previous == before.end() ? 0 : previous->second.runtime_ns;
        if (info.runtime_ns > previous_runtime) peak_active[tid] = info;
    }
    if (after.size() > peak_tasks.size()) peak_tasks = after;
    result.runtime_advancing_workers = peak_active.size();
    result.all_tasks = peak_tasks.size();
    for (auto& [tid, info] : peak_tasks) result.active.emplace_back(tid, std::move(info));
    return result;
}

std::vector<double> read_reference_prefix(const fs::path& path, const std::size_t elements) {
    if (fs::file_size(path) < elements * sizeof(double))
        throw std::runtime_error("reference prefix is too short: " + path.string());
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open reference: " + path.string());
    std::vector<double> reference(elements);
    input.read(reinterpret_cast<char*>(reference.data()),
               static_cast<std::streamsize>(elements * sizeof(double)));
    if (!input) throw std::runtime_error("cannot read reference prefix: " + path.string());
    return reference;
}

int run_batched_case(const Row& item, const int campaign, const std::string& method,
                     const fs::path& output_dir, const fs::path& reference_dir,
                     const bool diagnostics) {
    const int threads = std::stoi(item.at("threads"));
    const int batch = std::stoi(item.at("batch"));
    const bool inter = method == "tblis_2.0_armsve_inter";
    if (!inter && method != "tblis_2.0_armsve_intra")
        throw std::runtime_error("unknown batched TBLIS method: " + method);
    set_affinity(threads);
    tblis_set_num_threads(static_cast<unsigned>(inter ? 1 : threads));
    const BatchedOperand a = make_batched_operand(item, true);
    const BatchedOperand b = make_batched_operand(item, false);
    const auto c_shape = parse_shape(item.at("c_shape"));
    const auto c_strides = contiguous_strides(c_shape);
    const std::size_t per_output = product(c_shape);
    std::vector<float> c(batch * per_output, 0.0F);
    const auto a_labels = labels("abkl");
    const auto b_labels = labels("klcd");
    const auto c_labels = labels("abcd");
    std::unique_ptr<InterBatchExecutor> executor;
    if (inter) {
        executor = std::make_unique<InterBatchExecutor>(
            threads, a, b, c, c_shape, c_strides, a_labels, b_labels, c_labels);
    }
    const std::function<Record()> operation = [&]() {
        if (inter) return executor->run();
        return execute_batched_intra(
            a, b, c, c_shape, c_strides, a_labels, b_labels, c_labels);
    };
    for (int warmup = 0; warmup < kWarmup; ++warmup) operation();
    std::vector<Record> records;
    records.reserve(kRepeat);
    for (int repeat = 0; repeat < kRepeat; ++repeat) records.push_back(operation());

    const fs::path reference_path = reference_dir /
        (item.at("scale") + "_B32.f64");
    const auto reference = read_reference_prefix(reference_path, c.size());
    const auto [max_abs_error, max_rel_error] = validate(c, reference);
    if (max_rel_error > 1.0e-4)
        throw std::runtime_error("relative error exceeds 1e-4");

    std::vector<std::uint64_t> totals, transforms, packings, kernels, restores;
    for (const auto& record : records) {
        totals.push_back(record.total_ns);
        transforms.push_back(record.transform_ns);
        packings.push_back(record.packing_ns);
        kernels.push_back(record.kernel_ns);
        restores.push_back(record.restore_ns);
    }
    const auto median_total = integer_median(totals);
    const auto median_transform = integer_median(transforms);
    const auto median_packing = integer_median(packings);
    const auto median_kernel = integer_median(kernels);
    const auto median_restore = integer_median(restores);
    WorkerDiagnostic worker;
    if (diagnostics) worker = diagnose_operation(median_total, operation);

    const fs::path raw_path = output_dir / "raw" / method /
        (item.at("case_id") + "_c" + std::to_string(campaign) + ".tsv");
    fs::create_directories(raw_path.parent_path());
    if (fs::exists(raw_path)) throw std::runtime_error("refusing to overwrite raw file");
    const fs::path raw_temporary = raw_path.string() + ".tmp";
    std::ofstream raw(raw_temporary);
    raw << "case_id\tmethod\tcampaign\trepeat\tbatch\tthreads\taffinity\tmeasurement_scope"
           "\ttotal_ns\tprocess_cpu_ns\tprocess_core_equivalents\tcore_utilization_pct"
           "\ttransform_ns\tpacking_ns\tkernel_ns\trestore_ns\tflops\ttotal_gflops"
           "\tkernel_gflops\tminimum_bytes\tminimum_effective_bandwidth_gbps"
           "\tmax_abs_error\tmax_rel_error\n";
    raw << std::setprecision(12);
    const double flops = std::stod(item.at("flops"));
    const double minimum_bytes = std::stod(item.at("minimum_bytes"));
    for (int repeat = 0; repeat < kRepeat; ++repeat) {
        const auto& record = records[repeat];
        raw << item.at("case_id") << '\t' << method << '\t' << campaign << '\t'
            << repeat << '\t' << batch << '\t' << threads << '\t' << affinity_text()
            << (inter ? "\ttblis_inter_batch_e2e\t" : "\ttblis_intra_batch_phase_profile_e2e\t")
            << record.total_ns << '\t' << record.cpu_ns << '\t'
            << static_cast<double>(record.cpu_ns) / record.total_ns << '\t'
            << 100.0 * record.cpu_ns / (record.total_ns * threads) << '\t';
        if (inter) raw << "NA\tNA\tNA\tNA\t";
        else raw << record.transform_ns << '\t' << record.packing_ns << '\t'
                 << record.kernel_ns << '\t' << record.restore_ns << '\t';
        raw << static_cast<std::uint64_t>(flops) << '\t' << flops / record.total_ns << '\t';
        if (inter) raw << "NA\t";
        else raw << (record.kernel_ns ? flops / record.kernel_ns : 0.0) << '\t';
        raw << static_cast<std::uint64_t>(minimum_bytes) << '\t'
            << minimum_bytes / record.total_ns << '\t' << std::scientific
            << max_abs_error << '\t' << max_rel_error << std::defaultfloat << '\n';
    }
    raw.close();
    fs::rename(raw_temporary, raw_path);

    if (diagnostics) {
        const fs::path materialization_path = output_dir / "materialization" / method /
            (item.at("case_id") + "_c" + std::to_string(campaign) + ".tsv");
        fs::create_directories(materialization_path.parent_path());
        std::ofstream diagnostic(materialization_path);
        diagnostic << "case_id\tmethod\tcampaign\tbatch\tthreads\tmeasurement_scope"
                      "\tinput_materialize_self_ns\tinput_allocation_bytes"
                      "\ta_materialization_count\ta_materialization_bytes"
                      "\tb_materialization_count\tb_materialization_bytes"
                      "\twrapper_full_materialization_count\twrapper_full_materialization_bytes"
                      "\ttblis_transform_ns\ttblis_packing_ns\ttblis_kernel_ns\ttblis_restore_ns"
                      "\tpacking_calls\tpacking_bytes\n";
        diagnostic << item.at("case_id") << '\t' << method << '\t' << campaign << '\t'
                   << batch << '\t' << threads << "\ttblis_native_diagnostic\t0\t0\t0\t0"
                   << "\t0\t0\t0\t0\t";
        if (inter) diagnostic << "NA\tNA\tNA\tNA";
        else diagnostic << median_transform << '\t' << median_packing << '\t'
                        << median_kernel << '\t' << median_restore;
        diagnostic << "\tNA\tNA\n";

        const int outer_workers = inter ? executor->worker_count() : 1;
        const int observed_workers = inter
            ? outer_workers : static_cast<int>(worker.active.size());
        const fs::path thread_path = output_dir / "thread_diagnostics" / method /
            (item.at("case_id") + "_c" + std::to_string(campaign) + ".tsv");
        fs::create_directories(thread_path.parent_path());
        std::ofstream thread(thread_path);
        thread << "case_id\tmethod\tcampaign\tbatch\tthreads\tconfigured_threads"
                  "\tconfigured_inner_threads\tconfigured_outer_workers\tactual_active_workers"
                  "\truntime_advancing_workers\tactive_worker_tids\tactive_worker_affinities"
                  "\tdiagnostic_iterations\tdiagnostic_wall_ns\tdiagnostic_process_cpu_ns"
                  "\tprocess_core_equivalents\trequested_capacity_utilization_pct"
                  "\tall_task_count\tmeasurement_scope\n";
        std::ostringstream tids, affinities;
        if (inter) {
            const auto& observed_tids = executor->tids();
            for (std::size_t i = 0; i < observed_tids.size(); ++i) {
                if (i) { tids << ','; affinities << ';'; }
                tids << observed_tids[i];
                affinities << observed_tids[i] << ':' << affinity_text(observed_tids[i]);
            }
        } else {
            for (std::size_t i = 0; i < worker.active.size(); ++i) {
                if (i) { tids << ','; affinities << ';'; }
                tids << worker.active[i].first;
                affinities << worker.active[i].first << ':'
                           << worker.active[i].second.affinity;
            }
        }
        thread << item.at("case_id") << '\t' << method << '\t' << campaign << '\t'
               << batch << '\t' << threads << '\t' << observed_workers << '\t'
               << (inter ? 1 : threads) << '\t' << outer_workers << '\t'
               << observed_workers << '\t' << worker.runtime_advancing_workers << '\t'
               << tids.str() << '\t' << affinities.str() << '\t' << worker.iterations << '\t'
               << worker.wall_ns << '\t' << worker.cpu_ns << '\t'
               << static_cast<double>(worker.cpu_ns) / worker.wall_ns << '\t'
               << 100.0 * worker.cpu_ns / (worker.wall_ns * threads) << '\t'
               << worker.all_tasks << "\trepeated_worker_diagnostic_not_performance\n";
    }

    const std::vector<tblis::len_type> a_entry_shape = {
        static_cast<tblis::len_type>(batch), a.shape[0], a.shape[1], a.shape[2], a.shape[3]};
    const std::vector<tblis::len_type> b_entry_shape = {
        static_cast<tblis::len_type>(batch), b.shape[0], b.shape[1], b.shape[2], b.shape[3]};
    const std::vector<tblis::stride_type> a_entry_strides = {
        static_cast<tblis::stride_type>(a.batch_stride), a.strides[0], a.strides[1],
        a.strides[2], a.strides[3]};
    const std::vector<tblis::stride_type> b_entry_strides = {
        static_cast<tblis::stride_type>(b.batch_stride), b.strides[0], b.strides[1],
        b.strides[2], b.strides[3]};
    const fs::path metadata_path = output_dir / "metadata" / method /
        (item.at("case_id") + "_c" + std::to_string(campaign) + ".json");
    fs::create_directories(metadata_path.parent_path());
    const fs::path metadata_temporary = metadata_path.string() + ".tmp";
    std::ofstream metadata(metadata_temporary);
    metadata << "{\n  \"case_id\": \"" << item.at("case_id")
             << "\",\n  \"campaign\": " << campaign
             << ",\n  \"method\": \"" << method << "\""
             << ",\n  \"requested_threads\": " << threads
             << ",\n  \"configured_threads\": " << (inter ? executor->worker_count() : threads)
             << ",\n  \"configured_inner_threads\": " << (inter ? 1 : threads)
             << ",\n  \"configured_outer_workers\": " << (inter ? executor->worker_count() : 1)
             << ",\n  \"affinity\": \"" << affinity_text() << "\""
             << ",\n  \"warmup\": " << kWarmup << ",\n  \"repeat\": " << kRepeat
             << ",\n  \"equation\": \"pabkl,pklcd->pabcd\""
             << ",\n  \"wrapper_full_materialization_count\": 0"
             << ",\n  \"wrapper_full_materialization_bytes\": 0"
             << ",\n  \"a_entry\": {\"shape\": " << vector_json(a_entry_shape)
             << ", \"stride_elements\": " << vector_json(a_entry_strides)
             << ", \"storage_offset_elements\": 0, \"data_ptr\": "
             << reinterpret_cast<std::uintptr_t>(a.storage.data())
             << ", \"storage_data_ptr\": "
             << reinterpret_cast<std::uintptr_t>(a.storage.data())
             << ", \"storage_nbytes\": " << a.storage.size() * sizeof(float)
             << ", \"logical_nbytes\": " << a.batch * product(a.shape) * sizeof(float)
             << ", \"physical_order\": \"p" << a.physical_order << "\"}"
             << ",\n  \"b_entry\": {\"shape\": " << vector_json(b_entry_shape)
             << ", \"stride_elements\": " << vector_json(b_entry_strides)
             << ", \"storage_offset_elements\": 0, \"data_ptr\": "
             << reinterpret_cast<std::uintptr_t>(b.storage.data())
             << ", \"storage_data_ptr\": "
             << reinterpret_cast<std::uintptr_t>(b.storage.data())
             << ", \"storage_nbytes\": " << b.storage.size() * sizeof(float)
             << ", \"logical_nbytes\": " << b.batch * product(b.shape) * sizeof(float)
             << ", \"physical_order\": \"p" << b.physical_order << "\"}"
             << ",\n  \"tblis_a_data_ptr\": "
             << reinterpret_cast<std::uintptr_t>(a.storage.data())
             << ",\n  \"tblis_b_data_ptr\": "
             << reinterpret_cast<std::uintptr_t>(b.storage.data())
             << ",\n  \"a_logical_sha256\": \"" << a.logical_sha256 << "\""
             << ",\n  \"b_logical_sha256\": \"" << b.logical_sha256 << "\""
             << ",\n  \"reference_path\": \"" << reference_path.string() << "\""
             << ",\n  \"phase_semantics\": \""
             << (inter ? "N/A: concurrent calls cannot use the process-global profiling ABI"
                       : "sum of per-batch TBLIS profile durations") << "\""
             << ",\n  \"max_abs_error\": " << std::scientific << max_abs_error
             << ",\n  \"max_rel_error\": " << max_rel_error << "\n}\n";
    metadata.close();
    fs::rename(metadata_temporary, metadata_path);
    std::cout << "{\"case\":\"" << item.at("case_id") << "\",\"method\":\""
              << method << "\",\"campaign\":" << campaign
              << ",\"median_ns\":" << median_total
              << ",\"max_rel_error\":" << std::scientific << max_rel_error << "}\n";
    return 0;
}

int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> options;
        bool diagnostics = false;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--diagnostics") diagnostics = true;
            else {
                if (i + 1 >= argc) throw std::runtime_error("missing option value: " + option);
                options[option] = argv[++i];
            }
        }
        for (const auto* required : {
                 "--manifest", "--case", "--campaign", "--output-dir", "--reference-dir"})
            if (!options.count(required)) throw std::runtime_error(std::string("missing ") + required);
        const Row item = read_case(options.at("--manifest"), options.at("--case"));
        const int campaign = std::stoi(options.at("--campaign"));
        if (item.count("batch")) {
            if (!options.count("--method"))
                throw std::runtime_error("batched cases require --method");
            return run_batched_case(
                item, campaign, options.at("--method"), options.at("--output-dir"),
                options.at("--reference-dir"), diagnostics);
        }
        const int threads = std::stoi(item.at("threads"));
        set_affinity(threads);
        tblis_set_num_threads(static_cast<unsigned>(threads));
        if (tblis_get_num_threads() != static_cast<unsigned>(threads))
            throw std::runtime_error("TBLIS thread count mismatch");

        const Operand a = make_operand(item, true);
        const Operand b = make_operand(item, false);
        const auto c_shape = parse_shape(item.at("c_shape"));
        const auto c_strides = contiguous_strides(c_shape);
        std::vector<float> c(product(c_shape), 0.0F);
        const auto a_labels = labels("abkl");
        const auto b_labels = labels("klcd");
        const auto c_labels = labels("abcd");
        for (int warmup = 0; warmup < kWarmup; ++warmup)
            execute(a, b, c, c_shape, c_strides, a_labels, b_labels, c_labels);
        std::vector<Record> records;
        records.reserve(kRepeat);
        for (int repeat = 0; repeat < kRepeat; ++repeat)
            records.push_back(execute(
                a, b, c, c_shape, c_strides, a_labels, b_labels, c_labels));

        const fs::path reference_path = fs::path(options.at("--reference-dir")) /
            (item.at("scale") + ".f64");
        const auto reference = read_reference(reference_path, c.size());
        const auto [max_abs_error, max_rel_error] = validate(c, reference);
        if (max_rel_error > 1.0e-4) throw std::runtime_error("relative error exceeds 1e-4");

        std::vector<std::uint64_t> totals, transforms, packings, kernels, restores;
        for (const auto& record : records) {
            totals.push_back(record.total_ns);
            transforms.push_back(record.transform_ns);
            packings.push_back(record.packing_ns);
            kernels.push_back(record.kernel_ns);
            restores.push_back(record.restore_ns);
        }
        const auto median_total = integer_median(totals);
        const auto median_transform = integer_median(transforms);
        const auto median_packing = integer_median(packings);
        const auto median_kernel = integer_median(kernels);
        const auto median_restore = integer_median(restores);
        WorkerDiagnostic worker;
        if (diagnostics) {
            worker = diagnose_workers(
                median_total, a, b, c, c_shape, c_strides, a_labels, b_labels, c_labels);
        }

        const fs::path output_dir = options.at("--output-dir");
        const fs::path raw_path = output_dir / "raw" / "tblis_2.0_armsve" /
            (item.at("case_id") + "_c" + std::to_string(campaign) + ".tsv");
        fs::create_directories(raw_path.parent_path());
        if (fs::exists(raw_path)) throw std::runtime_error("refusing to overwrite raw file");
        const fs::path raw_temporary = raw_path.string() + ".tmp";
        std::ofstream raw(raw_temporary);
        raw << "case_id\tmethod\tcampaign\trepeat\tthreads\taffinity\tmeasurement_scope"
               "\ttotal_ns\tprocess_cpu_ns\tprocess_core_equivalents\tcore_utilization_pct"
               "\ttransform_ns\tpacking_ns\tkernel_ns\trestore_ns\tflops\ttotal_gflops"
               "\tkernel_gflops\tminimum_bytes\tminimum_effective_bandwidth_gbps"
               "\tmax_abs_error\tmax_rel_error\n";
        raw << std::setprecision(12);
        const double flops = std::stod(item.at("flops"));
        const double minimum_bytes = std::stod(item.at("minimum_bytes"));
        for (int repeat = 0; repeat < kRepeat; ++repeat) {
            const auto& record = records[repeat];
            raw << item.at("case_id") << "\ttblis_2.0_armsve\t" << campaign << '\t'
                << repeat << '\t' << threads << '\t' << affinity_text()
                << "\tnative_phase_profile_unprofiled_e2e\t" << record.total_ns << '\t'
                << record.cpu_ns << '\t' << static_cast<double>(record.cpu_ns) / record.total_ns
                << '\t' << 100.0 * record.cpu_ns / (record.total_ns * threads) << '\t'
                << record.transform_ns << '\t' << record.packing_ns << '\t'
                << record.kernel_ns << '\t' << record.restore_ns << '\t'
                << static_cast<std::uint64_t>(flops) << '\t' << flops / record.total_ns << '\t'
                << (record.kernel_ns ? flops / record.kernel_ns : 0.0) << '\t'
                << static_cast<std::uint64_t>(minimum_bytes) << '\t'
                << minimum_bytes / record.total_ns << '\t' << std::scientific
                << max_abs_error << '\t' << max_rel_error << std::defaultfloat << '\n';
        }
        raw.close();
        fs::rename(raw_temporary, raw_path);

        if (diagnostics) {
            const fs::path materialization_path = output_dir / "materialization" /
                "tblis_2.0_armsve" / (item.at("case_id") + "_c" +
                std::to_string(campaign) + ".tsv");
            fs::create_directories(materialization_path.parent_path());
            std::ofstream diagnostic(materialization_path);
            diagnostic << "case_id\tmethod\tcampaign\tthreads\tmeasurement_scope"
                          "\tinput_materialize_self_ns\tinput_allocation_bytes"
                          "\ta_materialization_count\ta_materialization_bytes"
                          "\tb_materialization_count\tb_materialization_bytes"
                          "\twrapper_full_materialization_count\twrapper_full_materialization_bytes"
                          "\ttblis_transform_ns\ttblis_packing_ns\ttblis_kernel_ns\ttblis_restore_ns"
                          "\tpacking_calls\tpacking_bytes\n";
            diagnostic << item.at("case_id") << "\ttblis_2.0_armsve\t" << campaign << '\t'
                       << threads << "\ttblis_native_diagnostic\t0\t0\t0\t0\t0\t0\t0\t0\t"
                       << median_transform << '\t' << median_packing << '\t' << median_kernel
                       << '\t' << median_restore << "\tNA\tNA\n";

            const fs::path thread_path = output_dir / "thread_diagnostics" /
                "tblis_2.0_armsve" / (item.at("case_id") + "_c" +
                std::to_string(campaign) + ".tsv");
            fs::create_directories(thread_path.parent_path());
            std::ofstream thread(thread_path);
            thread << "case_id\tmethod\tcampaign\tthreads\tconfigured_threads"
                      "\tactual_active_workers\truntime_advancing_workers"
                      "\tactive_worker_tids\tactive_worker_affinities"
                      "\tdiagnostic_iterations\tdiagnostic_wall_ns\tdiagnostic_process_cpu_ns"
                      "\tprocess_core_equivalents\trequested_capacity_utilization_pct"
                      "\tall_task_count\tmeasurement_scope\n";
            std::ostringstream tids, affinities;
            for (std::size_t i = 0; i < worker.active.size(); ++i) {
                if (i) { tids << ','; affinities << ';'; }
                tids << worker.active[i].first;
                affinities << worker.active[i].first << ':' << worker.active[i].second.affinity;
            }
            thread << item.at("case_id") << "\ttblis_2.0_armsve\t" << campaign << '\t'
                   << threads << '\t' << tblis_get_num_threads() << '\t' << worker.active.size()
                   << '\t' << worker.runtime_advancing_workers
                   << '\t' << tids.str() << '\t' << affinities.str() << '\t' << worker.iterations
                   << '\t' << worker.wall_ns << '\t' << worker.cpu_ns << '\t'
                   << static_cast<double>(worker.cpu_ns) / worker.wall_ns << '\t'
                   << 100.0 * worker.cpu_ns / (worker.wall_ns * threads) << '\t'
                   << worker.all_tasks << "\trepeated_worker_diagnostic_not_performance\n";
        }

        const fs::path metadata_path = output_dir / "metadata" / "tblis_2.0_armsve" /
            (item.at("case_id") + "_c" + std::to_string(campaign) + ".json");
        fs::create_directories(metadata_path.parent_path());
        const fs::path metadata_temporary = metadata_path.string() + ".tmp";
        std::ofstream metadata(metadata_temporary);
        metadata << "{\n  \"case_id\": \"" << item.at("case_id")
                 << "\",\n  \"campaign\": " << campaign
                 << ",\n  \"method\": \"tblis_2.0_armsve\""
                 << ",\n  \"requested_threads\": " << threads
                 << ",\n  \"configured_threads\": " << tblis_get_num_threads()
                 << ",\n  \"affinity\": \"" << affinity_text() << "\""
                 << ",\n  \"warmup\": " << kWarmup << ",\n  \"repeat\": " << kRepeat
                 << ",\n  \"equation\": \"abkl,klcd->abcd\""
                 << ",\n  \"wrapper_full_materialization_count\": 0"
                 << ",\n  \"wrapper_full_materialization_bytes\": 0"
                 << ",\n  \"a_entry\": {\"shape\": " << vector_json(a.shape)
                 << ", \"stride_elements\": " << vector_json(a.strides)
                 << ", \"storage_offset_elements\": 0, \"data_ptr\": "
                 << reinterpret_cast<std::uintptr_t>(a.storage.data())
                 << ", \"storage_data_ptr\": "
                 << reinterpret_cast<std::uintptr_t>(a.storage.data())
                 << ", \"storage_nbytes\": " << a.storage.size() * sizeof(float)
                 << ", \"logical_nbytes\": " << product(a.shape) * sizeof(float)
                 << ", \"physical_order\": \"" << a.physical_order << "\"}"
                 << ",\n  \"b_entry\": {\"shape\": " << vector_json(b.shape)
                 << ", \"stride_elements\": " << vector_json(b.strides)
                 << ", \"storage_offset_elements\": 0, \"data_ptr\": "
                 << reinterpret_cast<std::uintptr_t>(b.storage.data())
                 << ", \"storage_data_ptr\": "
                 << reinterpret_cast<std::uintptr_t>(b.storage.data())
                 << ", \"storage_nbytes\": " << b.storage.size() * sizeof(float)
                 << ", \"logical_nbytes\": " << product(b.shape) * sizeof(float)
                 << ", \"physical_order\": \"" << b.physical_order << "\"}"
                 << ",\n  \"tblis_a_data_ptr\": "
                 << reinterpret_cast<std::uintptr_t>(a.storage.data())
                 << ",\n  \"tblis_b_data_ptr\": "
                 << reinterpret_cast<std::uintptr_t>(b.storage.data())
                 << ",\n  \"a_logical_sha256\": \"" << a.logical_sha256 << "\""
                 << ",\n  \"b_logical_sha256\": \"" << b.logical_sha256 << "\""
                 << ",\n  \"reference_path\": \"" << reference_path.string() << "\""
                 << ",\n  \"phase_semantics\": \"packing/kernel are max per-thread accumulated durations\""
                 << ",\n  \"packing_count_semantics\": \"not exposed by current profiling ABI\""
                 << ",\n  \"max_abs_error\": " << std::scientific << max_abs_error
                 << ",\n  \"max_rel_error\": " << max_rel_error << "\n}\n";
        metadata.close();
        fs::rename(metadata_temporary, metadata_path);

        std::cout << "{\"case\":\"" << item.at("case_id")
                  << "\",\"method\":\"tblis_2.0_armsve\",\"campaign\":"
                  << campaign << ",\"median_ns\":" << median_total
                  << ",\"max_rel_error\":" << std::scientific << max_rel_error
                  << ",\"actual_workers\":"
                  << (diagnostics ? std::to_string(worker.active.size()) : "null") << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
