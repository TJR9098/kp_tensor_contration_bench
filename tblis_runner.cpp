#include <tblis.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sched.h>
#include <unistd.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Row = std::unordered_map<std::string, std::string>;

constexpr int kWarmup = 10;
constexpr int kRepeat = 50;

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
        if (values.size() != header.size())
            throw std::runtime_error("malformed manifest row");
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

std::size_t product(const std::vector<tblis::len_type>& shape) {
    std::size_t result = 1;
    for (const auto value : shape) result *= static_cast<std::size_t>(value);
    return result;
}

std::vector<tblis::stride_type> contiguous_strides(
    const std::vector<tblis::len_type>& shape) {
    std::vector<tblis::stride_type> strides(shape.size(), 1);
    for (std::size_t i = shape.size(); i-- > 1;)
        strides[i - 1] = strides[i] * shape[i];
    return strides;
}

float deterministic_value(const std::size_t index, const std::uint64_t salt) {
    const std::uint64_t bits = (index * 1103515245ULL + salt) & 0xFFFFULL;
    return static_cast<float>(bits) / 32768.0F - 1.0F;
}

struct Operand {
    std::string labels;
    std::string canonical_labels;
    std::vector<tblis::len_type> shape;
    std::vector<tblis::stride_type> strides;
    std::vector<float> storage;
};

Operand make_operand(const std::string& labels,
                     const std::string& canonical_labels,
                     const std::vector<tblis::len_type>& shape,
                     const bool padded,
                     const std::uint64_t salt) {
    if (labels.size() != shape.size() || canonical_labels.size() != shape.size())
        throw std::runtime_error("label/shape rank mismatch");
    std::map<char, tblis::len_type> size_by_label;
    for (std::size_t i = 0; i < labels.size(); ++i) size_by_label[labels[i]] = shape[i];
    std::vector<tblis::len_type> storage_shape = shape;
    if (padded) storage_shape.back() *= 2;
    auto strides = contiguous_strides(storage_shape);
    if (padded) strides.back() = 2;
    std::vector<float> storage(product(storage_shape), 0.0F);
    std::array<tblis::len_type, 256> coordinate_by_label{};
    std::vector<tblis::len_type> coordinates(shape.size());
    for (std::size_t logical = 0; logical < product(shape); ++logical) {
        std::size_t value = logical;
        for (std::size_t i = shape.size(); i-- > 0;) {
            coordinates[i] = static_cast<tblis::len_type>(value % shape[i]);
            value /= static_cast<std::size_t>(shape[i]);
            coordinate_by_label[static_cast<unsigned char>(labels[i])] = coordinates[i];
        }
        std::size_t canonical = 0;
        for (const char label : canonical_labels) {
            canonical *= static_cast<std::size_t>(size_by_label.at(label));
            canonical += static_cast<std::size_t>(
                coordinate_by_label[static_cast<unsigned char>(label)]);
        }
        std::size_t offset = 0;
        for (std::size_t i = 0; i < shape.size(); ++i)
            offset += static_cast<std::size_t>(coordinates[i] * strides[i]);
        storage[offset] = deterministic_value(canonical, salt);
    }
    return {labels, canonical_labels, shape, strides, std::move(storage)};
}

std::vector<float> materialize(const Operand& source) {
    std::vector<float> output(product(source.shape));
    std::vector<tblis::len_type> coordinates(source.shape.size());
    for (std::size_t logical = 0; logical < output.size(); ++logical) {
        std::size_t value = logical;
        std::size_t offset = 0;
        for (std::size_t i = source.shape.size(); i-- > 0;) {
            coordinates[i] = static_cast<tblis::len_type>(value % source.shape[i]);
            value /= static_cast<std::size_t>(source.shape[i]);
        }
        for (std::size_t i = 0; i < source.shape.size(); ++i)
            offset += static_cast<std::size_t>(coordinates[i] * source.strides[i]);
        output[logical] = source.storage[offset];
    }
    return output;
}

std::vector<tblis::label_type> labels(const std::string& text) {
    std::vector<tblis::label_type> output;
    for (const char value : text) output.push_back(static_cast<tblis::label_type>(value));
    return output;
}

std::uint64_t wall_ns(const Clock::time_point begin) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
}

std::uint64_t process_cpu_ns() {
    timespec value{};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0)
        throw std::runtime_error("clock_gettime failed");
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL + value.tv_nsec;
}

std::string affinity_text() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0)
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

struct Record {
    std::uint64_t total_ns = 0;
    std::uint64_t cpu_ns = 0;
    std::uint64_t transform_ns = 0;
    std::uint64_t packing_ns = 0;
    std::uint64_t kernel_ns = 0;
    std::uint64_t restore_ns = 0;
};

Record execute(Operand& a, Operand& b, std::vector<float>& c,
               const std::vector<tblis::len_type>& c_shape,
               const std::vector<tblis::stride_type>& c_strides,
               const std::vector<tblis::label_type>& a_labels,
               const std::vector<tblis::label_type>& b_labels,
               const std::vector<tblis::label_type>& c_labels,
               const bool materialize_each_call) {
    tblis_profile_reset();
    const auto cpu_begin = process_cpu_ns();
    const auto total_begin = Clock::now();
    std::vector<float> a_copy;
    std::vector<float> b_copy;
    std::vector<tblis::stride_type> a_strides = a.strides;
    std::vector<tblis::stride_type> b_strides = b.strides;
    float* a_data = a.storage.data();
    float* b_data = b.storage.data();
    std::uint64_t external_transform = 0;
    if (materialize_each_call) {
        const auto transform_begin = Clock::now();
        a_copy = materialize(a);
        b_copy = materialize(b);
        external_transform = wall_ns(transform_begin);
        a_data = a_copy.data();
        b_data = b_copy.data();
        a_strides = contiguous_strides(a.shape);
        b_strides = contiguous_strides(b.shape);
    }

    tblis::tblis_tensor a_tensor{}, b_tensor{}, c_tensor{};
    tblis::tblis_init_tensor_scaled_s(
        &a_tensor, 1.0F, static_cast<int>(a.shape.size()), a.shape.data(),
        a_data, a_strides.data());
    tblis::tblis_init_tensor_scaled_s(
        &b_tensor, 1.0F, static_cast<int>(b.shape.size()), b.shape.data(),
        b_data, b_strides.data());
    auto mutable_c_shape = c_shape;
    auto mutable_c_strides = c_strides;
    tblis::tblis_init_tensor_scaled_s(
        &c_tensor, 0.0F, static_cast<int>(mutable_c_shape.size()),
        mutable_c_shape.data(), c.data(), mutable_c_strides.data());
    tblis::tblis_tensor_mult(nullptr, nullptr, &a_tensor, a_labels.data(),
                             &b_tensor, b_labels.data(), &c_tensor, c_labels.data());
    Record record;
    record.total_ns = wall_ns(total_begin);
    record.cpu_ns = process_cpu_ns() - cpu_begin;
    tblis_profile profile{};
    tblis_profile_get(&profile);
    record.transform_ns = external_transform + profile.transform_ns;
    record.packing_ns = profile.packing_ns;
    record.kernel_ns = profile.kernel_ns;
    record.restore_ns = profile.restore_ns;
    return record;
}

std::pair<double, double> validate(const std::vector<float>& actual,
                                   const int batch, const int m, const int n,
                                   const int k) {
    if (actual.size() != static_cast<std::size_t>(batch) * m * n)
        throw std::runtime_error("output size mismatch");
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        for (int row = 0; row < m; ++row) {
            for (int column = 0; column < n; ++column) {
                double reference = 0.0;
                for (int reduction = 0; reduction < k; ++reduction) {
                    const std::size_t a_index =
                        (static_cast<std::size_t>(batch_index) * m + row) * k + reduction;
                    const std::size_t b_index =
                        (static_cast<std::size_t>(batch_index) * n + column) * k + reduction;
                    reference += static_cast<double>(deterministic_value(a_index, 12345)) *
                                 static_cast<double>(deterministic_value(b_index, 54321));
                }
                const std::size_t output_index =
                    (static_cast<std::size_t>(batch_index) * m + row) * n + column;
                const double value = actual[output_index];
                if (!std::isfinite(value)) throw std::runtime_error("non-finite output");
                const double error = std::abs(value - reference);
                max_abs = std::max(max_abs, error);
                max_rel = std::max(max_rel, error / std::max(std::abs(reference), 1.0));
            }
        }
    }
    return {max_abs, max_rel};
}

void set_affinity(const int threads) {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (threads == 1) CPU_SET(3, &set);
    else if (threads == 2) { CPU_SET(2, &set); CPU_SET(3, &set); }
    else if (threads == 4) for (int cpu = 0; cpu < 4; ++cpu) CPU_SET(cpu, &set);
    else throw std::runtime_error("unsupported thread count");
    if (sched_setaffinity(0, sizeof(set), &set) != 0)
        throw std::runtime_error("sched_setaffinity failed");
}

int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> options;
        for (int i = 1; i + 1 < argc; i += 2) options[argv[i]] = argv[i + 1];
        for (const auto* required : {"--manifest", "--case", "--campaign", "--output-dir"})
            if (!options.count(required)) throw std::runtime_error(std::string("missing ") + required);
        const Row item = read_case(options.at("--manifest"), options.at("--case"));
        const int campaign = std::stoi(options.at("--campaign"));
        const int threads = std::stoi(item.at("threads"));
        set_affinity(threads);
        tblis_set_num_threads(static_cast<unsigned>(threads));
        if (tblis_get_num_threads() != static_cast<unsigned>(threads))
            throw std::runtime_error("TBLIS thread count mismatch");

        const bool padded = item.at("layout") == "L4" || item.at("layout") == "L5";
        Operand a = make_operand(item.at("a_labels"), item.at("canonical_a_labels"),
                                 parse_shape(item.at("a_shape")), padded, 12345);
        Operand b = make_operand(item.at("b_labels"), item.at("canonical_b_labels"),
                                 parse_shape(item.at("b_shape")), padded, 54321);
        const auto c_shape = parse_shape(item.at("c_shape"));
        const auto c_strides = contiguous_strides(c_shape);
        std::vector<float> c(product(c_shape), 0.0F);
        const auto a_label_values = labels(item.at("a_labels"));
        const auto b_label_values = labels(item.at("b_labels"));
        const auto c_label_values = labels(item.at("c_labels"));
        const bool materialize_each_call = item.at("layout") == "L5";
        for (int warmup = 0; warmup < kWarmup; ++warmup)
            execute(a, b, c, c_shape, c_strides, a_label_values, b_label_values,
                    c_label_values, materialize_each_call);
        std::vector<Record> records;
        records.reserve(kRepeat);
        for (int repeat = 0; repeat < kRepeat; ++repeat)
            records.push_back(execute(
                a, b, c, c_shape, c_strides, a_label_values, b_label_values,
                c_label_values, materialize_each_call));
        const auto [max_abs_error, max_rel_error] = validate(
            c, std::stoi(item.at("B")), std::stoi(item.at("M")),
            std::stoi(item.at("N")), std::stoi(item.at("K")));
        if (max_rel_error > 1.0e-4)
            throw std::runtime_error("relative error exceeds 1e-4");

        const fs::path output_dir = options.at("--output-dir");
        const fs::path raw_path = output_dir / "raw" / "tblis_2.0_armsve" /
            (item.at("case_id") + "_c" + std::to_string(campaign) + ".tsv");
        fs::create_directories(raw_path.parent_path());
        if (fs::exists(raw_path)) throw std::runtime_error("refusing to overwrite raw file");
        const fs::path raw_temporary = raw_path.string() + ".tmp";
        std::ofstream raw(raw_temporary);
        raw << "case_id\tmethod\tcampaign\trepeat\tthreads\taffinity\tmeasurement_scope"
               "\ttotal_ns\tprocess_cpu_ns\tcore_utilization_pct\ttransform_ns"
               "\tpacking_ns\tkernel_ns\trestore_ns\tflops\ttotal_gflops"
               "\tkernel_gflops\tminimum_bytes\tminimum_effective_bandwidth_gbps"
               "\tmax_abs_error\tmax_rel_error\n";
        raw << std::setprecision(12);
        const double flops = std::stod(item.at("flops"));
        const double minimum_bytes = std::stod(item.at("minimum_bytes"));
        for (int repeat = 0; repeat < kRepeat; ++repeat) {
            const auto& record = records[repeat];
            raw << item.at("case_id") << "\ttblis_2.0_armsve\t" << campaign << '\t'
                << repeat << '\t' << threads << '\t' << affinity_text()
                << "\tnative_phase_profile\t" << record.total_ns << '\t'
                << record.cpu_ns << '\t'
                << std::min(100.0, 100.0 * record.cpu_ns /
                                      (record.total_ns * threads)) << '\t'
                << record.transform_ns << '\t' << record.packing_ns << '\t'
                << record.kernel_ns << '\t' << record.restore_ns << '\t'
                << static_cast<std::uint64_t>(flops) << '\t'
                << flops / record.total_ns << '\t'
                << (record.kernel_ns ? flops / record.kernel_ns : 0.0) << '\t'
                << static_cast<std::uint64_t>(minimum_bytes) << '\t'
                << minimum_bytes / record.total_ns << '\t' << std::scientific
                << max_abs_error << '\t' << max_rel_error << std::defaultfloat << '\n';
        }
        raw.close();
        fs::rename(raw_temporary, raw_path);

        const fs::path metadata_path = output_dir / "metadata" / "tblis_2.0_armsve" /
            (item.at("case_id") + "_c" + std::to_string(campaign) + ".json");
        fs::create_directories(metadata_path.parent_path());
        const fs::path metadata_temporary = metadata_path.string() + ".tmp";
        std::ofstream metadata(metadata_temporary);
        metadata << "{\n  \"case_id\": \"" << item.at("case_id")
                 << "\",\n  \"campaign\": " << campaign
                 << ",\n  \"method\": \"tblis_2.0_armsve\""
                 << ",\n  \"threads\": " << threads
                 << ",\n  \"affinity\": \"" << affinity_text()
                 << "\",\n  \"warmup\": " << kWarmup
                 << ",\n  \"repeat\": " << kRepeat
                 << ",\n  \"phase_semantics\": \"multithread packing/kernel are max per-thread accumulated durations\""
                 << ",\n  \"max_abs_error\": " << std::scientific << max_abs_error
                 << ",\n  \"max_rel_error\": " << max_rel_error << "\n}\n";
        metadata.close();
        fs::rename(metadata_temporary, metadata_path);

        std::vector<std::uint64_t> totals;
        for (const auto& record : records) totals.push_back(record.total_ns);
        std::sort(totals.begin(), totals.end());
        std::cout << "{\"case\":\"" << item.at("case_id")
                  << "\",\"method\":\"tblis_2.0_armsve\",\"campaign\":"
                  << campaign << ",\"median_ns\":" << (totals[24] + totals[25]) / 2
                  << ",\"max_rel_error\":" << std::scientific << max_rel_error << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
