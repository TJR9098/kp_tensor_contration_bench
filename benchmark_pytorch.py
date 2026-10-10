#!/usr/bin/env python3
"""Benchmark one manifest case with an installed PyTorch build."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import struct
import time
from pathlib import Path


WARMUP = 10
REPEAT = 50
MATERIALIZE_KEYS = {
    "aten::clone", "aten::contiguous", "aten::copy_", "aten::_to_copy",
    "aten::empty", "aten::empty_like", "aten::empty_strided",
}
METADATA_KEYS = {
    "aten::view", "aten::reshape", "aten::_reshape_alias", "aten::permute",
    "aten::as_strided", "aten::unsqueeze", "aten::squeeze", "aten::transpose",
}


def read_case(path: Path, case_id: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        matches = [row for row in csv.DictReader(stream, delimiter="\t")
                   if row["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {case_id}, got {len(matches)}")
    return matches[0]


def parse_shape(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def deterministic_tensor(torch, shape: list[int], salt: int):
    count = 1
    for dimension in shape:
        count *= dimension
    index = torch.arange(count, dtype=torch.int64)
    values = ((index * 1103515245 + salt) & 0xFFFF).to(torch.float32)
    return values.div_(32768.0).sub_(1.0).reshape(shape)


def permute_to(tensor, canonical_labels: str, physical_labels: str):
    order = [canonical_labels.index(label) for label in physical_labels]
    return tensor.permute(order).contiguous()


def padded_view(torch, tensor):
    shape = list(tensor.shape)
    storage_shape = shape[:-1] + [shape[-1] * 2]
    storage = torch.empty(storage_shape, dtype=tensor.dtype)
    view = storage[..., ::2]
    view.copy_(tensor)
    return view, storage


def tensor_fingerprint(tensor) -> str:
    values = tensor.detach().contiguous().view(-1).tolist()
    digest = hashlib.sha256()
    chunk_size = 8192
    for begin in range(0, len(values), chunk_size):
        chunk = values[begin:begin + chunk_size]
        digest.update(struct.pack(f"<{len(chunk)}f", *chunk))
    return digest.hexdigest()


def is_descendant(event, ancestor) -> bool:
    parent = event.cpu_parent
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.cpu_parent
    return False


def diagnostic_row(prof, case: dict[str, str], method: str, campaign: int) -> dict[str, object]:
    events = list(prof.events())
    operation_roots = [event for event in events if event.key == "benchmark::operation"]
    if not operation_roots:
        raise RuntimeError("profiler did not record benchmark::operation")
    operation_root = max(operation_roots, key=lambda event: event.cpu_time_total)
    operation_descendants = [
        event for event in events if is_descendant(event, operation_root)
    ]
    roots = [event for event in operation_descendants if event.key == "aten::einsum"]
    if not roots:
        raise RuntimeError("profiler did not record aten::einsum")
    root = max(roots, key=lambda event: event.cpu_time_total)
    bmms = [event for event in operation_descendants if event.key == "aten::bmm"]
    if not bmms:
        raise RuntimeError("profiler did not record aten::bmm")
    first_bmm = min(event.time_range.start for event in bmms)
    last_bmm = max(event.time_range.end for event in bmms)

    def self_ns(selected) -> int:
        return round(sum(event.self_cpu_time_total for event in selected) * 1000)

    materialize = [event for event in operation_descendants if event.key in MATERIALIZE_KEYS]
    metadata = [event for event in operation_descendants if event.key in METADATA_KEYS]
    outside_bmm = [
        event for event in materialize
        if not any(is_descendant(event, bmm) for bmm in bmms)
    ]
    input_materialize = [event for event in outside_bmm if event.time_range.end <= first_bmm]
    output_materialize = [event for event in outside_bmm if event.time_range.start >= last_bmm]
    inside_bmm = [
        event for event in materialize
        if any(is_descendant(event, bmm) for bmm in bmms)
    ]
    outside_metadata = [
        event for event in metadata
        if not any(is_descendant(event, bmm) for bmm in bmms)
    ]
    einsum_total = round(root.cpu_time_total * 1000)
    operation_total = round(operation_root.cpu_time_total * 1000)
    planner_self = round(root.self_cpu_time_total * 1000)
    bmm_total = round(sum(event.cpu_time_total for event in bmms) * 1000)
    classified = (
        planner_self + bmm_total + self_ns(input_materialize)
        + self_ns(output_materialize) + self_ns(outside_metadata)
    )
    return {
        "case_id": case["case_id"], "method": method, "campaign": campaign,
        "threads": int(case["threads"]),
        "operation_total_trace_ns": operation_total,
        "einsum_total_trace_ns": einsum_total,
        "planner_self_ns": planner_self,
        "input_materialize_self_ns": self_ns(input_materialize),
        "bmm_container_ns": bmm_total,
        "inside_bmm_materialize_self_ns": self_ns(inside_bmm),
        "output_materialize_self_ns": self_ns(output_materialize),
        "metadata_view_self_ns": self_ns(outside_metadata),
        "unclassified_trace_ns": max(0, operation_total - classified),
        "profiler_event_count": len(events),
        "measurement_scope": "profiler_diagnostic_not_performance",
    }


def loaded_libraries() -> list[str]:
    paths: set[str] = set()
    with Path("/proc/self/maps").open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip().split()
            if fields and fields[-1].startswith("/") and ".so" in fields[-1]:
                paths.add(fields[-1])
    return sorted(paths)


def write_one_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--campaign", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--export-trace", action="store_true")
    args = parser.parse_args()

    case = read_case(args.manifest, args.case)
    threads = int(case["threads"])
    cpus = {1: {3}, 2: {2, 3}, 4: {0, 1, 2, 3}}[threads]
    os.sched_setaffinity(0, cpus)

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    a_canonical_shape = parse_shape(case["a_shape"])
    b_canonical_shape = parse_shape(case["b_shape"])
    # Manifest shapes are in physical label order; recover canonical order.
    a_size = dict(zip(case["a_labels"], a_canonical_shape))
    b_size = dict(zip(case["b_labels"], b_canonical_shape))
    a_shape = [a_size[label] for label in case["canonical_a_labels"]]
    b_shape = [b_size[label] for label in case["canonical_b_labels"]]
    a_base = deterministic_tensor(torch, a_shape, 12345)
    b_base = deterministic_tensor(torch, b_shape, 54321)
    a = permute_to(a_base, case["canonical_a_labels"], case["a_labels"])
    b = permute_to(b_base, case["canonical_b_labels"], case["b_labels"])
    backing_storage = []
    if case["layout"] in {"L4", "L5"}:
        a, a_storage = padded_view(torch, a)
        b, b_storage = padded_view(torch, b)
        backing_storage = [a_storage, b_storage]

    equation = case["equation"]
    if case["layout"] == "L5":
        operation = lambda: torch.einsum(equation, a.contiguous(), b.contiguous())
    else:
        operation = lambda: torch.einsum(equation, a, b)

    with torch.inference_mode():
        reference = torch.einsum(equation, a.double(), b.double())
        for _ in range(WARMUP):
            operation()
        result = None
        timings: list[tuple[int, int, int]] = []
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for repeat in range(REPEAT):
                result = None
                cpu_begin = time.process_time_ns()
                wall_begin = time.perf_counter_ns()
                result = operation()
                wall_ns = time.perf_counter_ns() - wall_begin
                cpu_ns = time.process_time_ns() - cpu_begin
                timings.append((repeat, wall_ns, cpu_ns))
        finally:
            if gc_was_enabled:
                gc.enable()
        assert result is not None
        difference = (result.double() - reference).abs()
        max_abs_error = float(difference.max().item())
        max_rel_error = float((difference / reference.abs().clamp_min(1.0)).max().item())
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("benchmark::operation"):
                operation()

    diagnostic = diagnostic_row(prof, case, args.method, args.campaign)
    trace_path = None
    if args.export_trace:
        trace_path = args.output_dir / "traces" / args.method / f"{case['case_id']}_c{args.campaign}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))

    config = torch.__config__.show()
    config_hash = hashlib.sha256(config.encode()).hexdigest()
    raw_path = args.output_dir / "raw" / args.method / f"{case['case_id']}_c{args.campaign}.tsv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        raise FileExistsError(f"refusing to overwrite {raw_path}")
    fields = [
        "case_id", "method", "campaign", "repeat", "threads", "affinity",
        "measurement_scope", "total_ns", "process_cpu_ns", "core_utilization_pct",
        "transform_ns", "packing_ns", "kernel_ns", "restore_ns", "flops",
        "total_gflops", "kernel_gflops", "minimum_bytes",
        "minimum_effective_bandwidth_gbps", "max_abs_error", "max_rel_error",
    ]
    raw_temporary = raw_path.with_suffix(raw_path.suffix + ".tmp")
    with raw_temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for repeat, wall_ns, cpu_ns in timings:
            writer.writerow({
                "case_id": case["case_id"], "method": args.method,
                "campaign": args.campaign, "repeat": repeat, "threads": threads,
                "affinity": ",".join(map(str, sorted(os.sched_getaffinity(0)))),
                "measurement_scope": "framework_e2e", "total_ns": wall_ns,
                "process_cpu_ns": cpu_ns,
                "core_utilization_pct": min(100.0, 100.0 * cpu_ns / (wall_ns * threads)),
                "transform_ns": "NA", "packing_ns": "NA", "kernel_ns": "NA",
                "restore_ns": "NA", "flops": case["flops"],
                "total_gflops": int(case["flops"]) / wall_ns,
                "kernel_gflops": "NA", "minimum_bytes": case["minimum_bytes"],
                "minimum_effective_bandwidth_gbps": int(case["minimum_bytes"]) / wall_ns,
                "max_abs_error": f"{max_abs_error:.12e}",
                "max_rel_error": f"{max_rel_error:.12e}",
            })
    raw_temporary.replace(raw_path)
    diagnostic_path = args.output_dir / "diagnostics" / args.method / f"{case['case_id']}_c{args.campaign}.tsv"
    write_one_row(diagnostic_path, diagnostic)
    metadata = {
        "case": case, "method": args.method, "campaign": args.campaign,
        "torch_version": torch.__version__, "torch_config": config,
        "torch_config_sha256": config_hash, "platform": platform.platform(),
        "warmup": WARMUP, "repeat": REPEAT, "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "a_shape": list(a.shape), "a_stride": list(a.stride()),
        "b_shape": list(b.shape), "b_stride": list(b.stride()),
        "a_logical_sha256": tensor_fingerprint(a_base),
        "b_logical_sha256": tensor_fingerprint(b_base),
        "max_abs_error": max_abs_error, "max_rel_error": max_rel_error,
        "trace": str(trace_path) if trace_path else None,
        "loaded_shared_objects": loaded_libraries(),
        "backing_storage_count": len(backing_storage),
    }
    metadata_path = args.output_dir / "metadata" / args.method / f"{case['case_id']}_c{args.campaign}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata_temporary.replace(metadata_path)
    ordered = sorted(wall for _, wall, _ in timings)
    print(json.dumps({
        "case": case["case_id"], "method": args.method, "campaign": args.campaign,
        "median_ns": (ordered[24] + ordered[25]) // 2,
        "max_rel_error": max_rel_error,
        "einsum_trace_ns": diagnostic["einsum_total_trace_ns"],
    }))


if __name__ == "__main__":
    main()
