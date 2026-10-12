#!/usr/bin/env python3
"""Benchmark independent batched X0/X4 contractions with an installed PyTorch."""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

from benchmark_tblis_pytorch import (
    flatten_tensors,
    loaded_libraries,
    profiler_diagnostic,
    tensor_metadata,
    worker_diagnostic,
    write_one_row,
)


WARMUP = 10
REPEAT = 50
REFERENCE_BATCH = 32


def read_case(path: Path, case_id: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        matches = [
            row for row in csv.DictReader(stream, delimiter="\t")
            if row["case_id"] == case_id
        ]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {case_id}, got {len(matches)}")
    return matches[0]


def parse_shape(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def deterministic_tensor(torch, shape: list[int], salt: int, offset: int):
    count = math.prod(shape)
    index = torch.arange(offset, offset + count, dtype=torch.int64)
    values = ((index * 1103515245 + salt) & 0xFFFF).to(torch.float32)
    return values.div_(32768.0).sub_(1.0).reshape(shape)


def build_operand(torch, case: dict[str, str], operand: str):
    batch = int(case["batch"])
    if operand == "A":
        canonical_labels = "abkl"
        physical_labels = case["a_physical_order"]
        canonical_shape = parse_shape(case["a_shape"])
        salt = 12345
    else:
        canonical_labels = "klcd"
        physical_labels = case["b_physical_order"]
        canonical_shape = parse_shape(case["b_shape"])
        salt = 54321
    sizes = dict(zip(canonical_labels, canonical_shape))
    physical_shape = [sizes[label] for label in physical_labels]
    backing = torch.empty([batch, *physical_shape], dtype=torch.float32)
    per_batch_elements = math.prod(canonical_shape)
    to_physical = [canonical_labels.index(label) for label in physical_labels]
    for batch_index in range(batch):
        canonical = deterministic_tensor(
            torch, canonical_shape, salt, batch_index * per_batch_elements
        )
        backing[batch_index].copy_(canonical.permute(to_physical))
    physical_view = backing
    to_logical = [0, *[1 + physical_labels.index(label) for label in canonical_labels]]
    logical = physical_view.permute(to_logical)
    return logical, backing, physical_view


def tensor_fingerprint_batched(tensor) -> str:
    digest = hashlib.sha256()
    for batch_index in range(tensor.shape[0]):
        contiguous = tensor[batch_index].detach().contiguous()
        digest.update(ctypes.string_at(
            contiguous.data_ptr(), contiguous.numel() * contiguous.element_size()
        ))
    return digest.hexdigest()


def validate_result(result, reference) -> tuple[float, float]:
    max_abs = 0.0
    max_rel = 0.0
    for batch_index in range(result.shape[0]):
        difference = (result[batch_index].double() - reference[batch_index]).abs()
        max_abs = max(max_abs, float(difference.max().item()))
        relative = difference / reference[batch_index].abs().clamp_min(1.0)
        max_rel = max(max_rel, float(relative.max().item()))
    return max_abs, max_rel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--campaign", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--export-trace", action="store_true")
    args = parser.parse_args()
    case = read_case(args.manifest, args.case)
    threads = int(case["threads"])
    batch = int(case["batch"])
    affinity = {
        1: {3},
        2: {2, 3},
        4: set(range(4)),
        8: set(range(8)),
        32: set(range(32)),
    }[threads]
    os.sched_setaffinity(0, affinity)

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    from torch.utils._mode_utils import no_dispatch
    from torch.utils._python_dispatch import TorchDispatchMode

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    a, a_backing, a_physical = build_operand(torch, case, "A")
    b, b_backing, b_physical = build_operand(torch, case, "B")
    operation = lambda: torch.einsum(case["equation"], a, b)

    per_output_elements = int(case["output_elements"]) // batch
    reference_path = args.reference_dir / f"{case['scale']}_B{REFERENCE_BATCH}.f64"
    reference = torch.from_file(
        str(reference_path), shared=False, size=REFERENCE_BATCH * per_output_elements,
        dtype=torch.float64,
    ).reshape([REFERENCE_BATCH, *parse_shape(case["c_shape"])])[:batch]

    with torch.inference_mode():
        for _ in range(WARMUP):
            operation()
        timings: list[tuple[int, int, int]] = []
        result = None
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for repeat in range(REPEAT):
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
        max_abs_error, max_rel_error = validate_result(result, reference)
        del result
    if max_rel_error > 1.0e-4:
        raise RuntimeError(f"relative error exceeds 1e-4: {max_rel_error}")

    ordered = sorted(wall for _, wall, _ in timings)
    median_ns = (ordered[24] + ordered[25]) // 2
    thread_diagnostic = None
    diagnostic = None
    audit_payload = None
    trace_path = None
    if args.diagnostics:
        thread_diagnostic = worker_diagnostic(operation, median_ns, threads)

        class DispatchAudit(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.roles = {
                    a.untyped_storage().data_ptr(): "A",
                    b.untyped_storage().data_ptr(): "B",
                }
                self.events: list[dict[str, object]] = []
                self.materializations: list[dict[str, object]] = []

            def __torch_dispatch__(self, func, types, call_args=(), kwargs=None):
                kwargs = kwargs or {}
                with no_dispatch():
                    inputs = flatten_tensors(torch, (call_args, kwargs))
                    input_meta = [tensor_metadata(tensor) for tensor in inputs]
                    input_roles = {
                        self.roles.get(int(item["storage_data_ptr"]), "unknown")
                        for item in input_meta
                    }
                output = func(*call_args, **kwargs)
                with no_dispatch():
                    outputs = flatten_tensors(torch, output)
                    output_meta = [tensor_metadata(tensor) for tensor in outputs]
                    op_name = str(func)
                    role = next(iter(input_roles)) if len(input_roles) == 1 else "unknown"
                    if op_name.startswith("aten.bmm"):
                        role = "kernel_output"
                    for item in output_meta:
                        self.roles[int(item["storage_data_ptr"])] = role
                    self.events.append({
                        "index": len(self.events), "operation": op_name,
                        "input_roles": sorted(input_roles),
                        "inputs": input_meta, "outputs": output_meta,
                    })
                    if op_name.startswith("aten.clone"):
                        input_storages = {
                            int(item["storage_data_ptr"]) for item in input_meta
                        }
                        for item in output_meta:
                            if int(item["storage_data_ptr"]) not in input_storages:
                                self.materializations.append({
                                    "operation": op_name, "role": role,
                                    "bytes": int(item["logical_nbytes"]),
                                    "storage_data_ptr": int(item["storage_data_ptr"]),
                                    "shape": item["shape"],
                                    "stride_elements": item["stride_elements"],
                                })
                return output

        with torch.inference_mode(), profile(
            activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True
        ) as prof:
            with record_function("benchmark::operation"):
                operation()
        audit = DispatchAudit()
        with torch.no_grad(), audit:
            operation()
        diagnostic = profiler_diagnostic(prof, audit, case, args.method, args.campaign)
        audited_bytes = sum(int(record["bytes"]) for record in audit.materializations)
        if audited_bytes != int(diagnostic["input_allocation_bytes"]):
            raise RuntimeError(
                "dispatch materialization bytes do not match profiler input allocations: "
                f"{audited_bytes} != {diagnostic['input_allocation_bytes']}"
            )
        audit_payload = {
            "materializations": audit.materializations,
            "dispatch_events": audit.events,
        }
        if args.export_trace:
            trace_path = (
                args.output_dir / "traces" / args.method
                / f"{case['case_id']}_c{args.campaign}.json"
            )
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(trace_path))

    raw_path = args.output_dir / "raw" / args.method / f"{case['case_id']}_c{args.campaign}.tsv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        raise FileExistsError(f"refusing to overwrite {raw_path}")
    fields = [
        "case_id", "method", "campaign", "repeat", "batch", "threads", "affinity",
        "measurement_scope", "total_ns", "process_cpu_ns", "process_core_equivalents",
        "core_utilization_pct", "transform_ns", "packing_ns", "kernel_ns",
        "restore_ns", "flops", "total_gflops", "kernel_gflops", "minimum_bytes",
        "minimum_effective_bandwidth_gbps", "max_abs_error", "max_rel_error",
    ]
    temporary = raw_path.with_suffix(raw_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for repeat, wall_ns, cpu_ns in timings:
            writer.writerow({
                "case_id": case["case_id"], "method": args.method,
                "campaign": args.campaign, "repeat": repeat, "batch": batch,
                "threads": threads,
                "affinity": ",".join(map(str, sorted(os.sched_getaffinity(0)))),
                "measurement_scope": "framework_batched_e2e_unprofiled",
                "total_ns": wall_ns, "process_cpu_ns": cpu_ns,
                "process_core_equivalents": cpu_ns / wall_ns,
                "core_utilization_pct": 100.0 * cpu_ns / (wall_ns * threads),
                "transform_ns": "NA", "packing_ns": "NA", "kernel_ns": "NA",
                "restore_ns": "NA", "flops": case["flops"],
                "total_gflops": int(case["flops"]) / wall_ns,
                "kernel_gflops": "NA", "minimum_bytes": case["minimum_bytes"],
                "minimum_effective_bandwidth_gbps": int(case["minimum_bytes"]) / wall_ns,
                "max_abs_error": f"{max_abs_error:.12e}",
                "max_rel_error": f"{max_rel_error:.12e}",
            })
    temporary.replace(raw_path)

    if diagnostic is not None:
        write_one_row(
            args.output_dir / "materialization" / args.method
            / f"{case['case_id']}_c{args.campaign}.tsv",
            diagnostic,
        )
    if thread_diagnostic is not None:
        thread_row = {
            "case_id": case["case_id"], "method": args.method,
            "campaign": args.campaign, "batch": batch, "threads": threads,
            "configured_threads": torch.get_num_threads(),
            "configured_inner_threads": torch.get_num_threads(),
            "configured_outer_workers": 1,
            "interop_threads": torch.get_num_interop_threads(),
            "actual_active_workers": thread_diagnostic["actual_active_workers"],
            "active_worker_tids": ",".join(
                str(item["tid"]) for item in thread_diagnostic["active_workers"]
            ),
            "active_worker_affinities": ";".join(
                f"{item['tid']}:{item['affinity']}"
                for item in thread_diagnostic["active_workers"]
            ),
            "diagnostic_iterations": thread_diagnostic["diagnostic_iterations"],
            "diagnostic_wall_ns": thread_diagnostic["diagnostic_wall_ns"],
            "diagnostic_process_cpu_ns": thread_diagnostic["diagnostic_process_cpu_ns"],
            "process_core_equivalents": thread_diagnostic["process_core_equivalents"],
            "requested_capacity_utilization_pct": thread_diagnostic["requested_capacity_utilization_pct"],
            "all_task_count": thread_diagnostic["all_task_count"],
            "measurement_scope": "repeated_worker_diagnostic_not_performance",
        }
        write_one_row(
            args.output_dir / "thread_diagnostics" / args.method
            / f"{case['case_id']}_c{args.campaign}.tsv",
            thread_row,
        )

    config = torch.__config__.show()
    logical_a_hash = tensor_fingerprint_batched(a)
    logical_b_hash = tensor_fingerprint_batched(b)
    metadata = {
        "case": case, "method": args.method, "campaign": args.campaign,
        "torch_version": torch.__version__, "torch_config": config,
        "torch_config_sha256": hashlib.sha256(config.encode()).hexdigest(),
        "platform": platform.platform(), "warmup": WARMUP, "repeat": REPEAT,
        "requested_threads": threads, "configured_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "a_entry": tensor_metadata(a), "b_entry": tensor_metadata(b),
        "a_backing": tensor_metadata(a_backing), "b_backing": tensor_metadata(b_backing),
        "a_physical_view": tensor_metadata(a_physical),
        "b_physical_view": tensor_metadata(b_physical),
        "a_logical_sha256": logical_a_hash,
        "b_logical_sha256": logical_b_hash,
        "a_canonical_sha256": logical_a_hash,
        "b_canonical_sha256": logical_b_hash,
        "max_abs_error": max_abs_error, "max_rel_error": max_rel_error,
        "reference_path": str(reference_path),
        "reference_prefix_batches": batch,
        "trace": str(trace_path) if trace_path else None,
        "loaded_shared_objects": loaded_libraries(),
        "thread_diagnostic": thread_diagnostic,
        "dispatch_audit": audit_payload,
    }
    metadata_path = (
        args.output_dir / "metadata" / args.method
        / f"{case['case_id']}_c{args.campaign}.json"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata_temporary.replace(metadata_path)
    print(json.dumps({
        "case": case["case_id"], "method": args.method,
        "campaign": args.campaign, "median_ns": median_ns,
        "max_rel_error": max_rel_error,
        "actual_workers": (
            thread_diagnostic["actual_active_workers"] if thread_diagnostic else None
        ),
    }))


if __name__ == "__main__":
    main()
