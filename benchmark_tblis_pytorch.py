#!/usr/bin/env python3
"""Benchmark the fixed-equation X-layout cases with an installed PyTorch."""

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


WARMUP = 10
REPEAT = 50
DIAGNOSTIC_TARGET_NS = 250_000_000
MATERIALIZE_KEYS = {
    "aten::clone", "aten::contiguous", "aten::copy_", "aten::_to_copy",
    "aten::empty", "aten::empty_like", "aten::empty_strided",
}
METADATA_KEYS = {
    "aten::view", "aten::reshape", "aten::_reshape_alias", "aten::permute",
    "aten::as_strided", "aten::unsqueeze", "aten::squeeze", "aten::transpose",
    "aten::_unsafe_view",
}


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


def deterministic_tensor(torch, shape: list[int], salt: int):
    count = math.prod(shape)
    index = torch.arange(count, dtype=torch.int64)
    values = ((index * 1103515245 + salt) & 0xFFFF).to(torch.float32)
    return values.div_(32768.0).sub_(1.0).reshape(shape)


def build_operand(torch, case: dict[str, str], operand: str):
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
    canonical = deterministic_tensor(torch, canonical_shape, salt)
    to_physical = [canonical_labels.index(label) for label in physical_labels]
    physical_values = canonical.permute(to_physical).contiguous()
    physical_shape = [sizes[label] for label in physical_labels]
    padding = case["padding"].split(":")
    if padding[0] == operand:
        padded_label, step = padding[1], int(padding[2])
        padded_axis = physical_labels.index(padded_label)
        backing_shape = list(physical_shape)
        backing_shape[padded_axis] *= step
        backing = torch.empty(backing_shape, dtype=torch.float32)
        selection = [slice(None)] * len(backing_shape)
        selection[padded_axis] = slice(None, None, step)
        physical_view = backing[tuple(selection)]
        physical_view.copy_(physical_values)
    else:
        backing = physical_values
        physical_view = backing
    to_logical = [physical_labels.index(label) for label in canonical_labels]
    logical = physical_view.permute(to_logical)
    return logical, backing, physical_view, canonical


def tensor_metadata(tensor) -> dict[str, object]:
    storage = tensor.untyped_storage()
    return {
        "shape": list(tensor.shape),
        "stride_elements": list(tensor.stride()),
        "stride_bytes": [value * tensor.element_size() for value in tensor.stride()],
        "storage_offset_elements": tensor.storage_offset(),
        "data_ptr": tensor.data_ptr(),
        "storage_data_ptr": storage.data_ptr(),
        "storage_nbytes": storage.nbytes(),
        "logical_nbytes": tensor.numel() * tensor.element_size(),
        "is_contiguous": tensor.is_contiguous(),
        "has_base": tensor._base is not None,
        "dtype": str(tensor.dtype),
    }


def tensor_fingerprint(tensor) -> str:
    contiguous = tensor.detach().contiguous()
    payload = ctypes.string_at(
        contiguous.data_ptr(), contiguous.numel() * contiguous.element_size()
    )
    return hashlib.sha256(payload).hexdigest()


def flatten_tensors(torch, value) -> list:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in flatten_tensors(torch, item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in flatten_tensors(torch, item)]
    return []


def loaded_libraries() -> list[str]:
    paths: set[str] = set()
    with Path("/proc/self/maps").open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip().split()
            if fields and fields[-1].startswith("/") and ".so" in fields[-1]:
                paths.add(fields[-1])
    return sorted(paths)


def task_snapshot() -> dict[int, dict[str, object]]:
    snapshot: dict[int, dict[str, object]] = {}
    for task_dir in Path("/proc/self/task").iterdir():
        if not task_dir.name.isdigit():
            continue
        try:
            stat = (task_dir / "stat").read_text(encoding="utf-8")
            fields = stat[stat.rfind(")") + 2:].split()
            ticks = int(fields[11]) + int(fields[12])
            allowed = ""
            name = ""
            for line in (task_dir / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith("Cpus_allowed_list:"):
                    allowed = line.split(":", 1)[1].strip()
                elif line.startswith("Name:"):
                    name = line.split(":", 1)[1].strip()
            snapshot[int(task_dir.name)] = {
                "ticks": ticks, "affinity": allowed, "name": name,
            }
        except FileNotFoundError:
            continue
    return snapshot


def worker_diagnostic(operation, median_ns: int, requested_threads: int) -> dict[str, object]:
    iterations = max(1, min(20_000, math.ceil(DIAGNOSTIC_TARGET_NS / max(1, median_ns))))
    before = task_snapshot()
    cpu_begin = time.process_time_ns()
    wall_begin = time.perf_counter_ns()
    for _ in range(iterations):
        operation()
    wall_ns = time.perf_counter_ns() - wall_begin
    cpu_ns = time.process_time_ns() - cpu_begin
    after = task_snapshot()
    active: list[dict[str, object]] = []
    for tid, record in after.items():
        delta = int(record["ticks"]) - int(before.get(tid, {"ticks": 0})["ticks"])
        if delta > 0:
            active.append({
                "tid": tid, "name": record["name"], "cpu_ticks": delta,
                "affinity": record["affinity"],
            })
    return {
        "requested_threads": requested_threads,
        "actual_active_workers": len(active),
        "active_workers": active,
        "diagnostic_iterations": iterations,
        "diagnostic_wall_ns": wall_ns,
        "diagnostic_process_cpu_ns": cpu_ns,
        "process_core_equivalents": cpu_ns / wall_ns,
        "requested_capacity_utilization_pct": 100.0 * cpu_ns / (wall_ns * requested_threads),
        "all_task_count": len(after),
    }


def is_descendant(event, ancestor) -> bool:
    parent = event.cpu_parent
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.cpu_parent
    return False


def self_ns(events) -> int:
    return round(sum(event.self_cpu_time_total for event in events) * 1000)


def profiler_diagnostic(prof, audit, case: dict[str, str], method: str,
                        campaign: int) -> dict[str, object]:
    events = list(prof.events())
    roots = [event for event in events if event.key == "benchmark::operation"]
    if not roots:
        raise RuntimeError("profiler did not record benchmark::operation")
    operation_root = max(roots, key=lambda event: event.cpu_time_total)
    descendants = [event for event in events if is_descendant(event, operation_root)]
    einsums = [event for event in descendants if event.key == "aten::einsum"]
    bmms = [event for event in descendants if event.key == "aten::bmm"]
    if not einsums or not bmms:
        raise RuntimeError("expected aten::einsum and aten::bmm in diagnostic")
    einsum = max(einsums, key=lambda event: event.cpu_time_total)
    first_bmm = min(event.time_range.start for event in bmms)
    last_bmm = max(event.time_range.end for event in bmms)
    materialize = [event for event in descendants if event.key in MATERIALIZE_KEYS]
    metadata = [event for event in descendants if event.key in METADATA_KEYS]
    inside_bmm = [
        event for event in materialize
        if any(is_descendant(event, bmm) for bmm in bmms)
    ]
    outside_bmm = [event for event in materialize if event not in inside_bmm]
    input_materialize = [event for event in outside_bmm if event.time_range.end <= first_bmm]
    output_materialize = [event for event in outside_bmm if event.time_range.start >= last_bmm]
    input_metadata = [event for event in metadata if event.time_range.end <= first_bmm]
    output_metadata = [event for event in metadata if event.time_range.start >= last_bmm]
    input_allocation_bytes = sum(
        max(0, int(event.self_cpu_memory_usage)) for event in descendants
        if event.time_range.end <= first_bmm
    )
    role_counts = {role: 0 for role in ("A", "B", "unknown")}
    role_bytes = {role: 0 for role in role_counts}
    for record in audit.materializations:
        role = record["role"] if record["role"] in role_counts else "unknown"
        role_counts[role] += 1
        role_bytes[role] += int(record["bytes"])
    operation_total = round(operation_root.cpu_time_total * 1000)
    bmm_total = round(sum(event.cpu_time_total for event in bmms) * 1000)
    input_materialize_ns = self_ns(input_materialize)
    input_metadata_ns = self_ns(input_metadata)
    output_materialize_ns = self_ns(output_materialize)
    output_metadata_ns = self_ns(output_metadata)
    planner_self_ns = round(einsum.self_cpu_time_total * 1000)
    classified = (
        planner_self_ns + bmm_total + input_materialize_ns + input_metadata_ns
        + output_materialize_ns + output_metadata_ns
    )
    return {
        "case_id": case["case_id"], "method": method, "campaign": campaign,
        "threads": int(case["threads"]),
        "measurement_scope": "profiler_dispatch_diagnostic_not_performance",
        "operation_total_trace_ns": operation_total,
        "einsum_total_trace_ns": round(einsum.cpu_time_total * 1000),
        "planner_self_ns": planner_self_ns,
        "input_materialize_self_ns": input_materialize_ns,
        "input_metadata_self_ns": input_metadata_ns,
        "bmm_container_ns": bmm_total,
        "inside_bmm_materialize_self_ns": self_ns(inside_bmm),
        "output_materialize_self_ns": output_materialize_ns,
        "output_metadata_self_ns": output_metadata_ns,
        "unclassified_trace_ns": max(0, operation_total - classified),
        "input_allocation_bytes": input_allocation_bytes,
        "a_materialization_count": role_counts["A"],
        "a_materialization_bytes": role_bytes["A"],
        "b_materialization_count": role_counts["B"],
        "b_materialization_bytes": role_bytes["B"],
        "unknown_materialization_count": role_counts["unknown"],
        "unknown_materialization_bytes": role_bytes["unknown"],
        "bmm_count": len(bmms),
        "profiler_event_count": len(events),
    }


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
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--export-trace", action="store_true")
    args = parser.parse_args()
    case = read_case(args.manifest, args.case)
    threads = int(case["threads"])
    affinity = {1: {3}, 2: {2, 3}, 4: {0, 1, 2, 3}}[threads]
    os.sched_setaffinity(0, affinity)

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    from torch.utils._mode_utils import no_dispatch
    from torch.utils._python_dispatch import TorchDispatchMode

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    a, a_backing, a_physical, a_canonical = build_operand(torch, case, "A")
    b, b_backing, b_physical, b_canonical = build_operand(torch, case, "B")
    operation = lambda: torch.einsum(case["equation"], a, b)

    reference_path = args.reference_dir / f"{case['scale']}.f64"
    reference = torch.from_file(
        str(reference_path), shared=False, size=int(case["output_elements"]),
        dtype=torch.float64,
    ).reshape(parse_shape(case["c_shape"]))

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
        difference = (result.double() - reference).abs()
        max_abs_error = float(difference.max().item())
        max_rel_error = float((difference / reference.abs().clamp_min(1.0)).max().item())
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
                    event = {
                        "index": len(self.events), "operation": op_name,
                        "input_roles": sorted(input_roles),
                        "inputs": input_meta, "outputs": output_meta,
                    }
                    self.events.append(event)
                    if op_name.startswith("aten.clone"):
                        input_storages = {int(item["storage_data_ptr"]) for item in input_meta}
                        for item in output_meta:
                            if int(item["storage_data_ptr"]) not in input_storages:
                                record = {
                                    "operation": op_name, "role": role,
                                    "bytes": int(item["logical_nbytes"]),
                                    "storage_data_ptr": int(item["storage_data_ptr"]),
                                    "shape": item["shape"], "stride_elements": item["stride_elements"],
                                }
                                self.materializations.append(record)
                return output

        # Keep the timed profiler trace free of Python dispatch instrumentation.
        with torch.inference_mode(), profile(
            activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True
        ) as prof:
            with record_function("benchmark::operation"):
                operation()
        # InferenceMode makes einsum opaque to TorchDispatchMode in PyTorch 2.5.
        # A separate no_grad call exposes the same inner permute/reshape/bmm ops
        # so their real storage identities can be audited without tainting time.
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

    config = torch.__config__.show()
    raw_path = args.output_dir / "raw" / args.method / f"{case['case_id']}_c{args.campaign}.tsv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        raise FileExistsError(f"refusing to overwrite {raw_path}")
    fields = [
        "case_id", "method", "campaign", "repeat", "threads", "affinity",
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
                "campaign": args.campaign, "repeat": repeat, "threads": threads,
                "affinity": ",".join(map(str, sorted(os.sched_getaffinity(0)))),
                "measurement_scope": "framework_e2e_unprofiled",
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
            "campaign": args.campaign, "threads": threads,
            "configured_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "actual_active_workers": thread_diagnostic["actual_active_workers"],
            "active_worker_tids": ",".join(
                str(item["tid"]) for item in thread_diagnostic["active_workers"]
            ),
            "active_worker_affinities": ";".join(
                f"{item['tid']}:{item['affinity']}" for item in thread_diagnostic["active_workers"]
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
        "a_logical_sha256": tensor_fingerprint(a),
        "b_logical_sha256": tensor_fingerprint(b),
        "a_canonical_sha256": tensor_fingerprint(a_canonical),
        "b_canonical_sha256": tensor_fingerprint(b_canonical),
        "max_abs_error": max_abs_error, "max_rel_error": max_rel_error,
        "reference_path": str(reference_path), "trace": str(trace_path) if trace_path else None,
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
