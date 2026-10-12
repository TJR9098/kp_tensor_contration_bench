#!/usr/bin/env python3
"""Validate, summarize, and report the batched multithreading experiment."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from summarize_tblis_experiment import write_cdf_grid


METHODS = [
    "pytorch_2.5.0_openblas",
    "pytorch_2.5.0_kml_patched",
    "tblis_2.0_armsve_intra",
    "tblis_2.0_armsve_inter",
]
METHOD_LABELS = {
    "pytorch_2.5.0_openblas": "PyTorch-OpenBLAS",
    "pytorch_2.5.0_kml_patched": "PyTorch-KML",
    "tblis_2.0_armsve_intra": "TBLIS-intra",
    "tblis_2.0_armsve_inter": "TBLIS-inter",
}
METHOD_COLORS = {
    "pytorch_2.5.0_openblas": "#4c78a8",
    "pytorch_2.5.0_kml_patched": "#e45756",
    "tblis_2.0_armsve_intra": "#2f9e44",
    "tblis_2.0_armsve_inter": "#b279a2",
}
PHASES = ["transform_ns", "packing_ns", "kernel_ns", "restore_ns"]
SCALES = ("S", "M", "L", "XL")
LAYOUTS = ("X0", "X4")
BATCHES = (1, 4, 32)
THREADS = (1, 2, 4, 8, 32)
MAX_THREADS = max(THREADS)
EXPECTED_CASES = len(SCALES) * len(LAYOUTS) * len(BATCHES) * len(THREADS)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def gather_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("*/*.tsv")):
        rows.extend(read_tsv(path))
    return rows


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "NA") for field in fields})


def integer_median(values: list[int]) -> int:
    return round(statistics.median(values))


def nearest_rank_p95(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def summarize_raw(rows: list[dict[str, str]], cases: dict[str, dict[str, str]],
                  campaigns: int) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["method"])].append(row)
    expected = {(case_id, method) for case_id in cases for method in METHODS}
    if set(groups) != expected:
        raise AssertionError(
            f"raw coverage mismatch: missing={expected-set(groups)} extra={set(groups)-expected}"
        )
    output: list[dict[str, object]] = []
    for case_id, case in cases.items():
        for method in METHODS:
            samples = groups[(case_id, method)]
            assert len(samples) == campaigns * 50
            assert Counter(int(row["campaign"]) for row in samples) == {
                campaign: 50 for campaign in range(campaigns)
            }
            totals = [int(row["total_ns"]) for row in samples]
            campaign_medians = [
                integer_median([
                    int(row["total_ns"]) for row in samples
                    if int(row["campaign"]) == campaign
                ])
                for campaign in range(campaigns)
            ]
            median_total = integer_median(totals)
            batch = int(case["batch"])
            result: dict[str, object] = {
                "case_id": case_id, "method": method,
                "scale": case["scale"], "layout": case["layout"],
                "batch": batch, "threads": int(case["threads"]),
                "equation": case["equation"], "M": int(case["M"]),
                "N": int(case["N"]), "K": int(case["K"]),
                "samples": len(samples), "median_total_ns": median_total,
                "p95_total_ns": nearest_rank_p95(totals),
                "median_per_contraction_ns": median_total / batch,
                "batch_invocations_per_s": 1.0e9 / median_total,
                "contractions_per_s": batch * 1.0e9 / median_total,
                "median_process_cpu_ns": integer_median([
                    int(row["process_cpu_ns"]) for row in samples
                ]),
                "median_process_core_equivalents": statistics.median([
                    float(row["process_core_equivalents"]) for row in samples
                ]),
                "median_core_utilization_pct": statistics.median([
                    float(row["core_utilization_pct"]) for row in samples
                ]),
                "minimum_effective_bandwidth_gbps": int(case["minimum_bytes"]) / median_total,
                "total_gflops": int(case["flops"]) / median_total,
                "max_abs_error": max(float(row["max_abs_error"]) for row in samples),
                "max_rel_error": max(float(row["max_rel_error"]) for row in samples),
                "campaign_median_min_ns": min(campaign_medians),
                "campaign_median_max_ns": max(campaign_medians),
                "campaign_spread_pct": 100.0 * (
                    max(campaign_medians) - min(campaign_medians)
                ) / integer_median(campaign_medians),
            }
            if method == "tblis_2.0_armsve_intra":
                medians = {
                    field: integer_median([int(row[field]) for row in samples])
                    for field in PHASES
                }
                result.update(medians)
                result["kernel_gflops"] = (
                    int(case["flops"]) / medians["kernel_ns"]
                    if medians["kernel_ns"] else 0.0
                )
                result["transform_ratio"] = medians["transform_ns"] / median_total
                result["packing_ratio"] = medians["packing_ns"] / median_total
                result["kernel_ratio"] = medians["kernel_ns"] / median_total
            else:
                assert all(row[field] == "NA" for row in samples for field in PHASES)
                result.update({field: "NA" for field in PHASES})
                result.update({
                    "kernel_gflops": "NA", "transform_ratio": "NA",
                    "packing_ratio": "NA", "kernel_ratio": "NA",
                })
            output.append(result)
    return output


def attach_diagnostics(summary: list[dict[str, object]],
                       rows: list[dict[str, str]]) -> None:
    by_key = {(row["case_id"], row["method"]): row for row in rows}
    assert len(by_key) == len(summary)
    for result in summary:
        row = by_key[(str(result["case_id"]), str(result["method"]))]
        result["configured_threads"] = int(row["configured_threads"])
        result["configured_inner_threads"] = int(row["configured_inner_threads"])
        result["configured_outer_workers"] = int(row["configured_outer_workers"])
        result["actual_active_workers"] = int(row["actual_active_workers"])
        result["diagnostic_core_equivalents"] = float(row["process_core_equivalents"])
        result["diagnostic_capacity_utilization_pct"] = float(
            row["requested_capacity_utilization_pct"]
        )
        result["active_worker_affinities"] = row["active_worker_affinities"]


def add_scaling_metrics(summary: list[dict[str, object]]) -> None:
    by_key = {
        (row["scale"], row["layout"], row["batch"], row["threads"], row["method"]): row
        for row in summary
    }
    for row in summary:
        baseline_threads = by_key[
            (row["scale"], row["layout"], row["batch"], 1, row["method"])
        ]
        speedup = float(baseline_threads["median_total_ns"]) / float(row["median_total_ns"])
        row["speedup_vs_t1"] = speedup
        row["parallel_efficiency"] = speedup / int(row["threads"])
        baseline_batch = by_key[
            (row["scale"], row["layout"], 1, row["threads"], row["method"])
        ]
        row["throughput_vs_b1"] = (
            int(row["batch"]) * float(baseline_batch["median_total_ns"])
            / float(row["median_total_ns"])
        )


def performance_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| 用例 | 方法 | Batch | T | 总耗时中位数（ns） | P95（ns） | 单缩并平均（ns） | E2E GFLOP/s | T1加速比 | 相对B1吞吐 | 核心等效数 | 最大相对误差 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {METHOD_LABELS[str(row['method'])]} | {row['batch']} | "
            f"{row['threads']} | {row['median_total_ns']} | {row['p95_total_ns']} | "
            f"{float(row['median_per_contraction_ns']):.1f} | {float(row['total_gflops']):.3f} | "
            f"{float(row['speedup_vs_t1']):.3f} | {float(row['throughput_vs_b1']):.3f} | "
            f"{float(row['median_process_core_equivalents']):.3f} | "
            f"{float(row['max_rel_error']):.3e} |"
        )
    return "\n".join(lines)


def thread_comparison_table(rows: list[dict[str, object]], threads: int) -> str:
    selected = [row for row in rows if int(row["threads"]) == threads]
    lines = [
        "| 规模/布局/Batch | OpenBLAS（ms） | KML（ms） | TBLIS-intra（ms） | TBLIS-inter（ms） | 最快方法 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    by_key = {(row["case_id"], row["method"]): row for row in selected}
    for scale in SCALES:
        for layout in LAYOUTS:
            for batch in BATCHES:
                case_id = f"{scale}_{layout}_B{batch}_T{threads}"
                values = {
                    method: int(by_key[(case_id, method)]["median_total_ns"])
                    for method in METHODS
                }
                winner = min(values, key=values.get)
                lines.append(
                    f"| {scale}/{layout}/B{batch} | {values[METHODS[0]]/1e6:.3f} | "
                    f"{values[METHODS[1]]/1e6:.3f} | {values[METHODS[2]]/1e6:.3f} | "
                    f"{values[METHODS[3]]/1e6:.3f} | {METHOD_LABELS[winner]} |"
                )
    return "\n".join(lines)


def phase_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| 用例 | Transform（ns） | Packing（ns） | Kernel（ns） | Restore（ns） | Kernel GFLOP/s | Transform占比 | Packing占比 | Kernel占比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["method"] != "tblis_2.0_armsve_intra":
            continue
        lines.append(
            f"| {row['case_id']} | {row['transform_ns']} | {row['packing_ns']} | "
            f"{row['kernel_ns']} | {row['restore_ns']} | {float(row['kernel_gflops']):.3f} | "
            f"{100*float(row['transform_ratio']):.2f}% | "
            f"{100*float(row['packing_ratio']):.2f}% | "
            f"{100*float(row['kernel_ratio']):.2f}% |"
        )
    return "\n".join(lines)


def schedule_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| 用例 | 方法 | 请求T | Inner T | Outer workers | 实际计算worker | 诊断核心等效数 | 亲和性 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if not str(row["method"]).startswith("tblis_"):
            continue
        lines.append(
            f"| {row['case_id']} | {METHOD_LABELS[str(row['method'])]} | {row['threads']} | "
            f"{row['configured_inner_threads']} | {row['configured_outer_workers']} | "
            f"{row['actual_active_workers']} | {float(row['diagnostic_core_equivalents']):.3f} | "
            f"`{row['active_worker_affinities']}` |"
        )
    return "\n".join(lines)


def write_cdfs(result_dir: Path, raw_rows: list[dict[str, str]],
               cases: dict[str, dict[str, str]]) -> tuple[list[dict[str, object]], list[Path]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["case_id"], row["method"])].append(int(row["total_ns"]))
    points: list[dict[str, object]] = []
    for (case_id, method), values in grouped.items():
        for rank, value in enumerate(sorted(values), 1):
            points.append({
                "case_id": case_id, "method": method, "rank": rank,
                "samples": len(values), "end_to_end_ns": value,
                "cdf": rank / len(values),
            })
    cdf_dir = result_dir / "cdf"
    cdf_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    index_lines = ["# 多批次端到端耗时 CDF", ""]
    for case_id, case in cases.items():
        curves = [{
            "legend": METHOD_LABELS[method], "color": METHOD_COLORS[method],
            "values": grouped[(case_id, method)],
        } for method in METHODS]
        path = cdf_dir / f"{case_id}.svg"
        write_cdf_grid(path, f"{case_id} 端到端耗时 CDF", [(case_id, curves)], 1)
        paths.append(path)
        index_lines.append(
            f"- [{case_id}]({case_id}.svg)：{case['scale']}/{case['layout']}/"
            f"B{case['batch']}/T{case['threads']}"
        )
    (cdf_dir / "index.MD").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    batch_dashes = {1: "", 4: "7,4", 32: "2,3"}
    panels = []
    for scale in SCALES:
        for layout in LAYOUTS:
            curves = []
            for batch in BATCHES:
                case_id = f"{scale}_{layout}_B{batch}_T{MAX_THREADS}"
                for method in METHODS:
                    curves.append({
                        "legend": f"{METHOD_LABELS[method]}/B{batch}",
                        "color": METHOD_COLORS[method], "dash": batch_dashes[batch],
                        "values": grouped[(case_id, method)],
                    })
            panels.append((f"{scale}/{layout}/T{MAX_THREADS}", curves))
    overview = result_dir / f"cdf_batch_t{MAX_THREADS}_overview.svg"
    write_cdf_grid(
        overview,
        f"多批次 T{MAX_THREADS} 端到端耗时 CDF",
        panels,
        2,
        x_scale="log",
        subtitle=(
            "横轴：整个 batch 的端到端耗时（µs，对数刻度）；纵轴：经验 CDF；"
            "颜色区分方法，线型区分 B1/B4/B32；每条曲线含 150 个正式样本。"
        ),
    )
    paths.append(overview)
    return points, paths


def write_timeline(result_dir: Path, summary: list[dict[str, object]],
                   diagnostics: list[dict[str, str]]) -> list[dict[str, object]]:
    summary_by = {(row["case_id"], row["method"]): row for row in summary}
    diag_by = {(row["case_id"], row["method"]): row for row in diagnostics}
    selected = [
        f"M_X0_B1_T{MAX_THREADS}", f"M_X4_B32_T{MAX_THREADS}",
        f"XL_X0_B1_T{MAX_THREADS}", f"XL_X4_B32_T{MAX_THREADS}",
    ]
    colors = {
        "planner": "#4c78a8", "input-copy": "#f58518", "bmm": "#54a24b",
        "restore": "#eeca3b", "transform": "#4c78a8", "packing": "#f58518",
        "kernel": "#54a24b", "other": "#bab0ac",
    }
    records: list[dict[str, object]] = []
    rows: list[tuple[str, str, int, list[tuple[str, int]]]] = []
    for case_id in selected:
        for method in METHODS:
            if method.startswith("pytorch_"):
                diag = diag_by[(case_id, method)]
                total = int(diag["operation_total_trace_ns"])
                segments = [
                    ("planner", int(diag["planner_self_ns"])),
                    ("input-copy", int(diag["input_materialize_self_ns"])),
                    ("bmm", int(diag["bmm_container_ns"])),
                    ("restore", int(diag["output_materialize_self_ns"]) + int(diag["output_metadata_self_ns"])),
                ]
                segments.append(("other", max(0, total - sum(v for _, v in segments))))
                source = "PyTorch profiler diagnostic"
            elif method == "tblis_2.0_armsve_intra":
                item = summary_by[(case_id, method)]
                total = int(item["median_total_ns"])
                segments = [(name.removesuffix("_ns"), int(item[name])) for name in PHASES]
                segments.append(("other", max(0, total - sum(v for _, v in segments))))
                source = "TBLIS-intra performance medians"
            else:
                item = summary_by[(case_id, method)]
                total = int(item["median_total_ns"])
                segments = [("other", total)]
                source = "TBLIS-inter E2E; concurrent phase profile unavailable"
            rows.append((case_id, method, total, segments))
            records.append({
                "case_id": case_id, "method": method, "source": source,
                "total_ns": total,
                "segments": ";".join(f"{name}:{value}" for name, value in segments),
            })
    width, label_width, bar_width, row_height, top = 1500, 430, 850, 24, 78
    height = top + len(rows) * row_height + 50
    maximum = max(total for _, _, total, _ in rows)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="18" y="28" font-family="sans-serif" font-size="19" font-weight="600">Batched contraction diagnostic timeline</text>',
        '<text x="18" y="51" font-family="sans-serif" font-size="12" fill="#555">Profiler and unprofiled performance medians use different scopes; TBLIS-inter phases are intentionally unavailable.</text>',
    ]
    for index, (case_id, method, total, segments) in enumerate(rows):
        y = top + index * row_height
        svg.append(
            f'<text x="18" y="{y+13}" font-family="monospace" font-size="10">'
            f'{html.escape(case_id + " " + METHOD_LABELS[method])}</text>'
        )
        x = label_width
        for name, value in segments:
            segment_width = bar_width * value / maximum
            if segment_width > 0:
                svg.append(
                    f'<rect x="{x:.2f}" y="{y}" width="{segment_width:.2f}" height="15" '
                    f'fill="{colors[name]}"><title>{name}: {value} ns</title></rect>'
                )
            x += segment_width
        svg.append(
            f'<text x="{label_width+bar_width+10}" y="{y+12}" '
            f'font-family="monospace" font-size="10">{total} ns</text>'
        )
    svg.append("</svg>")
    (result_dir / "timeline_batch.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
    return records


def conclusions(summary: list[dict[str, object]]) -> str:
    by_key = {(row["case_id"], row["method"]): row for row in summary}
    winner_counts: Counter[str] = Counter()
    case_ids = sorted({str(row["case_id"]) for row in summary})
    for case_id in case_ids:
        winner = min(METHODS, key=lambda method: int(by_key[(case_id, method)]["median_total_ns"]))
        winner_counts[winner] += 1
    lines = ["## 自动摘要", "", "按端到端中位数统计的最快方法数量："]
    for method in METHODS:
        lines.append(
            f"- `{METHOD_LABELS[method]}`：{winner_counts[method]}/{len(case_ids)}。"
        )
    lines.extend([
        "",
        f"T{MAX_THREADS} 下两种 TBLIS 调度的 `inter/intra` 耗时比（小于 1 表示 batch 间并行更快）：",
        "",
    ])
    for scale in SCALES:
        parts = []
        for layout in LAYOUTS:
            for batch in BATCHES:
                case_id = f"{scale}_{layout}_B{batch}_T{MAX_THREADS}"
                intra = float(by_key[(case_id, "tblis_2.0_armsve_intra")]["median_total_ns"])
                inter = float(by_key[(case_id, "tblis_2.0_armsve_inter")]["median_total_ns"])
                parts.append(f"{layout}/B{batch}={inter/intra:.3f}x")
        lines.append(f"- {scale}：" + "，".join(parts) + "。")
    return "\n".join(lines)


def stability_notes(summary: list[dict[str, object]]) -> str:
    tail_rows = []
    campaign_rows = []
    phase_coverage_rows = []
    for row in summary:
        median_ns = float(row["median_total_ns"])
        tail_ratio = float(row["p95_total_ns"]) / median_ns
        campaign_spread = float(row["campaign_spread_pct"])
        tail_rows.append((tail_ratio, row))
        campaign_rows.append((campaign_spread, row))
        if row["method"] == "tblis_2.0_armsve_intra":
            phase_sum = sum(float(row[phase]) for phase in PHASES)
            phase_coverage_rows.append((phase_sum / median_ns, row))

    high_tail = sum(ratio >= 2.0 for ratio, _ in tail_rows)
    high_campaign_spread = sum(spread >= 10.0 for spread, _ in campaign_rows)
    lines = [
        "## 稳定性警示",
        "",
        f"- {high_tail}/{len(summary)} 个方法组的 `P95/中位数 >= 2`；"
        f"{high_campaign_spread}/{len(summary)} 个方法组的 campaign 中位数跨度不小于 10%。",
        "- TBLIS-intra 的阶段计数不包含线程运行时等待；当阶段合计远小于端到端时间时，"
        "差值应解释为同步、调度或未归因运行时开销，不能归入 Kernel。",
        "",
        "长尾最明显的方法组：",
        "",
        "| 用例 | 方法 | P95/中位数 | campaign 跨度 |",
        "|---|---|---:|---:|",
    ]
    for ratio, row in sorted(tail_rows, reverse=True, key=lambda item: item[0])[:10]:
        lines.append(
            f"| {row['case_id']} | {METHOD_LABELS[str(row['method'])]} | "
            f"{ratio:.3f}x | {float(row['campaign_spread_pct']):.2f}% |"
        )

    lines.extend([
        "",
        "TBLIS-intra 阶段合计覆盖端到端时间最低的方法组：",
        "",
        "| 用例 | 阶段合计/总耗时 | 未归因占比 |",
        "|---|---:|---:|",
    ])
    for coverage, row in sorted(phase_coverage_rows, key=lambda item: item[0])[:10]:
        lines.append(
            f"| {row['case_id']} | {100.0 * coverage:.2f}% | "
            f"{100.0 * max(0.0, 1.0 - coverage):.2f}% |"
        )
    return "\n".join(lines)


def render_report(summary: list[dict[str, object]], campaigns: int) -> str:
    method_groups = len(summary)
    configurations = method_groups // len(METHODS)
    return f"""# 大规模多批次多线程张量缩并实验

## 实验目标

在此前 `batch=1` 大单次缩并实验上增加独立 batch，固定表达式 `pabkl,pklcd->pabcd`，比较 PyTorch 两条框架路径以及 TBLIS 的批内、批间两种并行策略。

## 方法实现

- `PyTorch-OpenBLAS`：官方 PyTorch 2.5.0 aarch64 wheel，一次执行完整 batch 的 `torch.einsum`。
- `PyTorch-KML`：KPEX/KML patched PyTorch 2.5.0，一次执行完整 batch 的 `torch.einsum`。
- `TBLIS-intra`：batch 串行；每个 contraction 使用 T 个 TBLIS 内部线程。
- `TBLIS-inter`：建立 `min(T,batch)` 个持久 outer worker，每个 worker 固定到一个 CPU，每个 contraction 的 TBLIS inner thread 数为 1。线程池创建不计入每次 operation，任务分派和同步计入端到端时间。

## 实验协议

- 规模为 S/M/L/XL，布局为 X0/X4，batch 为 1/4/32，线程为 1/2/4/8/32，共 {configurations} 个配置、{method_groups} 个方法组。
- FP32 被测计算；A、B 的每个 batch 均使用独立确定性逻辑值。FP64 参考逐 batch 生成，B1/B4 使用 B32 参考文件的严格前缀。
- 每组 {campaigns} 个 campaign；每轮 warmup 10 次、正式记录 50 次，共 {campaigns * 50} 个样本。
- P95 使用 nearest-rank：排序后第 `ceil(0.95*n)` 个样本，不做插值。
- `total_ns` 是整个 batch 的一次调用；GFLOP/s 使用 `batch*2*M*N*K`，单缩并平均时间为 `total_ns/batch`。
- `相对B1吞吐`=`batch*B1总耗时/当前总耗时`，表示每秒 contraction 数相对同线程 B1 的倍率，不是强扩展 speedup。
- TBLIS-inter 的并发调用不能使用进程全局 profiling ABI，阶段时间明确为 N/A；不把相互干扰的计数写入报告。
- T1/2/4/8/32 分别绑定 CPU3、CPU2-3、CPU0-3、CPU0-7、CPU0-31。当前 ECS 不暴露 DDR/uncore PMU，带宽只能给出算法最少字节估算。

## T{MAX_THREADS} 方法对比

{thread_comparison_table(summary, MAX_THREADS)}

## 完整性能表

{performance_table(summary)}

## TBLIS-intra 阶段统计

{phase_table(summary)}

## TBLIS 调度与线程诊断

{schedule_table(summary)}

## 端到端耗时 CDF

横轴为整个 batch 的端到端耗时，纵轴为经验 CDF。逐用例图见 [`cdf/index.MD`](cdf/index.MD)。

![T{MAX_THREADS}多批次CDF](cdf_batch_t{MAX_THREADS}_overview.svg)

## Timeline

![代表性Timeline](timeline_batch.svg)

PyTorch 条带来自独立 profiler 调用；TBLIS-intra 来自正式样本阶段中位数；TBLIS-inter 只显示端到端总时间。三者用于解释路径，不能将 profiler 时间直接替代主表时间。

{conclusions(summary)}

{stability_notes(summary)}

## 结果文件

- `summary.tsv`：{method_groups} 个方法组的汇总结果。
- `raw.tsv`：全部正式样本。
- `thread_diagnostics.tsv`：inner/outer 配置、实际 worker、核心等效数和亲和性。
- `materialization_diagnostics.tsv`：PyTorch 输入物化与 TBLIS wrapper/phase 诊断。
- `layout_audit.tsv`：入口 shape、stride、存储和逻辑哈希。
- `cdf_points.tsv`、`cdf/` 和 `cdf_batch_t{MAX_THREADS}_overview.svg`：CDF 精确数据与图形。
- `timeline_batch.svg`、`timeline_selected.tsv`：代表性 timeline。
- `validation.log`、`provenance.json`：自动校验与构件哈希。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir
    status = (result_dir / "status.txt").read_text(encoding="utf-8").strip()
    if status not in {"DATA_COMPLETE", "COMPLETE"}:
        raise SystemExit("result directory is not DATA_COMPLETE or COMPLETE")
    provenance = json.loads((result_dir / "provenance.json").read_text(encoding="utf-8"))
    campaigns = int(provenance["campaigns"])
    case_rows = read_tsv(result_dir / "case_manifest.tsv")
    cases = {row["case_id"]: row for row in case_rows}
    assert len(cases) == EXPECTED_CASES
    raw_rows = gather_rows(result_dir / "raw")
    materialization_rows = gather_rows(result_dir / "materialization")
    thread_rows = gather_rows(result_dir / "thread_diagnostics")
    summary = summarize_raw(raw_rows, cases, campaigns)
    attach_diagnostics(summary, thread_rows)
    add_scaling_metrics(summary)

    metadata: dict[tuple[str, str], dict[str, object]] = {}
    for method in METHODS:
        for path in sorted((result_dir / "metadata" / method).glob("*_c0.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            case_id = item["case"]["case_id"] if "case" in item else item["case_id"]
            metadata[(case_id, method)] = item
    assert len(metadata) == len(cases) * len(METHODS)
    for case_id, case in cases.items():
        hashes = {
            (metadata[(case_id, method)]["a_logical_sha256"],
             metadata[(case_id, method)]["b_logical_sha256"])
            for method in METHODS
        }
        assert len(hashes) == 1, f"logical hash mismatch: {case_id}"
        strides = {
            (tuple(metadata[(case_id, method)]["a_entry"]["stride_elements"]),
             tuple(metadata[(case_id, method)]["b_entry"]["stride_elements"]))
            for method in METHODS
        }
        assert len(strides) == 1, f"entry stride mismatch: {case_id}"
        for method in METHODS[2:]:
            item = metadata[(case_id, method)]
            assert item["a_entry"]["data_ptr"] == item["tblis_a_data_ptr"]
            assert item["b_entry"]["data_ptr"] == item["tblis_b_data_ptr"]
        assert int(case["flops"]) == int(case["batch"]) * 2 * int(case["M"]) * int(case["N"]) * int(case["K"])
    for scale in SCALES:
        for batch in BATCHES:
            logical_hashes = {
                (metadata[(case_id, method)]["a_logical_sha256"],
                 metadata[(case_id, method)]["b_logical_sha256"])
                for case_id, case in cases.items()
                if case["scale"] == scale and int(case["batch"]) == batch
                for method in METHODS
            }
            assert len(logical_hashes) == 1, (
                f"logical values changed with layout/thread: {scale}/B{batch}"
            )
    assert max(float(row["max_rel_error"]) for row in summary) <= 1.0e-4

    thread_by = {(row["case_id"], row["method"]): row for row in thread_rows}
    for case_id, case in cases.items():
        threads = int(case["threads"])
        batch = int(case["batch"])
        intra = thread_by[(case_id, "tblis_2.0_armsve_intra")]
        assert int(intra["configured_inner_threads"]) == threads
        assert int(intra["configured_outer_workers"]) == 1
        assert int(intra["actual_active_workers"]) == threads
        inter = thread_by[(case_id, "tblis_2.0_armsve_inter")]
        expected_outer = min(threads, batch)
        assert int(inter["configured_inner_threads"]) == 1
        assert int(inter["configured_outer_workers"]) == expected_outer
        assert int(inter["actual_active_workers"]) == expected_outer

    first_case = next(iter(cases))
    native = metadata[(first_case, METHODS[0])]
    kml = metadata[(first_case, METHODS[1])]
    assert any("libopenblas" in path for path in native["loaded_shared_objects"])
    assert any("/sve/kblas/locking/libkblas.so" in path for path in kml["loaded_shared_objects"])

    write_tsv(result_dir / "raw.tsv", raw_rows)
    write_tsv(result_dir / "summary.tsv", summary)
    write_tsv(result_dir / "materialization_diagnostics.tsv", materialization_rows)
    write_tsv(result_dir / "thread_diagnostics.tsv", thread_rows)
    audit_rows = []
    for (case_id, method), item in metadata.items():
        audit_rows.append({
            "case_id": case_id, "method": method,
            "a_shape": item["a_entry"]["shape"],
            "a_stride_elements": item["a_entry"]["stride_elements"],
            "a_storage_nbytes": item["a_entry"]["storage_nbytes"],
            "a_logical_sha256": item["a_logical_sha256"],
            "b_shape": item["b_entry"]["shape"],
            "b_stride_elements": item["b_entry"]["stride_elements"],
            "b_storage_nbytes": item["b_entry"]["storage_nbytes"],
            "b_logical_sha256": item["b_logical_sha256"],
        })
    write_tsv(result_dir / "layout_audit.tsv", audit_rows)
    cdf_points, cdf_paths = write_cdfs(result_dir, raw_rows, cases)
    write_tsv(result_dir / "cdf_points.tsv", cdf_points)
    timeline = write_timeline(result_dir, summary, materialization_rows)
    write_tsv(result_dir / "timeline_selected.tsv", timeline)
    (result_dir / "suites.MD").write_text(
        render_report(summary, campaigns), encoding="utf-8"
    )

    trace_paths = sorted((result_dir / "traces").glob("*/*.json"))
    expected_traces = 2 * sum(
        case["scale"] in {"M", "XL"}
        and int(case["threads"]) in {1, MAX_THREADS}
        and ((case["layout"] == "X0" and int(case["batch"]) == 1)
             or (case["layout"] == "X4" and int(case["batch"]) == 32))
        for case in cases.values()
    )
    assert len(trace_paths) == expected_traces
    for path in trace_paths:
        json.loads(path.read_text(encoding="utf-8"))
    ET.parse(result_dir / "timeline_batch.svg")
    for path in cdf_paths:
        ET.parse(path)
    validation = [
        f"PASS: {EXPECTED_CASES} unique S/M/L/XL x X0/X4 x B1/B4/B32 x T1/T2/T4/T8/T32 cases",
        f"PASS: {len(summary)} case-method groups and {len(raw_rows)} raw samples",
        f"PASS: every group contains {campaigns} campaigns x 50 samples",
        "PASS: batch FLOPs equal batch * 2*M*N*K for every case",
        "PASS: logical input SHA256 and entry strides match across all four methods",
        "PASS: logical input values are invariant across X0/X4 and T1/T2/T4/T8/T32",
        "PASS: TBLIS wrapper pointers equal the audited entry pointers",
        "PASS: TBLIS intra/inter inner/outer worker configurations match the protocol",
        "PASS: official OpenBLAS and patched SVE/locking KML runtime linkage verified",
        f"PASS: maximum relative error {max(float(row['max_rel_error']) for row in summary):.3e} <= 1e-4",
        f"PASS: {len(thread_rows)} thread and {len(materialization_rows)} materialization diagnostics",
        f"PASS: {len(trace_paths)} PyTorch Chrome traces are valid JSON",
        f"PASS: {len(cdf_points)} ECDF points and {len(cdf_paths)} CDF SVG files",
        "INFO: TBLIS-inter phase fields are N/A because the profiling ABI is process-global",
        "INFO: hardware DDR bandwidth is N/A because no DDR/uncore PMU is exposed",
    ]
    (result_dir / "validation.log").write_text("\n".join(validation) + "\n", encoding="utf-8")
    (result_dir / "status.txt").write_text("COMPLETE\n", encoding="utf-8")
    print(f"wrote report with {len(summary)} groups and {len(raw_rows)} raw samples")


if __name__ == "__main__":
    main()
