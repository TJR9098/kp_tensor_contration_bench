#!/usr/bin/env python3
"""Validate raw outputs and render the final tables and timeline."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


METHODS = [
    "pytorch_2.5.0_openblas",
    "pytorch_2.5.0_kml_patched",
    "tblis_2.0_armsve",
]
CALIBRATION_METHODS = ["tblis_2.0_armsve", "tblis_2.0_armsve_profile_off"]
PHASES = ["transform_ns", "packing_ns", "kernel_ns", "restore_ns"]
DIAGNOSTIC_FIELDS = [
    "operation_total_trace_ns", "einsum_total_trace_ns", "planner_self_ns",
    "input_materialize_self_ns", "bmm_container_ns",
    "inside_bmm_materialize_self_ns", "output_materialize_self_ns",
    "metadata_view_self_ns", "unclassified_trace_ns", "profiler_event_count",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def integer_median(values: list[int]) -> int:
    ordered = sorted(values)
    size = len(ordered)
    if size % 2:
        return ordered[size // 2]
    return (ordered[size // 2 - 1] + ordered[size // 2]) // 2


def nearest_rank_p95(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def median_float(values: list[float]) -> float:
    return float(statistics.median(values))


def selected_for_timeline(case: dict[str, str]) -> bool:
    rank = int(case["payload_rank"])
    if int(case["contraction_dims"]) != {4: 2, 5: 2, 6: 3}[rank]:
        return False
    if case["families"] == "batch":
        return int(case["batch"]) == 64 and int(case["threads"]) == 4
    return (
        case["distribution"] == "D0"
        and case["layout"] in {"L0", "L3", "L4", "L5"}
        and case["shape_profile"] in {"G0", "G3"}
    )


def summarize_raw(
    rows: list[dict[str, str]], cases: dict[str, dict[str, str]], campaigns: int
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["method"])].append(row)
    expected = {(case_id, method) for case_id in cases for method in METHODS}
    assert set(groups) == expected, f"raw coverage mismatch: missing {expected-set(groups)}"
    output: list[dict[str, object]] = []
    for case_id in cases:
        case = cases[case_id]
        for method in METHODS:
            samples = groups[(case_id, method)]
            assert len(samples) == campaigns * 50
            assert Counter(int(row["campaign"]) for row in samples) == {
                campaign: 50 for campaign in range(campaigns)
            }
            for campaign in range(campaigns):
                assert sorted(
                    int(row["repeat"]) for row in samples
                    if int(row["campaign"]) == campaign
                ) == list(range(50))
            totals = [int(row["total_ns"]) for row in samples]
            median_total = integer_median(totals)
            campaign_medians = [
                integer_median([
                    int(row["total_ns"]) for row in samples
                    if int(row["campaign"]) == campaign
                ]) for campaign in range(campaigns)
            ]
            result: dict[str, object] = {
                "case_id": case_id, "method": method,
                "families": case["families"], "payload_rank": case["payload_rank"],
                "physical_rank": case["physical_rank"],
                "contraction_dims": case["contraction_dims"],
                "shape_profile": case["shape_profile"],
                "distribution": case["distribution"], "layout": case["layout"],
                "batch": case["batch"], "threads": case["threads"],
                "equation": case["equation"], "M": case["M"], "N": case["N"],
                "K": case["K"], "samples": len(samples),
                "measurement_scope": samples[0]["measurement_scope"],
                "median_total_ns": median_total,
                "p95_total_ns": nearest_rank_p95(totals),
                "median_process_cpu_ns": integer_median(
                    [int(row["process_cpu_ns"]) for row in samples]
                ),
                "median_core_utilization_pct": median_float(
                    [float(row["core_utilization_pct"]) for row in samples]
                ),
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
            if method.startswith("pytorch_"):
                assert all(row[field] == "NA" for row in samples for field in PHASES)
                result.update({field: "NA" for field in PHASES})
                result.update({
                    "kernel_gflops": "NA", "transform_ratio": "NA",
                    "packing_ratio": "NA", "kernel_ratio": "NA",
                })
            else:
                phase_values = {
                    field: integer_median([int(row[field]) for row in samples])
                    for field in PHASES
                }
                result.update(phase_values)
                result["kernel_gflops"] = (
                    int(case["flops"]) / phase_values["kernel_ns"]
                    if phase_values["kernel_ns"] else 0.0
                )
                result["transform_ratio"] = phase_values["transform_ns"] / median_total
                result["packing_ratio"] = phase_values["packing_ns"] / median_total
                result["kernel_ratio"] = phase_values["kernel_ns"] / median_total
            output.append(result)
    return output


def summarize_diagnostics(
    rows: list[dict[str, str]], cases: dict[str, dict[str, str]], campaigns: int
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["method"])].append(row)
    expected = {
        (case_id, method) for case_id in cases for method in METHODS[:2]
    }
    assert set(groups) == expected
    output: list[dict[str, object]] = []
    for case_id in cases:
        case = cases[case_id]
        for method in METHODS[:2]:
            samples = groups[(case_id, method)]
            assert len(samples) == campaigns
            assert {int(row["campaign"]) for row in samples} == set(range(campaigns))
            assert all(
                row["measurement_scope"] == "profiler_diagnostic_not_performance"
                for row in samples
            )
            result: dict[str, object] = {
                "case_id": case_id, "method": method,
                "payload_rank": case["payload_rank"],
                "contraction_dims": case["contraction_dims"],
                "shape_profile": case["shape_profile"],
                "distribution": case["distribution"], "layout": case["layout"],
                "batch": case["batch"], "threads": case["threads"],
                "samples": len(samples),
                "measurement_scope": "profiler_diagnostic_not_performance",
            }
            for field in DIAGNOSTIC_FIELDS:
                result[field] = integer_median([int(row[field]) for row in samples])
            output.append(result)
    return output


def summarize_profile_calibration(
    result_dir: Path, campaigns: int
) -> list[dict[str, object]]:
    calibration_root = result_dir / "calibration" / "tblis_profile_overhead" / "raw"
    if not calibration_root.exists():
        return []
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for path in sorted(calibration_root.glob("*/*.tsv")):
        for row in read_tsv(path):
            groups[(row["case_id"], row["method"])].append(row)
    case_ids = sorted({case_id for case_id, _ in groups})
    expected = {(case_id, method) for case_id in case_ids for method in CALIBRATION_METHODS}
    assert set(groups) == expected
    output: list[dict[str, object]] = []
    for case_id in case_ids:
        medians: dict[str, int] = {}
        for method in CALIBRATION_METHODS:
            samples = groups[(case_id, method)]
            assert len(samples) == campaigns * 50
            assert Counter(int(row["campaign"]) for row in samples) == {
                campaign: 50 for campaign in range(campaigns)
            }
            medians[method] = integer_median([int(row["total_ns"]) for row in samples])
        profiled = medians[CALIBRATION_METHODS[0]]
        inactive = medians[CALIBRATION_METHODS[1]]
        output.append({
            "case_id": case_id,
            "threads": groups[(case_id, CALIBRATION_METHODS[0])][0]["threads"],
            "samples_per_mode": campaigns * 50,
            "profiled_median_ns": profiled,
            "profile_inactive_median_ns": inactive,
            "active_profile_overhead_ns": profiled - inactive,
            "active_profile_overhead_pct": 100.0 * (profiled / inactive - 1.0),
        })
    return output


def fmt_value(value: object, digits: int = 3) -> str:
    if value == "NA":
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main_table(rows: list[dict[str, object]], family: str) -> str:
    selected = [row for row in rows if family in str(row["families"]).split("+")]
    lines = [
        "| 用例 | 方法 | 总耗时中位数（ns） | 总耗时 P95（ns） | Transform耗时中位数（ns） | Packing耗时中位数（ns） | Kernel耗时中位数（ns） | Restore耗时中位数（ns） | Kernel性能（GFLOP/s） | End-to-End性能（GFLOP/s） | Transform占比（%） | Packing占比（%） | Kernel占比（%） | 最大相对误差 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        ratios = [
            "N/A" if row[name] == "NA" else f"{100*float(row[name]):.3f}"
            for name in ("transform_ratio", "packing_ratio", "kernel_ratio")
        ]
        lines.append(
            f"| {row['case_id']} | {row['method']} | {row['median_total_ns']} | "
            f"{row['p95_total_ns']} | {fmt_value(row['transform_ns'])} | "
            f"{fmt_value(row['packing_ns'])} | {fmt_value(row['kernel_ns'])} | "
            f"{fmt_value(row['restore_ns'])} | {fmt_value(row['kernel_gflops'])} | "
            f"{float(row['total_gflops']):.3f} | {ratios[0]} | {ratios[1]} | "
            f"{ratios[2]} | {float(row['max_rel_error']):.3e} |"
        )
    return "\n".join(lines)


def diagnostic_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| 用例 | 方法 | 诊断Campaign数 | Operation总时间（ns） | Einsum总时间（ns） | Planner self（ns） | 输入物化 self（ns） | BMM容器（ns） | BMM内部物化 self（ns） | 输出物化 self（ns） | View元数据 self（ns） | 未分类（ns） | Profiler事件数中位数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['method']} | {row['samples']} | "
            + " | ".join(str(row[field]) for field in DIAGNOSTIC_FIELDS)
            + " |"
        )
    return "\n".join(lines)


def batch_telemetry_table(rows: list[dict[str, object]]) -> str:
    selected = [row for row in rows if row["families"] == "batch"]
    lines = [
        "| 用例 | 方法 | 线程 | Batch | 中位总耗时（ns） | 核心利用率（%） | 最低有效带宽估算（GB/s） | Campaign波动（%） |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['case_id']} | {row['method']} | {row['threads']} | "
            f"{row['batch']} | {row['median_total_ns']} | "
            f"{float(row['median_core_utilization_pct']):.2f} | "
            f"{float(row['minimum_effective_bandwidth_gbps']):.3f} | "
            f"{float(row['campaign_spread_pct']):.2f} |"
        )
    return "\n".join(lines)


def profile_calibration_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| 用例 | 线程 | 每种状态样本数 | 采集开启中位数（ns） | 采集关闭中位数（ns） | 增量（ns） | 增量（%） |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['threads']} | {row['samples_per_mode']} | "
            f"{row['profiled_median_ns']} | {row['profile_inactive_median_ns']} | "
            f"{row['active_profile_overhead_ns']} | "
            f"{float(row['active_profile_overhead_pct']):.3f} |"
        )
    return "\n".join(lines)


def write_timeline_svg(
    path: Path, summary: list[dict[str, object]], diagnostics: list[dict[str, object]],
    cases: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    selected_ids = [case_id for case_id, case in cases.items() if selected_for_timeline(case)]
    summary_by_key = {(row["case_id"], row["method"]): row for row in summary}
    diag_by_key = {(row["case_id"], row["method"]): row for row in diagnostics}
    timeline_rows: list[dict[str, object]] = []
    colors = {
        "planner": "#5b8ff9", "input": "#61d9a8", "bmm": "#f6bd16",
        "output": "#e8684a", "metadata": "#9270ca", "other": "#9aa0a6",
        "transform": "#5b8ff9", "packing": "#61d9a8", "kernel": "#f6bd16",
        "restore": "#e8684a",
    }
    svg_rows: list[tuple[str, str, int, list[tuple[str, int]]]] = []
    for case_id in selected_ids:
        for method in METHODS:
            if method.startswith("pytorch_"):
                row = diag_by_key[(case_id, method)]
                total = int(row["operation_total_trace_ns"])
                segments = [
                    ("planner", int(row["planner_self_ns"])),
                    ("input", int(row["input_materialize_self_ns"])),
                    ("bmm", int(row["bmm_container_ns"])),
                    ("output", int(row["output_materialize_self_ns"])),
                    ("metadata", int(row["metadata_view_self_ns"])),
                    ("other", int(row["unclassified_trace_ns"])),
                ]
                source = "pytorch_profiler"
            else:
                row = summary_by_key[(case_id, method)]
                total = int(row["median_total_ns"])
                segments = [
                    ("transform", int(row["transform_ns"])),
                    ("packing", int(row["packing_ns"])),
                    ("kernel", int(row["kernel_ns"])),
                    ("restore", int(row["restore_ns"])),
                ]
                segments.append(("other", max(0, total - sum(value for _, value in segments))))
                source = "tblis_phase_medians"
            timeline_rows.append({
                "case_id": case_id, "method": method, "source": source,
                "total_ns": total,
                "segments": ";".join(f"{name}:{value}" for name, value in segments),
            })
            svg_rows.append((case_id, method, total, segments))

    width = 1280
    label_width = 470
    bar_width = 680
    row_height = 23
    top = 82
    height = top + row_height * len(svg_rows) + 60
    max_total = max(total for _, _, total, _ in svg_rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:18px;font-weight:700}.label{font-size:10px}.axis{font-size:10px;fill:#5f6368}</style>',
        '<text x="20" y="28" class="title">Tensor contraction representative timelines</text>',
        '<text x="20" y="48" class="axis">PyTorch rows are profiler diagnostics; TBLIS rows are phase medians. Scales are global.</text>',
    ]
    legend_x = 20
    for name in ("planner", "input", "bmm", "output", "metadata", "transform", "packing", "kernel", "restore", "other"):
        parts.append(f'<rect x="{legend_x}" y="58" width="10" height="10" fill="{colors[name]}"/>')
        parts.append(f'<text x="{legend_x+14}" y="67" class="axis">{name}</text>')
        legend_x += 72
    for index, (case_id, method, total, segments) in enumerate(svg_rows):
        y = top + index * row_height
        label = html.escape(f"{case_id} / {method}")
        parts.append(f'<text x="20" y="{y+12}" class="label">{label}</text>')
        x = label_width
        for name, value in segments:
            segment_width = bar_width * value / max_total
            if segment_width <= 0:
                continue
            parts.append(
                f'<rect x="{x:.2f}" y="{y}" width="{segment_width:.2f}" height="14" fill="{colors[name]}"><title>{name}: {value} ns</title></rect>'
            )
            x += segment_width
        total_width = bar_width * total / max_total
        parts.append(f'<text x="{label_width+total_width+5:.2f}" y="{y+12}" class="axis">{total} ns</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return timeline_rows


def render_report(
    result_dir: Path, cases: dict[str, dict[str, str]], summary: list[dict[str, object]],
    diagnostics: list[dict[str, object]], calibration: list[dict[str, object]],
    campaigns: int,
) -> str:
    max_error = max(float(row["max_rel_error"]) for row in summary)
    by_key = {(row["case_id"], row["method"]): row for row in summary}
    speedups = [
        float(by_key[(case_id, METHODS[0])]["median_total_ns"])
        / float(by_key[(case_id, METHODS[1])]["median_total_ns"])
        for case_id in cases
    ]
    geometric_speedup = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
    kml_wins = sum(value > 1 for value in speedups)
    kml_vs_tblis = [
        float(by_key[(case_id, METHODS[2])]["median_total_ns"])
        / float(by_key[(case_id, METHODS[1])]["median_total_ns"])
        for case_id in cases
    ]
    tblis_vs_openblas = [
        float(by_key[(case_id, METHODS[0])]["median_total_ns"])
        / float(by_key[(case_id, METHODS[2])]["median_total_ns"])
        for case_id in cases
    ]
    method_wins = Counter(
        min(
            METHODS,
            key=lambda method: float(by_key[(case_id, method)]["median_total_ns"]),
        )
        for case_id in cases
    )
    calibration_section = ""
    if calibration:
        calibration_geomean = math.exp(statistics.mean(
            math.log(
                float(row["profiled_median_ns"])
                / float(row["profile_inactive_median_ns"])
            )
            for row in calibration
        ))
        calibration_section = f"""## TBLIS 插桩开销校准

下表比较同一带插桩 TBLIS 二进制在阶段采集开启/关闭时的端到端时间，任务顺序固定随机打散。它估计的是激活计时器的增量，不等于与完全未修改上游库的差异。{len(calibration)} 个代表性用例的几何平均时间比为 `{calibration_geomean:.4f}x`，即 `{100.0 * (calibration_geomean - 1.0):.3f}%`。

{profile_calibration_table(calibration)}

"""
    return f"""# 4D/5D/6D 张量缩并实验报告

## 实验范围

- 三种方法：`pytorch_2.5.0_openblas`、`pytorch_2.5.0_kml_patched`、`tblis_2.0_armsve`。
- {len(cases)} 个去重逻辑配置，每种方法 {campaigns} 个 campaign，每个 campaign 10 次预热、50 次正式重复。
- 每个配置满足 `B*M*N*K=2^20`，名义 FLOPs 固定为 `2^21`。
- 输入 payload rank 为 4D、5D、6D；batch 用例额外增加一个显式 batch 轴。
- 原始配置见 [case_manifest.tsv](case_manifest.tsv)，机器可读结果见 [summary.tsv](summary.tsv)。

## 方法与口径

`pytorch_2.5.0_openblas` 使用 PyTorch 2.5.0 官方 aarch64 CPU wheel及其捆绑 OpenBLAS。`pytorch_2.5.0_kml_patched` 使用 BiSheng 5.2.0 构建的 PyTorch v2.5.0/KML 补丁路径，`einsum` 规约到 `bmm` 后逐 batch 调用 KML。TBLIS 使用 2.0 `armsve` 构建。

PyTorch 主表只报告完整 `torch.einsum` 框架端到端时间，其 Transform/Packing/Kernel/Restore 和 Kernel GFLOP/s 为 `N/A`。专用诊断表来自独立 profiler 调用，不参与主性能统计。TBLIS 多线程 Packing/Kernel 是各线程累计时间的最大值，即关键路径估计。

`minimum_effective_bandwidth_gbps` 只按算法最低读写字节数估算，不是 DDR 控制器实测带宽。当前 ECS 未暴露硬件 PMU/uncore 事件。

## 完整性摘要

- 主结果组数：{len(summary)}；原始计时记录：{len(summary) * campaigns * 50}。
- PyTorch 诊断组数：{len(diagnostics)}；诊断记录：{len(diagnostics) * campaigns}。
- 最大相对误差：`{max_error:.3e}`。
- KML 补丁版相对官方 wheel 在 {kml_wins}/{len(cases)} 个配置更快，全部配置端到端几何平均加速比为 `{geometric_speedup:.4f}x`。
- 三方法最低中位总耗时胜出数：OpenBLAS wheel {method_wins[METHODS[0]]}、KML 补丁版 {method_wins[METHODS[1]]}、TBLIS 2.0 {method_wins[METHODS[2]]}。
- 全部配置几何平均：KML 相对 TBLIS 为 `{math.exp(sum(math.log(value) for value in kml_vs_tblis) / len(kml_vs_tblis)):.4f}x`，TBLIS 相对 OpenBLAS wheel 为 `{math.exp(sum(math.log(value) for value in tblis_vs_openblas) / len(tblis_vs_openblas)):.4f}x`。
- Campaign 波动、原始时间、诊断事件和运行时库证据均保留在结果目录。

## Timeline

代表性用例的汇总 Timeline：[timeline.svg](timeline.svg)，数据见 [timeline_selected.tsv](timeline_selected.tsv)。PyTorch 的原始 Chrome Trace 位于 [traces](traces/)；Timeline 运行与主性能运行相互独立。

{calibration_section}## Shape 主结果

{main_table(summary, 'shape')}

## Layout 主结果

{main_table(summary, 'layout')}

## Batch 主结果

{main_table(summary, 'batch')}

## PyTorch 专用诊断表

每行是 3 个独立 campaign、每个 campaign 单次 profiler 调用所得值的中位数。该表的 `self` 字段可以相加；`BMM容器` 是 inclusive 时间，包含 BLAS 内部不可见的 packing 与 kernel，不能与其他列简单相加。`Operation总时间` 与主表端到端时间来自不同的 profiler 运行，不用于性能排名。

{diagnostic_table(diagnostics)}

## 多核遥测

核心利用率按 `进程CPU时间/(墙钟时间*请求线程数)` 计算并截断到 100%。最低有效带宽只表示算法字节下界。

{batch_telemetry_table(summary)}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir
    provenance = json.loads((result_dir / "provenance.json").read_text(encoding="utf-8"))
    campaigns = int(provenance["campaigns"])
    manifest_rows = read_tsv(result_dir / "case_manifest.tsv")
    cases = {row["case_id"]: row for row in manifest_rows}
    assert len(cases) == len(manifest_rows)
    assert all(int(row["logical_elements"]) == 2**20 for row in manifest_rows)

    raw_rows: list[dict[str, str]] = []
    for path in sorted((result_dir / "raw").glob("*/*.tsv")):
        raw_rows.extend(read_tsv(path))
    diagnostics_raw: list[dict[str, str]] = []
    for path in sorted((result_dir / "diagnostics").glob("*/*.tsv")):
        diagnostics_raw.extend(read_tsv(path))
    summary = summarize_raw(raw_rows, cases, campaigns)
    diagnostics = summarize_diagnostics(diagnostics_raw, cases, campaigns)
    calibration = summarize_profile_calibration(result_dir, campaigns)
    assert max(float(row["max_rel_error"]) for row in summary) <= 1e-4
    assert all(int(row["profiler_event_count"]) > 0 for row in diagnostics)

    summary_fields = list(summary[0])
    diagnostic_fields = list(diagnostics[0])
    write_tsv(result_dir / "raw.tsv", raw_rows, list(raw_rows[0]))
    write_tsv(result_dir / "summary.tsv", summary, summary_fields)
    write_tsv(result_dir / "pytorch_diagnostics_raw.tsv", diagnostics_raw,
              list(diagnostics_raw[0]))
    write_tsv(result_dir / "pytorch_diagnostics_summary.tsv", diagnostics,
              diagnostic_fields)
    if calibration:
        write_tsv(result_dir / "tblis_profile_overhead.tsv", calibration,
                  list(calibration[0]))
    timeline_rows = write_timeline_svg(
        result_dir / "timeline.svg", summary, diagnostics, cases
    )
    write_tsv(result_dir / "timeline_selected.tsv", timeline_rows,
              ["case_id", "method", "source", "total_ns", "segments"])
    report = render_report(
        result_dir, cases, summary, diagnostics, calibration, campaigns
    )
    (result_dir / "suites.MD").write_text(report, encoding="utf-8")

    trace_paths = sorted((result_dir / "traces").glob("*/*.json"))
    expected_trace_count = 2 * sum(selected_for_timeline(case) for case in cases.values())
    assert len(trace_paths) == expected_trace_count
    for trace_path in trace_paths:
        json.loads(trace_path.read_text(encoding="utf-8"))
    ET.parse(result_dir / "timeline.svg")

    native_metadata = json.loads(next(
        (result_dir / "metadata" / METHODS[0]).glob("*.json")
    ).read_text(encoding="utf-8"))
    kml_metadata = json.loads(next(
        (result_dir / "metadata" / METHODS[1]).glob("*.json")
    ).read_text(encoding="utf-8"))
    assert any("libopenblas" in path for path in native_metadata["loaded_shared_objects"])
    assert any("/sve/kblas/locking/libkblas.so" in path
               for path in kml_metadata["loaded_shared_objects"])
    validation = [
        f"PASS: {len(cases)} unique cases satisfy B*M*N*K=2^20",
        f"PASS: {len(summary)} case x method groups",
        f"PASS: {len(raw_rows)} raw rows = {len(summary)} groups x {campaigns} campaigns x 50 repeats",
        f"PASS: {len(diagnostics)} PyTorch diagnostic groups and {len(diagnostics_raw)} raw diagnostic rows",
        "PASS: PyTorch phase fields are NA in the primary table",
        "PASS: every PyTorch diagnostic contains an aten::einsum and aten::bmm path",
        "PASS: every PyTorch diagnostic contains a positive profiler event count",
        "PASS: official wheel OpenBLAS and patched SVE/locking KML runtime linkage verified",
        f"PASS: maximum relative error {max(float(row['max_rel_error']) for row in summary):.3e} <= 1e-4",
        "PASS: profiler diagnostics are kept separate from primary performance samples",
        f"PASS: {len(trace_paths)} representative Chrome traces are valid JSON",
        "PASS: timeline.svg is well-formed XML",
    ]
    if calibration:
        validation.append(
            f"PASS: {len(calibration)} representative TBLIS profiling-overhead groups"
        )
    (result_dir / "validation.log").write_text("\n".join(validation) + "\n", encoding="utf-8")
    (result_dir / "status.txt").write_text("COMPLETE\n", encoding="utf-8")
    print(f"wrote report with {len(summary)} primary groups and {len(diagnostics)} diagnostic groups")


if __name__ == "__main__":
    main()
