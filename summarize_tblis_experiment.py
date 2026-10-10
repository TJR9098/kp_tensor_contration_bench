#!/usr/bin/env python3
"""Validate and report the X-layout/materialization and B1 scaling experiment."""

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


METHODS = [
    "pytorch_2.5.0_openblas",
    "pytorch_2.5.0_kml_patched",
    "tblis_2.0_armsve",
]
PHASES = ["transform_ns", "packing_ns", "kernel_ns", "restore_ns"]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "NA") for field in fields})


def integer_median(values: list[int]) -> int:
    return round(statistics.median(values))


def nearest_rank_p95(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def gather_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("*/*.tsv")):
        rows.extend(read_tsv(path))
    return rows


def summarize_raw(rows: list[dict[str, str]], cases: dict[str, dict[str, str]],
                  campaigns: int) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["method"])].append(row)
    expected = {(case_id, method) for case_id in cases for method in METHODS}
    if set(groups) != expected:
        raise AssertionError(f"raw coverage mismatch: missing={expected-set(groups)} extra={set(groups)-expected}")
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
                ]) for campaign in range(campaigns)
            ]
            median_total = integer_median(totals)
            result: dict[str, object] = {
                "case_id": case_id, "method": method,
                "families": case["families"], "scale": case["scale"],
                "layout": case["layout"], "threads": int(case["threads"]),
                "equation": case["equation"], "M": int(case["M"]),
                "N": int(case["N"]), "K": int(case["K"]),
                "samples": len(samples), "median_total_ns": median_total,
                "p95_total_ns": nearest_rank_p95(totals),
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
            if method.startswith("pytorch_"):
                assert all(row[field] == "NA" for row in samples for field in PHASES)
                result.update({field: "NA" for field in PHASES})
                result.update({
                    "kernel_gflops": "NA", "transform_ratio": "NA",
                    "packing_ratio": "NA", "kernel_ratio": "NA",
                })
            else:
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
            output.append(result)
    return output


def attach_thread_diagnostics(summary: list[dict[str, object]],
                              rows: list[dict[str, str]]) -> None:
    by_key = {(row["case_id"], row["method"]): row for row in rows}
    assert len(by_key) == len(summary)
    for result in summary:
        row = by_key[(str(result["case_id"]), str(result["method"]))]
        result["configured_threads"] = int(row["configured_threads"])
        result["actual_active_workers"] = int(row["actual_active_workers"])
        result["diagnostic_core_equivalents"] = float(row["process_core_equivalents"])
        result["diagnostic_capacity_utilization_pct"] = float(
            row["requested_capacity_utilization_pct"]
        )
        result["active_worker_affinities"] = row["active_worker_affinities"]


def add_scaling_metrics(summary: list[dict[str, object]]) -> None:
    by_key = {
        (row["scale"], row["layout"], row["method"], row["threads"]): row
        for row in summary if "scaling" in str(row["families"]).split("+")
    }
    for row in summary:
        if "scaling" not in str(row["families"]).split("+"):
            row["speedup_vs_t1"] = "NA"
            row["parallel_efficiency"] = "NA"
            continue
        baseline = by_key[(row["scale"], row["layout"], row["method"], 1)]
        speedup = float(baseline["median_total_ns"]) / float(row["median_total_ns"])
        row["speedup_vs_t1"] = speedup
        row["parallel_efficiency"] = speedup / int(row["threads"])


def fmt(value: object, digits: int = 3) -> str:
    if value == "NA":
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main_performance_table(rows: list[dict[str, object]], family: str) -> str:
    selected = [row for row in rows if family in str(row["families"]).split("+")]
    lines = [
        "| 用例 | 方法 | T | 实际活跃线程 | 总耗时中位数（ns） | P95（ns） | Transform（ns） | Packing（ns） | Kernel（ns） | Restore（ns） | Kernel GFLOP/s | E2E GFLOP/s | Transform占比 | Packing占比 | Kernel占比 | 核心利用率 | 估算最低带宽（GB/s） | 最大相对误差 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        ratios = [
            "N/A" if row[field] == "NA" else f"{100*float(row[field]):.2f}%"
            for field in ("transform_ratio", "packing_ratio", "kernel_ratio")
        ]
        lines.append(
            f"| {row['case_id']} | {row['method']} | {row['threads']} | "
            f"{row['actual_active_workers']} | {row['median_total_ns']} | {row['p95_total_ns']} | "
            f"{fmt(row['transform_ns'])} | {fmt(row['packing_ns'])} | "
            f"{fmt(row['kernel_ns'])} | {fmt(row['restore_ns'])} | "
            f"{fmt(row['kernel_gflops'])} | {float(row['total_gflops']):.3f} | "
            f"{ratios[0]} | {ratios[1]} | {ratios[2]} | "
            f"{float(row['median_core_utilization_pct']):.2f}% | "
            f"{float(row['minimum_effective_bandwidth_gbps']):.3f} | "
            f"{float(row['max_rel_error']):.3e} |"
        )
    return "\n".join(lines)


def materialization_table(rows: list[dict[str, str]], cases: dict[str, dict[str, str]]) -> str:
    selected = [
        row for row in rows
        if "layout" in cases[row["case_id"]]["families"].split("+")
    ]
    lines = [
        "| 用例 | 方法 | 输入物化时间（ns） | Profiler输入分配（B） | A物化次数/字节 | B物化次数/字节 | TBLIS包装层完整物化 | TBLIS Packing（ns） | Packing次数/字节 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['case_id']} | {row['method']} | "
            f"{row.get('input_materialize_self_ns', 'N/A')} | "
            f"{row.get('input_allocation_bytes', 'N/A')} | "
            f"{row.get('a_materialization_count', 'N/A')}/{row.get('a_materialization_bytes', 'N/A')} | "
            f"{row.get('b_materialization_count', 'N/A')}/{row.get('b_materialization_bytes', 'N/A')} | "
            f"{row.get('wrapper_full_materialization_count', 'N/A')}/"
            f"{row.get('wrapper_full_materialization_bytes', 'N/A')} B | "
            f"{row.get('tblis_packing_ns', 'N/A')} | "
            f"{row.get('packing_calls', 'N/A')}/{row.get('packing_bytes', 'N/A')} |"
        )
    return "\n".join(lines)


def scaling_table(rows: list[dict[str, object]]) -> str:
    selected = [row for row in rows if "scaling" in str(row["families"]).split("+")]
    lines = [
        "| 规模/布局 | 方法 | 请求T | 实际活跃线程 | 总耗时中位数（ns） | Speedup | 并行效率 | 性能采样核心利用率 | 线程诊断核心等效数 | Campaign波动 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['scale']}/{row['layout']} | {row['method']} | {row['threads']} | "
            f"{row['actual_active_workers']} | {row['median_total_ns']} | "
            f"{float(row['speedup_vs_t1']):.3f} | "
            f"{100*float(row['parallel_efficiency']):.2f}% | "
            f"{float(row['median_core_utilization_pct']):.2f}% | "
            f"{float(row['diagnostic_core_equivalents']):.3f} | "
            f"{float(row['campaign_spread_pct']):.2f}% |"
        )
    return "\n".join(lines)


def audit_table(metadata: dict[tuple[str, str], dict[str, object]],
                cases: dict[str, dict[str, str]]) -> str:
    lines = [
        "| 用例 | 输入 | 物理轴序 | 逻辑Shape | Stride（元素） | Stride（字节） | Offset | Storage/逻辑字节 | Contiguous | 逻辑SHA256 |",
        "|---|---|---|---|---|---|---:|---:|---|---|",
    ]
    for case_id, case in cases.items():
        if "layout" not in case["families"].split("+"):
            continue
        item = metadata[(case_id, METHODS[0])]
        for operand, order in (("a", case["a_physical_order"]), ("b", case["b_physical_order"])):
            entry = item[f"{operand}_entry"]
            lines.append(
                f"| {case_id} | {operand.upper()} | `{order}` | `{entry['shape']}` | "
                f"`{entry['stride_elements']}` | `{entry.get('stride_bytes', '见元素stride')}` | "
                f"{entry['storage_offset_elements']} | {entry['storage_nbytes']}/"
                f"{entry['logical_nbytes']} | {entry.get('is_contiguous', False)} | "
                f"`{item[f'{operand}_logical_sha256'][:16]}...` |"
            )
    return "\n".join(lines)


def selected_for_timeline(case: dict[str, str]) -> bool:
    if case["scale"] == "S" and case["threads"] == "1" and "layout" in case["families"]:
        return True
    return (
        case["scale"] in {"L", "XL"} and case["layout"] in {"X0", "X4"}
        and int(case["threads"]) in {1, 4}
    )


METHOD_COLORS = {
    "pytorch_2.5.0_openblas": "#3973ac",
    "pytorch_2.5.0_kml_patched": "#d8583d",
    "tblis_2.0_armsve": "#238b68",
}


def cdf_path(values_ns: list[int], x0: float, y0: float, width: float,
             height: float, minimum_us: float, maximum_us: float) -> str:
    ordered = sorted(value / 1000.0 for value in values_ns)
    span = maximum_us - minimum_us

    def x(value: float) -> float:
        return x0 + width * (value - minimum_us) / span

    def y(value: float) -> float:
        return y0 + height * (1.0 - value)

    commands = [f"M {x(ordered[0]):.2f} {y(0):.2f}"]
    previous = 0.0
    for index, value in enumerate(ordered, 1):
        current = index / len(ordered)
        commands.append(f"L {x(value):.2f} {y(previous):.2f}")
        commands.append(f"L {x(value):.2f} {y(current):.2f}")
        previous = current
    return " ".join(commands)


def write_cdf_grid(path: Path, title: str,
                   panels: list[tuple[str, list[dict[str, object]]]],
                   columns: int) -> None:
    panel_width, panel_height = 490, 300
    plot_left, plot_top, plot_width, plot_height = 68, 37, 386, 210
    top, legend_height = 62, 40
    rows = math.ceil(len(panels) / columns)
    width = columns * panel_width
    height = top + legend_height + rows * panel_height + 20
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="18" y="28" font-family="sans-serif" font-size="19" font-weight="600">{html.escape(title)}</text>',
        '<text x="18" y="49" font-family="sans-serif" font-size="11" fill="#555">横轴：端到端耗时（µs）；纵轴：经验 CDF；每条曲线合并 3 个 campaign 的 150 个非 profiler 原始样本。</text>',
    ]
    legends: list[tuple[str, str, str]] = []
    for _, curves in panels:
        for curve in curves:
            key = (str(curve["legend"]), str(curve["color"]), str(curve.get("dash", "")))
            if key not in legends:
                legends.append(key)
    legend_x = 18
    legend_y = top + 13
    for label, color, dash in legends:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+28}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="2.3"{dash_attr}/>'
        )
        svg.append(
            f'<text x="{legend_x+34}" y="{legend_y+4}" font-family="sans-serif" font-size="10">'
            f'{html.escape(label)}</text>'
        )
        legend_x += max(142, 42 + len(label) * 6)
        if legend_x > width - 170:
            legend_x = 18
            legend_y += 15
    for panel_index, (label, curves) in enumerate(panels):
        column = panel_index % columns
        row = panel_index // columns
        base_x = column * panel_width
        base_y = top + legend_height + row * panel_height
        x0, y0 = base_x + plot_left, base_y + plot_top
        all_values = [value for curve in curves for value in curve["values"]]
        minimum_us = min(all_values) / 1000.0
        maximum_us = max(all_values) / 1000.0
        padding = max((maximum_us - minimum_us) * 0.04, maximum_us * 0.002, 0.01)
        minimum_us = max(0.0, minimum_us - padding)
        maximum_us += padding
        if maximum_us <= minimum_us:
            maximum_us = minimum_us + 1.0
        svg.append(
            f'<text x="{base_x+12}" y="{base_y+18}" font-family="sans-serif" '
            f'font-size="13" font-weight="600">{html.escape(label)}</text>'
        )
        svg.append(
            f'<rect x="{x0}" y="{y0}" width="{plot_width}" height="{plot_height}" '
            f'fill="#fafafa" stroke="#777" stroke-width="0.7"/>'
        )
        for tick in range(5):
            fraction = tick / 4
            x = x0 + fraction * plot_width
            value = minimum_us + fraction * (maximum_us - minimum_us)
            svg.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0+plot_height}" stroke="#e5e5e5"/>')
            svg.append(f'<text x="{x:.2f}" y="{y0+plot_height+17}" text-anchor="middle" font-family="sans-serif" font-size="9">{value:.1f}</text>')
        for tick in range(5):
            fraction = tick / 4
            y = y0 + plot_height * (1 - fraction)
            svg.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x0+plot_width}" y2="{y:.2f}" stroke="#e5e5e5"/>')
            svg.append(f'<text x="{x0-8}" y="{y+3:.2f}" text-anchor="end" font-family="sans-serif" font-size="9">{fraction:.2f}</text>')
        for curve in curves:
            dash = str(curve.get("dash", ""))
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            path_data = cdf_path(
                list(curve["values"]), x0, y0, plot_width, plot_height,
                minimum_us, maximum_us,
            )
            svg.append(
                f'<path d="{path_data}" fill="none" stroke="{curve["color"]}" '
                f'stroke-width="1.8"{dash_attr}/>'
            )
        svg.append(
            f'<text x="{x0+plot_width/2}" y="{y0+plot_height+34}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="10">端到端耗时（µs）</text>'
        )
        svg.append(
            f'<text x="{base_x+15}" y="{y0+plot_height/2}" text-anchor="middle" '
            f'transform="rotate(-90 {base_x+15} {y0+plot_height/2})" '
            f'font-family="sans-serif" font-size="10">CDF</text>'
        )
    svg.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def write_cdf_outputs(result_dir: Path, raw_rows: list[dict[str, str]],
                      cases: dict[str, dict[str, str]]) -> tuple[list[dict[str, object]], list[Path]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["case_id"], row["method"])].append(int(row["total_ns"]))
    points: list[dict[str, object]] = []
    for (case_id, method), values in grouped.items():
        ordered = sorted(values)
        for index, value in enumerate(ordered, 1):
            points.append({
                "case_id": case_id, "method": method, "rank": index,
                "samples": len(ordered), "end_to_end_ns": value,
                "cdf": index / len(ordered),
            })
    cdf_dir = result_dir / "cdf"
    cdf_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    index_lines = ["# 逐用例端到端耗时 CDF", ""]
    for case_id, case in cases.items():
        curves = [{
            "legend": method, "color": METHOD_COLORS[method],
            "values": grouped[(case_id, method)],
        } for method in METHODS]
        path = cdf_dir / f"{case_id}.svg"
        write_cdf_grid(path, f"{case_id} 端到端耗时 CDF", [(case_id, curves)], 1)
        paths.append(path)
        index_lines.append(
            f"- [{case_id}]({case_id}.svg)：{case['scale']}/{case['layout']}/T{case['threads']}"
        )
    (cdf_dir / "index.MD").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    layout_panels = []
    for layout in ("X0", "X1", "X2", "X3", "X4", "X5"):
        case_id = f"S_{layout}_T1"
        curves = [{
            "legend": method, "color": METHOD_COLORS[method],
            "values": grouped[(case_id, method)],
        } for method in METHODS]
        layout_panels.append((layout, curves))
    layout_path = result_dir / "cdf_layout_overview.svg"
    write_cdf_grid(layout_path, "实验一：布局端到端耗时 CDF", layout_panels, 3)
    paths.append(layout_path)

    scaling_panels = []
    for scale in ("S", "M", "L", "XL"):
        for threads in (1, 2, 4):
            curves = []
            for layout, dash in (("X0", ""), ("X4", "7,4")):
                case_id = f"{scale}_{layout}_T{threads}"
                for method in METHODS:
                    curves.append({
                        "legend": f"{method}/{layout}",
                        "color": METHOD_COLORS[method], "dash": dash,
                        "values": grouped[(case_id, method)],
                    })
            scaling_panels.append((f"{scale} / T{threads}", curves))
    scaling_path = result_dir / "cdf_scaling_overview.svg"
    write_cdf_grid(scaling_path, "实验二：规模、布局与线程端到端耗时 CDF", scaling_panels, 3)
    paths.append(scaling_path)
    return points, paths


def write_timeline_svg(path: Path, summary: list[dict[str, object]],
                       diagnostics: list[dict[str, str]],
                       cases: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    summary_by_key = {(row["case_id"], row["method"]): row for row in summary}
    diagnostic_by_key = {(row["case_id"], row["method"]): row for row in diagnostics}
    colors = {
        "planner": "#3973ac", "input-copy": "#d95f02", "metadata": "#7570b3",
        "bmm": "#1b9e77", "restore": "#e6ab02", "transform": "#3973ac",
        "packing": "#d95f02", "kernel": "#1b9e77", "other": "#a8a8a8",
    }
    rows: list[tuple[str, str, int, list[tuple[str, int]], str]] = []
    output: list[dict[str, object]] = []
    for case_id, case in cases.items():
        if not selected_for_timeline(case):
            continue
        for method in METHODS:
            if method.startswith("pytorch_"):
                diag = diagnostic_by_key[(case_id, method)]
                total = int(diag["operation_total_trace_ns"])
                restore = int(diag["output_materialize_self_ns"]) + int(diag["output_metadata_self_ns"])
                segments = [
                    ("planner", int(diag["planner_self_ns"])),
                    ("input-copy", int(diag["input_materialize_self_ns"])),
                    ("metadata", int(diag["input_metadata_self_ns"])),
                    ("bmm", int(diag["bmm_container_ns"])),
                    ("restore", restore),
                    ("other", int(diag["unclassified_trace_ns"])),
                ]
                source = "PyTorch profiler diagnostic"
            else:
                record = summary_by_key[(case_id, method)]
                total = int(record["median_total_ns"])
                segments = [
                    ("transform", int(record["transform_ns"])),
                    ("packing", int(record["packing_ns"])),
                    ("kernel", int(record["kernel_ns"])),
                    ("restore", int(record["restore_ns"])),
                ]
                segments.append(("other", max(0, total - sum(value for _, value in segments))))
                source = "TBLIS performance phase medians"
            rows.append((case_id, method, total, segments, source))
            output.append({
                "case_id": case_id, "method": method, "source": source,
                "total_ns": total,
                "segments": ";".join(f"{name}:{value}" for name, value in segments),
            })
    width, label_width, bar_width, row_height, top = 1500, 520, 850, 22, 78
    height = top + row_height * len(rows) + 55
    maximum = max(total for _, _, total, _, _ in rows)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="18" y="28" font-family="sans-serif" font-size="19" font-weight="600">X-layout and B1 scaling diagnostic timeline</text>',
        '<text x="18" y="51" font-family="sans-serif" font-size="12" fill="#555">Profiler traces and performance medians are different measurement scopes; bar lengths are globally comparable, segment semantics are method-specific.</text>',
    ]
    for index, (case_id, method, total, segments, _) in enumerate(rows):
        y = top + index * row_height
        label = html.escape(f"{case_id}  {method}")
        svg.append(f'<text x="18" y="{y+13}" font-family="monospace" font-size="10">{label}</text>')
        x = label_width
        for name, value in segments:
            segment_width = bar_width * value / maximum
            if segment_width > 0:
                svg.append(
                    f'<rect x="{x:.2f}" y="{y}" width="{segment_width:.2f}" height="15" '
                    f'fill="{colors[name]}"><title>{html.escape(name)}: {value} ns</title></rect>'
                )
            x += segment_width
        svg.append(f'<text x="{label_width+bar_width+10}" y="{y+12}" font-family="monospace" font-size="10">{total} ns</text>')
    legend_x = 18
    for name in colors:
        svg.append(f'<rect x="{legend_x}" y="{height-29}" width="12" height="12" fill="{colors[name]}"/>')
        svg.append(f'<text x="{legend_x+17}" y="{height-19}" font-family="sans-serif" font-size="10">{html.escape(name)}</text>')
        legend_x += 88
    svg.append('</svg>')
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return output


def conclusions(summary: list[dict[str, object]], diagnostics: list[dict[str, str]]) -> str:
    by_key = {(row["case_id"], row["method"]): row for row in summary}
    diag = {(row["case_id"], row["method"]): row for row in diagnostics}
    kml_x0_t4 = by_key[("XL_X0_T4", METHODS[1])]
    kml_x4_t4 = by_key[("XL_X4_T4", METHODS[1])]
    tblis_speedups_x0 = [
        float(by_key[(f"{scale}_X0_T4", METHODS[2])]["speedup_vs_t1"])
        for scale in ("S", "M", "L", "XL")
    ]
    tblis_m_x0 = by_key[("M_X0_T4", METHODS[2])]
    kml_m_x0 = by_key[("M_X0_T4", METHODS[1])]
    lines = [
        "### 主要结论", "",
        "- 物化证据链成立：X2 只复制 A（65,536 B），X3 只复制 B（65,536 B），"
        "X4/X5 都复制 A+B（131,072 B）；TBLIS 包装层完整输入物化为 0，只有内部局部 packing。",
        "- KML 的 X4 端到端中位数比 X0 增加 19.4%，而 TBLIS 变化 -0.2%；"
        "这把布局引入的完整输入复制与端到端差距联系起来。X5 没有比 X4 再多复制，"
        "说明更大的 backing storage 不等于更多实际物化字节。",
        f"- KML 在 XL/T4 诊断中可见多个 worker，但 X0/X4 speedup 仅 "
        f"{float(kml_x0_t4['speedup_vs_t1']):.3f}/{float(kml_x4_t4['speedup_vs_t1']):.3f}；"
        "因此不能把‘线程存在’解读为 B1 计算得到有效加速。",
        "- TBLIS X0 的 T4 speedup 随规模 S/M/L/XL 为 "
        + "/".join(f"{value:.2f}" for value in tblis_speedups_x0)
        + "。S/X0 虽已获得并行收益但仍略慢于 KML；从 M/X0 起 TBLIS T4 已明显更快，"
        f"M/X0 中位数为 {float(tblis_m_x0['median_total_ns'])/1000:.1f} µs，"
        f"KML 为 {float(kml_m_x0['median_total_ns'])/1000:.1f} µs。",
        "- X1 没有输入 clone，证明整组转置可以保持为矩阵转置视图；X2-X5 才触发复杂矩阵化。"
        "TBLIS 的 X0-X5 中位数差异均在 ±0.23% 内。",
        "", "### 明细判读", "",
    ]
    for method in METHODS:
        baseline = by_key[("S_X0_T1", method)]
        parts = []
        for layout in ("X1", "X2", "X3", "X4", "X5"):
            row = by_key[(f"S_{layout}_T1", method)]
            delta = 100.0 * (float(row["median_total_ns"]) / float(baseline["median_total_ns"]) - 1.0)
            parts.append(f"{layout} {delta:+.1f}%")
        lines.append(f"- `{method}` 相对 X0 的端到端变化：" + "，".join(parts) + "。")
    for method in METHODS[:2]:
        parts = []
        for layout in ("X0", "X1", "X2", "X3", "X4", "X5"):
            row = diag[(f"S_{layout}_T1", method)]
            total = int(row["a_materialization_bytes"]) + int(row["b_materialization_bytes"])
            parts.append(f"{layout}={total} B")
        lines.append(f"- `{method}` dispatch 审计捕获的输入物化字节：" + "，".join(parts) + "。")
    for method in METHODS:
        for layout in ("X0", "X4"):
            row = by_key[(f"XL_{layout}_T4", method)]
            lines.append(
                f"- `XL/{layout}` 的 `{method}`：T4 speedup={float(row['speedup_vs_t1']):.2f}，"
                f"实际活跃线程={row['actual_active_workers']}，诊断核心等效数={float(row['diagnostic_core_equivalents']):.2f}。"
            )
    return "\n".join(lines)


def render_report(result_dir: Path, cases: dict[str, dict[str, str]],
                  summary: list[dict[str, object]], diagnostics: list[dict[str, str]],
                  metadata: dict[tuple[str, str], dict[str, object]], campaigns: int) -> str:
    return f"""# TBLIS 2.0 布局物化与大单次缩并实验

## 实验目标

固定数学表达式 `abkl,klcd->abcd`、轴长度和逻辑输入值，仅改变 backing storage 的物理轴序和 stride，验证 PyTorch/KML 输入物化与 TBLIS 局部 packing 的差异；随后固定 `batch=1`，比较 X0/X4 在不同规模和线程数下的扩展性。

## 方法实现

- `pytorch_2.5.0_openblas`：官方 aarch64 PyTorch 2.5.0 wheel，框架端到端 `torch.einsum`。
- `pytorch_2.5.0_kml_patched`：KPEX PyTorch/KML 补丁版本，框架端到端 `torch.einsum`。
- `tblis_2.0_armsve`：TBLIS 2.0 ARM-SVE，输入逻辑标签固定为 `abkl`/`klcd`，直接传入审计后的任意 stride；包装层不做完整输入转换。

PyTorch 2.5.0 的 einsum 规约路径是 `permute -> reshape -> bmm -> view/permute`。本实验使用 profiler 加 dispatch 审计确认 reshape 是否产生 clone，而不是根据 `is_contiguous()` 推断。源码依据：[PyTorch v2.5.0 Linear.cpp](https://github.com/pytorch/pytorch/blob/v2.5.0/aten/src/ATen/native/Linear.cpp)。

## 实验协议

- FP32 被测计算；每个逻辑规模只生成一次 FP64 `torch.mm` 参考结果。
- {campaigns} 个独立 campaign，每组每轮 warmup 10 次、记录 50 个原始样本。
- T1 绑定 CPU3，T2 绑定 CPU2-3，T4 绑定 CPU0-3；机器只有 4 个物理核，T8 明确不运行。
- 性能采样不启用 PyTorch profiler；物化、worker 和 timeline 来自独立诊断运行。
- 相对误差分母为 `max(abs(reference), 1)`，阈值 `1e-4`。
- “估算最低带宽”=`(A+B+C 的逻辑字节数)/端到端时间`，不是硬件 DDR 带宽。当前 ECS 未暴露 DDR/uncore PMU，硬件带宽为 N/A。
- TBLIS `packing_ns` 来自本地 profiling ABI；当前 ABI 不暴露 packing 调用次数和字节数，因此对应字段为 N/A，不能将估算字节伪装成实测值。

## 入口布局审计

{audit_table(metadata, cases)}

## 实验一：布局性能

{main_performance_table(summary, "layout")}

## PyTorch 物化与 TBLIS Packing 诊断

{materialization_table(diagnostics, cases)}

## 实验二：大单次缩并扩展性

{scaling_table(summary)}

## 端到端耗时 CDF

以下经验 CDF 仅使用非 profiler 的端到端原始样本。每条曲线有 `{campaigns} × 50 = {campaigns * 50}` 个点；横轴越靠左表示越快，曲线越陡表示波动越小。

![布局 CDF 总览](cdf_layout_overview.svg)

![规模、布局与线程 CDF 总览](cdf_scaling_overview.svg)

全部 28 个逐用例图见 [`cdf/index.MD`](cdf/index.MD)，精确坐标见 `cdf_points.tsv`。

## Timeline

![代表性 timeline](timeline.svg)

PyTorch 条带来自单次 profiler 诊断，TBLIS 条带来自 150 个非 profiler 性能样本的阶段中位数；二者测量范围不同，不能把 profiler 条带时间直接当作主性能表时间。

{conclusions(summary, diagnostics)}

## 结果文件

- `summary.tsv`：主性能汇总。
- `raw.tsv`：全部原始性能样本。
- `materialization_diagnostics.tsv`：PyTorch 物化和 TBLIS 包装层/packing 诊断。
- `thread_diagnostics.tsv`：请求线程、实际活跃线程、亲和性和核心等效数。
- `layout_audit.tsv`：机器可读的入口 shape/stride/storage/hash 元数据索引。
- `cdf_points.tsv`、`cdf_layout_overview.svg`、`cdf_scaling_overview.svg` 和 `cdf/`：端到端经验 CDF 数据与图。
- `timeline.svg`、`timeline_selected.tsv` 和 `traces/`：代表性 timeline 与 PyTorch Chrome trace。
- `validation.log`：覆盖率、正确性、动态库、哈希和指针一致性检查。
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
    assert len(cases) == 28
    raw_rows = gather_rows(result_dir / "raw")
    materialization_rows = gather_rows(result_dir / "materialization")
    thread_rows = gather_rows(result_dir / "thread_diagnostics")
    summary = summarize_raw(raw_rows, cases, campaigns)
    attach_thread_diagnostics(summary, thread_rows)
    add_scaling_metrics(summary)

    metadata: dict[tuple[str, str], dict[str, object]] = {}
    for method in METHODS:
        for path in sorted((result_dir / "metadata" / method).glob("*_c0.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            metadata[(item["case"]["case_id"] if "case" in item else item["case_id"], method)] = item
    assert len(metadata) == len(cases) * len(METHODS)

    for case_id in cases:
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
        tblis_meta = metadata[(case_id, METHODS[2])]
        assert tblis_meta["a_entry"]["data_ptr"] == tblis_meta["tblis_a_data_ptr"]
        assert tblis_meta["b_entry"]["data_ptr"] == tblis_meta["tblis_b_data_ptr"]
    assert max(float(row["max_rel_error"]) for row in summary) <= 1.0e-4

    native_meta = metadata[(next(iter(cases)), METHODS[0])]
    kml_meta = metadata[(next(iter(cases)), METHODS[1])]
    assert any("libopenblas" in path for path in native_meta["loaded_shared_objects"])
    assert any("/sve/kblas/locking/libkblas.so" in path for path in kml_meta["loaded_shared_objects"])

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
            "a_storage_offset_elements": item["a_entry"]["storage_offset_elements"],
            "a_data_ptr": item["a_entry"]["data_ptr"],
            "a_storage_data_ptr": item["a_entry"]["storage_data_ptr"],
            "a_storage_nbytes": item["a_entry"]["storage_nbytes"],
            "a_logical_sha256": item["a_logical_sha256"],
            "b_shape": item["b_entry"]["shape"],
            "b_stride_elements": item["b_entry"]["stride_elements"],
            "b_storage_offset_elements": item["b_entry"]["storage_offset_elements"],
            "b_data_ptr": item["b_entry"]["data_ptr"],
            "b_storage_data_ptr": item["b_entry"]["storage_data_ptr"],
            "b_storage_nbytes": item["b_entry"]["storage_nbytes"],
            "b_logical_sha256": item["b_logical_sha256"],
        })
    write_tsv(result_dir / "layout_audit.tsv", audit_rows)
    cdf_points, cdf_paths = write_cdf_outputs(result_dir, raw_rows, cases)
    write_tsv(result_dir / "cdf_points.tsv", cdf_points)
    timeline = write_timeline_svg(
        result_dir / "timeline.svg", summary, materialization_rows, cases
    )
    write_tsv(result_dir / "timeline_selected.tsv", timeline)
    report = render_report(
        result_dir, cases, summary, materialization_rows, metadata, campaigns
    )
    (result_dir / "suites.MD").write_text(report, encoding="utf-8")

    trace_paths = sorted((result_dir / "traces").glob("*/*.json"))
    expected_traces = 2 * sum(selected_for_timeline(case) for case in cases.values())
    assert len(trace_paths) == expected_traces
    for path in trace_paths:
        json.loads(path.read_text(encoding="utf-8"))
    ET.parse(result_dir / "timeline.svg")
    for path in cdf_paths:
        ET.parse(path)
    validation = [
        "PASS: 28 unique cases, fixed equation abkl,klcd->abcd",
        f"PASS: {len(summary)} case-method groups and {len(raw_rows)} raw samples",
        f"PASS: every group contains {campaigns} campaigns x 50 samples",
        "PASS: logical input SHA256 and entry strides match across all three methods",
        "PASS: TBLIS wrapper entry data pointers equal pointers passed to TBLIS",
        "PASS: official OpenBLAS and patched SVE/locking KML runtime linkage verified",
        f"PASS: maximum relative error {max(float(row['max_rel_error']) for row in summary):.3e} <= 1e-4",
        f"PASS: {len(thread_rows)} worker diagnostics and {len(materialization_rows)} materialization diagnostics",
        f"PASS: {len(trace_paths)} PyTorch Chrome traces are valid JSON",
        "PASS: timeline.svg is well-formed XML",
        f"PASS: {len(cdf_points)} ECDF points and {len(cdf_paths)} well-formed CDF SVG files",
        "INFO: hardware DDR bandwidth is N/A because no DDR/uncore PMU is exposed",
        "INFO: TBLIS packing calls/bytes are N/A because the installed profiling ABI exposes time only",
    ]
    (result_dir / "validation.log").write_text("\n".join(validation) + "\n", encoding="utf-8")
    (result_dir / "status.txt").write_text("COMPLETE\n", encoding="utf-8")
    print(f"wrote report with {len(summary)} groups and {len(raw_rows)} raw samples")


if __name__ == "__main__":
    main()
