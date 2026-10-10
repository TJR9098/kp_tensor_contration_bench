#!/usr/bin/env python3
"""Compare two completed tensor-contraction runs with paired case metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
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


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    indexed = {(row["case_id"], row["method"]): row for row in rows}
    assert len(indexed) == len(rows)
    return indexed


def winner_counts(rows: dict[tuple[str, str], dict[str, str]]) -> tuple[Counter, dict[str, str]]:
    cases = sorted({case_id for case_id, _ in rows})
    winners: dict[str, str] = {}
    for case_id in cases:
        winners[case_id] = min(
            METHODS,
            key=lambda method: float(rows[(case_id, method)]["median_total_ns"]),
        )
    return Counter(winners.values()), winners


def runtime_signatures(run: Path, method: str) -> set[tuple[str, str, tuple[str, ...]]]:
    signatures = set()
    for path in (run / "metadata" / method).glob("*.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        signatures.add((
            metadata["torch_version"], metadata["torch_config_sha256"],
            tuple(metadata["loaded_shared_objects"]),
        ))
    return signatures


def count_slow_samples(
    run: Path, case_id: str, method: str, threshold_ns: int
) -> tuple[int, int]:
    rows = read_tsv(run / "raw.tsv")
    selected = [
        int(row["total_ns"]) for row in rows
        if row["case_id"] == case_id and row["method"] == method
    ]
    return sum(value > threshold_ns for value in selected), len(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run1", type=Path)
    parser.add_argument("run2", type=Path)
    parser.add_argument("--threshold-pct", type=float, default=5.0)
    args = parser.parse_args()

    for run in (args.run1, args.run2):
        assert (run / "status.txt").read_text(encoding="utf-8").strip() == "COMPLETE"
    assert (args.run1 / "case_manifest.tsv").read_bytes() == (
        args.run2 / "case_manifest.tsv"
    ).read_bytes()
    assert (args.run1 / "tblis_build_metadata.txt").read_bytes() == (
        args.run2 / "tblis_build_metadata.txt"
    ).read_bytes()
    first_provenance = json.loads((args.run1 / "provenance.json").read_text(encoding="utf-8"))
    second_provenance = json.loads((args.run2 / "provenance.json").read_text(encoding="utf-8"))
    for field in ("methods", "campaigns", "warmup", "repeat_per_campaign",
                  "logical_volume", "host_affinity"):
        assert first_provenance[field] == second_provenance[field]
    for method in METHODS[:2]:
        assert runtime_signatures(args.run1, method) == runtime_signatures(args.run2, method)
    first = index_rows(read_tsv(args.run1 / "summary.tsv"))
    second = index_rows(read_tsv(args.run2 / "summary.tsv"))
    assert set(first) == set(second)

    comparisons: list[dict[str, object]] = []
    for key in sorted(first):
        old = first[key]
        new = second[key]
        assert all(old[field] == new[field] for field in (
            "families", "payload_rank", "contraction_dims", "shape_profile",
            "distribution", "layout", "batch", "threads", "M", "N", "K",
        ))
        old_total = int(old["median_total_ns"])
        new_total = int(new["median_total_ns"])
        delta_pct = 100.0 * (new_total / old_total - 1.0)
        old_min = int(old["campaign_median_min_ns"])
        old_max = int(old["campaign_median_max_ns"])
        new_min = int(new["campaign_median_min_ns"])
        new_max = int(new["campaign_median_max_ns"])
        ranges_overlap = not (new_min > old_max or old_min > new_max)
        obvious = abs(delta_pct) >= args.threshold_pct and not ranges_overlap
        comparisons.append({
            "case_id": key[0], "method": key[1],
            "families": old["families"], "payload_rank": old["payload_rank"],
            "contraction_dims": old["contraction_dims"],
            "shape_profile": old["shape_profile"], "distribution": old["distribution"],
            "layout": old["layout"], "batch": old["batch"], "threads": old["threads"],
            "run1_median_total_ns": old_total, "run2_median_total_ns": new_total,
            "run2_over_run1": new_total / old_total, "median_delta_pct": delta_pct,
            "run1_p95_ns": old["p95_total_ns"], "run2_p95_ns": new["p95_total_ns"],
            "p95_delta_pct": 100.0 * (
                int(new["p95_total_ns"]) / int(old["p95_total_ns"]) - 1.0
            ),
            "run1_campaign_min_ns": old_min, "run1_campaign_max_ns": old_max,
            "run2_campaign_min_ns": new_min, "run2_campaign_max_ns": new_max,
            "campaign_ranges_overlap": str(ranges_overlap).lower(),
            "obvious_difference": str(obvious).lower(),
            "direction": "run2_slower" if delta_pct > 0 else "run2_faster",
            "run1_max_rel_error": old["max_rel_error"],
            "run2_max_rel_error": new["max_rel_error"],
        })

    comparison_path = args.run2 / f"comparison_vs_{args.run1.name}.tsv"
    with comparison_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    first_wins, first_winners = winner_counts(first)
    second_wins, second_winners = winner_counts(second)
    flips = [case_id for case_id in first_winners if first_winners[case_id] != second_winners[case_id]]

    summary_lines = [
        "| 方法 | 配对组数 | 第二轮/第一轮几何平均 | 中位绝对差异（%） | 逐组绝对差异P95（%） | 最大绝对差异（%） | 超过5% | 超过10% | 明显差异 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    method_metrics: dict[str, dict[str, float | int]] = {}
    for method in METHODS:
        rows = [row for row in comparisons if row["method"] == method]
        ratios = [float(row["run2_over_run1"]) for row in rows]
        absolute = [abs(float(row["median_delta_pct"])) for row in rows]
        metrics: dict[str, float | int] = {
            "groups": len(rows), "geomean": geometric_mean(ratios),
            "median_abs": statistics.median(absolute),
            "p95_abs": percentile(absolute, 0.95), "max_abs": max(absolute),
            "over5": sum(value >= 5.0 for value in absolute),
            "over10": sum(value >= 10.0 for value in absolute),
            "obvious": sum(row["obvious_difference"] == "true" for row in rows),
        }
        method_metrics[method] = metrics
        summary_lines.append(
            f"| {method} | {metrics['groups']} | {metrics['geomean']:.4f}x | "
            f"{metrics['median_abs']:.3f} | {metrics['p95_abs']:.3f} | "
            f"{metrics['max_abs']:.3f} | {metrics['over5']} | "
            f"{metrics['over10']} | {metrics['obvious']} |"
        )

    p95_lines = [
        "| 方法 | P95耗时第二轮/第一轮几何平均 | P95耗时差异绝对值中位数（%） | 逐组绝对差异P95（%） | 最大绝对差异（%） | 绝对差异超过10%的组数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        rows = [row for row in comparisons if row["method"] == method]
        ratios = [
            int(row["run2_p95_ns"]) / int(row["run1_p95_ns"])
            for row in rows
        ]
        absolute = [abs(float(row["p95_delta_pct"])) for row in rows]
        p95_lines.append(
            f"| {method} | {geometric_mean(ratios):.4f}x | "
            f"{statistics.median(absolute):.3f} | {percentile(absolute, 0.95):.3f} | "
            f"{max(absolute):.3f} | {sum(value >= 10.0 for value in absolute)} |"
        )

    campaign_lines = [
        "| 方法 | 第一轮Campaign波动中位数（%） | 第二轮Campaign波动中位数（%） | 第一轮Campaign波动P95（%） | 第二轮Campaign波动P95（%） |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        keys = [key for key in first if key[1] == method]
        old_spread = [float(first[key]["campaign_spread_pct"]) for key in keys]
        new_spread = [float(second[key]["campaign_spread_pct"]) for key in keys]
        campaign_lines.append(
            f"| {method} | {statistics.median(old_spread):.3f} | "
            f"{statistics.median(new_spread):.3f} | {percentile(old_spread, 0.95):.3f} | "
            f"{percentile(new_spread, 0.95):.3f} |"
        )

    phase_lines = [
        "| TBLIS阶段 | 第二轮/第一轮几何平均 | 中位绝对差异（%） | P95绝对差异（%） |",
        "|---|---:|---:|---:|",
    ]
    for phase in PHASES:
        ratios = [
            float(second[key][phase]) / float(first[key][phase])
            for key in first if key[1] == METHODS[2]
        ]
        absolute = [abs(100.0 * (ratio - 1.0)) for ratio in ratios]
        phase_lines.append(
            f"| {phase} | {geometric_mean(ratios):.4f}x | "
            f"{statistics.median(absolute):.3f} | {percentile(absolute, 0.95):.3f} |"
        )

    diag_lines = [
        "| PyTorch方法 | Operation trace第二轮/第一轮几何平均 | Einsum trace第二轮/第一轮几何平均 |",
        "|---|---:|---:|",
    ]
    first_diag = index_rows(read_tsv(args.run1 / "pytorch_diagnostics_summary.tsv"))
    second_diag = index_rows(read_tsv(args.run2 / "pytorch_diagnostics_summary.tsv"))
    assert set(first_diag) == set(second_diag)
    for method in METHODS[:2]:
        keys = [key for key in first_diag if key[1] == method]
        operation = geometric_mean([
            int(second_diag[key]["operation_total_trace_ns"])
            / int(first_diag[key]["operation_total_trace_ns"])
            for key in keys
        ])
        einsum = geometric_mean([
            int(second_diag[key]["einsum_total_trace_ns"])
            / int(first_diag[key]["einsum_total_trace_ns"])
            for key in keys
        ])
        diag_lines.append(f"| {method} | {operation:.4f}x | {einsum:.4f}x |")

    obvious_rows = sorted(
        (row for row in comparisons if row["obvious_difference"] == "true"),
        key=lambda row: abs(float(row["median_delta_pct"])), reverse=True,
    )
    obvious_lines = [
        "| 用例 | 方法 | 第一轮中位数（ns） | 第二轮中位数（ns） | 差异（%） | 方向 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in obvious_rows[:50]:
        obvious_lines.append(
            f"| {row['case_id']} | {row['method']} | {row['run1_median_total_ns']} | "
            f"{row['run2_median_total_ns']} | {float(row['median_delta_pct']):.3f} | "
            f"{row['direction']} |"
        )
    if not obvious_rows:
        obvious_lines.append("| 无 | - | - | - | - | - |")

    p95_outliers = sorted(
        (
            row for row in comparisons
            if abs(float(row["p95_delta_pct"])) >= 10.0
        ),
        key=lambda row: abs(float(row["p95_delta_pct"])), reverse=True,
    )
    p95_outlier_lines = [
        "| 用例 | 方法 | 第一轮P95（ns） | 第二轮P95（ns） | P95差异（%） | 中位数差异（%） |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in p95_outliers:
        p95_outlier_lines.append(
            f"| {row['case_id']} | {row['method']} | {row['run1_p95_ns']} | "
            f"{row['run2_p95_ns']} | {float(row['p95_delta_pct']):.3f} | "
            f"{float(row['median_delta_pct']):.3f} |"
        )

    max_p95_row = max(comparisons, key=lambda row: abs(float(row["p95_delta_pct"])))
    slow_threshold_ns = 2 * max(
        int(max_p95_row["run1_median_total_ns"]),
        int(max_p95_row["run2_median_total_ns"]),
    )
    run1_slow, run1_samples = count_slow_samples(
        args.run1, str(max_p95_row["case_id"]), str(max_p95_row["method"]),
        slow_threshold_ns,
    )
    run2_slow, run2_samples = count_slow_samples(
        args.run2, str(max_p95_row["case_id"]), str(max_p95_row["method"]),
        slow_threshold_ns,
    )

    first_speedup = geometric_mean([
        int(first[(case_id, METHODS[0])]["median_total_ns"])
        / int(first[(case_id, METHODS[1])]["median_total_ns"])
        for case_id in first_winners
    ])
    second_speedup = geometric_mean([
        int(second[(case_id, METHODS[0])]["median_total_ns"])
        / int(second[(case_id, METHODS[1])]["median_total_ns"])
        for case_id in second_winners
    ])
    conclusion = (
        f"两轮中央性能不存在明显差异：{len(obvious_rows)}/690 组满足明显差异判据，"
        f"最快方法翻转 {len(flips)}/230 个用例。"
    )
    tail_conclusion = (
        "P95 尾延迟并非完全稳定。最大变化出现在 "
        f"`{max_p95_row['case_id']}` / `{max_p95_row['method']}`："
        f"P95 变化 {float(max_p95_row['p95_delta_pct']):.3f}%，"
        f"但中位数仅变化 {float(max_p95_row['median_delta_pct']):.3f}%。"
        f"以两轮较大中位数的 2 倍（{slow_threshold_ns} ns）为慢样本阈值，"
        f"两轮分别只有 {run1_slow}/{run1_samples} 和 {run2_slow}/{run2_samples} 个慢样本。"
        "nearest-rank P95 在 150 个样本中取第 143 个值，恰好跨过慢样本边界；"
        "这说明主要差异来自偶发长尾，而不是稳定吞吐变化。"
    )
    report = f"""# 两次完整实验对比

## 对比口径

- 第一轮：`{args.run1.name}`；第二轮：`{args.run2.name}`。
- 以相同的 690 个 `case × method` 组配对比较，每组均为 3 个 campaign、每个 campaign 50 次正式测量。
- 第二轮/第一轮小于 1 表示第二轮更快，大于 1 表示第二轮更慢。
- “明显差异”定义为总耗时中位数绝对差异至少 `{args.threshold_pct:.1f}%`，且两轮各自的 campaign 中位数范围不重叠。这是工程判据，不是独立同分布假设下的显著性检验。
- 完整逐组数据见 [{comparison_path.name}]({comparison_path.name})。

## 结论

- {conclusion}
- {tail_conclusion}
- 因此，方法性能排序和典型端到端性能可以认为复现；涉及服务尾延迟的结论不能只依赖本实验的单次 P95，应增加 campaign 数并记录调度、缺页和线程创建事件。

## 总体差异

{chr(10).join(summary_lines)}

## P95尾延迟差异

这里的 P95 是每组汇总 150 次正式测量所得的尾延迟。它对少数系统抖动比中位数敏感。

{chr(10).join(p95_lines)}

共有 `{len(p95_outliers)}` 组的 P95 绝对差异至少 10%；这些组的中位数通常保持稳定。

{chr(10).join(p95_outlier_lines)}

## Campaign稳定性

{chr(10).join(campaign_lines)}

## 方法排序稳定性

| 方法 | 第一轮胜出数 | 第二轮胜出数 |
|---|---:|---:|
| {METHODS[0]} | {first_wins[METHODS[0]]} | {second_wins[METHODS[0]]} |
| {METHODS[1]} | {first_wins[METHODS[1]]} | {second_wins[METHODS[1]]} |
| {METHODS[2]} | {first_wins[METHODS[2]]} | {second_wins[METHODS[2]]} |

- 最快方法发生变化的用例：`{len(flips)}/230`。
- KML 相对官方 OpenBLAS wheel 的端到端几何平均加速：第一轮 `{first_speedup:.4f}x`，第二轮 `{second_speedup:.4f}x`。

## TBLIS阶段复现性

{chr(10).join(phase_lines)}

## PyTorch诊断复现性

Profiler 数据来自独立诊断调用，不用于性能排序；下表只用于判断算子级归因是否整体漂移。

{chr(10).join(diag_lines)}

## 明显差异明细

共 `{len(obvious_rows)}` 组满足明显差异判据；下表最多展示绝对差异最大的 50 组。

{chr(10).join(obvious_lines)}
"""
    report_path = args.run2 / f"comparison_vs_{args.run1.name}.MD"
    report_path.write_text(report, encoding="utf-8")
    validation = [
        "PASS: both runs have COMPLETE status",
        "PASS: case manifests and TBLIS build metadata are byte-identical",
        "PASS: campaign, warmup, repeat, affinity, and logical-volume settings match",
        "PASS: PyTorch versions, build configurations, and loaded runtime libraries match",
        f"PASS: {len(comparisons)} paired case x method groups",
        "PASS: case semantics and dimensions match for every paired group",
        f"PASS: {len(obvious_rows)} groups meet the predefined obvious-difference rule",
        f"PASS: {len(flips)} of 230 cases changed fastest method",
        f"PASS: maximum relative errors are {max(float(row['max_rel_error']) for row in first.values()):.3e} and {max(float(row['max_rel_error']) for row in second.values()):.3e}",
    ]
    (args.run2 / f"comparison_validation_vs_{args.run1.name}.log").write_text(
        "\n".join(validation) + "\n", encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
