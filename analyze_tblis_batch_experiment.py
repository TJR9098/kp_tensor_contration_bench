#!/usr/bin/env python3
"""Generate the consolidated CDF and analysis for the 32-vCPU batched experiment."""

from __future__ import annotations

import argparse
import math
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from summarize_tblis_batch_experiment import (
    METHODS,
    METHOD_COLORS,
    METHOD_LABELS,
    read_tsv,
)
from summarize_tblis_experiment import write_cdf_grid


SCALES = ("S", "M", "L", "XL")
LAYOUTS = ("X0", "X4")
BATCHES = (1, 4, 32)
THREADS = (1, 2, 4, 8, 32)
MAX_THREADS = max(THREADS)
BATCH_DASHES = {1: "", 4: "7,4", 32: "2,3"}


def median(values: list[float]) -> float:
    return statistics.median(values)


def table(lines: list[list[str]], align: str) -> str:
    header = "| " + " | ".join(lines[0]) + " |"
    separator = "|" + align + "|"
    rows = ["| " + " | ".join(row) + " |" for row in lines[1:]]
    return "\n".join([header, separator, *rows])


def write_combined_cdf(result_dir: Path, raw: list[dict[str, str]]) -> Path:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in raw:
        grouped[(row["case_id"], row["method"])].append(int(row["total_ns"]))
    panels: list[tuple[str, list[dict[str, object]]]] = []
    for scale in SCALES:
        for layout in LAYOUTS:
            for threads in THREADS:
                curves = []
                for batch in BATCHES:
                    case_id = f"{scale}_{layout}_B{batch}_T{threads}"
                    for method in METHODS:
                        values = grouped[(case_id, method)]
                        if len(values) != 150:
                            raise AssertionError(
                                f"{case_id}/{method}: expected 150 samples, got {len(values)}"
                            )
                        curves.append({
                            "legend": f"{METHOD_LABELS[method]}/B{batch}",
                            "color": METHOD_COLORS[method],
                            "dash": BATCH_DASHES[batch],
                            "values": values,
                        })
                panels.append((f"{scale}/{layout}/T{threads}", curves))
    output = result_dir / "cdf_batch_all_overview.svg"
    write_cdf_grid(
        output,
        "多批次、多线程端到端耗时 CDF 总览",
        panels,
        columns=len(THREADS),
        x_scale="log",
        subtitle=(
            "横轴：整个 batch 的端到端耗时（µs，对数刻度）；纵轴：经验 CDF；"
            "颜色区分方法，线型区分 B1/B4/B32；每条曲线含 150 个正式样本。"
        ),
    )
    return output


def winner_table(summary: list[dict[str, str]]) -> str:
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in summary:
        by_case[row["case_id"]].append(row)
    counts: dict[int | str, Counter[str]] = {
        batch: Counter() for batch in BATCHES
    }
    counts["all"] = Counter()
    for rows in by_case.values():
        winner = min(rows, key=lambda row: int(row["median_total_ns"]))["method"]
        batch = int(rows[0]["batch"])
        counts[batch][winner] += 1
        counts["all"][winner] += 1
    lines = [["Batch", *[METHOD_LABELS[method] for method in METHODS], "配置数"]]
    for batch in (*BATCHES, "all"):
        label = f"B{batch}" if isinstance(batch, int) else "全部"
        per_batch = len(SCALES) * len(LAYOUTS) * len(THREADS)
        total = per_batch if isinstance(batch, int) else per_batch * len(BATCHES)
        lines.append([label, *[str(counts[batch][method]) for method in METHODS], str(total)])
    return table(lines, "---|---:|---:|---:|---:|---:")


def speedup_table(summary: list[dict[str, str]], threads: int) -> str:
    lines = [["方法", "B1", "B4", "B32"]]
    for method in METHODS:
        row = [METHOD_LABELS[method]]
        for batch in BATCHES:
            values = [
                float(item["speedup_vs_t1"])
                for item in summary
                if item["method"] == method and int(item["batch"]) == batch
                and int(item["threads"]) == threads
            ]
            row.append(f"{median(values):.3f}x")
        lines.append(row)
    return table(lines, "---|---:|---:|---:")


def throughput_table(summary: list[dict[str, str]], threads: int) -> str:
    lines = [["方法", "B4/B1 吞吐", "B32/B1 吞吐", "B32/B1 总耗时"]]
    for method in METHODS:
        selected = [
            row for row in summary
            if row["method"] == method and int(row["threads"]) == threads
        ]
        b4 = median([
            float(row["throughput_vs_b1"]) for row in selected if int(row["batch"]) == 4
        ])
        b32 = median([
            float(row["throughput_vs_b1"]) for row in selected if int(row["batch"]) == 32
        ])
        by_shape = {
            (row["scale"], row["layout"], int(row["batch"])): row
            for row in selected
        }
        latency = median([
            int(by_shape[(scale, layout, 32)]["median_total_ns"])
            / int(by_shape[(scale, layout, 1)]["median_total_ns"])
            for scale in SCALES for layout in LAYOUTS
        ])
        lines.append([METHOD_LABELS[method], f"{b4:.3f}x", f"{b32:.3f}x", f"{latency:.3f}x"])
    return table(lines, "---|---:|---:|---:")


def thread_gflops_table(summary: list[dict[str, str]], threads: int) -> str:
    lines = [["规模", *[METHOD_LABELS[method] for method in METHODS]]]
    for scale in SCALES:
        row = [scale]
        for method in METHODS:
            values = {}
            for batch in BATCHES:
                values[batch] = statistics.mean([
                    float(item["total_gflops"])
                    for item in summary
                    if item["scale"] == scale and item["method"] == method
                    and int(item["batch"]) == batch and int(item["threads"]) == threads
                ])
            row.append(" / ".join(f"{values[batch]:.1f}" for batch in BATCHES))
        lines.append(row)
    return table(lines, "---|---:|---:|---:|---:")


def schedule_table(summary: list[dict[str, str]], threads: int) -> str:
    by = {
        (row["scale"], row["layout"], int(row["batch"]), int(row["threads"]), row["method"]): row
        for row in summary
    }
    intra = "tblis_2.0_armsve_intra"
    inter = "tblis_2.0_armsve_inter"
    kml = "pytorch_2.5.0_kml_patched"
    lines = [["规模", "X0 inter/intra", "X4 inter/intra", "X0 inter/KML", "X4 inter/KML"]]
    for scale in SCALES:
        row = [scale]
        for layout in LAYOUTS:
            row.append(
                f"{int(by[(scale, layout, 32, threads, inter)]['median_total_ns']) / int(by[(scale, layout, 32, threads, intra)]['median_total_ns']):.3f}x"
            )
        for layout in LAYOUTS:
            row.append(
                f"{int(by[(scale, layout, 32, threads, inter)]['median_total_ns']) / int(by[(scale, layout, 32, threads, kml)]['median_total_ns']):.3f}x"
            )
        lines.append(row)
    return table(lines, "---|---:|---:|---:|---:")


def layout_table(
    summary: list[dict[str, str]], diagnostics: list[dict[str, str]], threads: int
) -> str:
    by = {
        (row["scale"], row["layout"], int(row["batch"]), int(row["threads"]), row["method"]): row
        for row in summary
    }
    diag = {(row["case_id"], row["method"]): row for row in diagnostics}
    lines = [["规模", "KML X4 clone", *[METHOD_LABELS[method] for method in METHODS]]]
    for scale in SCALES:
        case_id = f"{scale}_X4_B32_T{threads}"
        item = diag[(case_id, "pytorch_2.5.0_kml_patched")]
        clone_bytes = int(item["a_materialization_bytes"]) + int(item["b_materialization_bytes"])
        row = [scale, f"{clone_bytes / 2**20:.1f} MiB"]
        for method in METHODS:
            x0 = int(by[(scale, "X0", 32, threads, method)]["median_total_ns"])
            x4 = int(by[(scale, "X4", 32, threads, method)]["median_total_ns"])
            row.append(f"{x4 / x0:.3f}x")
        lines.append(row)
    return table(lines, "---|---:|---:|---:|---:|---:")


def distribution_table(summary: list[dict[str, str]], raw: list[dict[str, str]]) -> str:
    medians = {(row["case_id"], row["method"]): int(row["median_total_ns"]) for row in summary}
    raw_by: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        batch = int(row["case_id"].split("_B", 1)[1].split("_", 1)[0])
        raw_by[(batch, row["method"])].append(row)
    lines = [["Batch", "方法", "P95/中位数中位值", "最大值", ">2x中位数样本"]]
    for batch in BATCHES:
        for method in METHODS:
            selected = [
                row for row in summary
                if int(row["batch"]) == batch and row["method"] == method
            ]
            ratios = [int(row["p95_total_ns"]) / int(row["median_total_ns"]) for row in selected]
            samples = raw_by[(batch, method)]
            outliers = sum(
                int(row["total_ns"]) > 2 * medians[(row["case_id"], method)]
                for row in samples
            )
            lines.append([
                f"B{batch}", METHOD_LABELS[method], f"{median(ratios):.3f}",
                f"{max(ratios):.3f}",
                f"{outliers}/{len(samples)} ({100*outliers/len(samples):.2f}%)",
            ])
    return table(lines, "---|---|---:|---:|---:")


def tblis_tail_table(summary: list[dict[str, str]], threads: int) -> str:
    method = "tblis_2.0_armsve_intra"
    lines = [["Batch", "P95/中位数中位值", "最大值", "最大值用例"]]
    for batch in BATCHES:
        selected = [
            row for row in summary
            if row["method"] == method and int(row["batch"]) == batch
            and int(row["threads"]) == threads
        ]
        ratios = {
            row["case_id"]: int(row["p95_total_ns"]) / int(row["median_total_ns"])
            for row in selected
        }
        worst = max(ratios, key=ratios.get)
        lines.append([f"B{batch}", f"{median(list(ratios.values())):.3f}", f"{ratios[worst]:.3f}", worst])
    return table(lines, "---|---:|---:|---")


def campaign_table(summary: list[dict[str, str]]) -> str:
    lines = [["Batch", "方法", "波动中位值", "最大值", "最大值用例"]]
    for batch in BATCHES:
        for method in METHODS:
            selected = [
                row for row in summary
                if int(row["batch"]) == batch and row["method"] == method
            ]
            worst = max(selected, key=lambda row: float(row["campaign_spread_pct"]))
            lines.append([
                f"B{batch}", METHOD_LABELS[method],
                f"{median([float(row['campaign_spread_pct']) for row in selected]):.2f}%",
                f"{float(worst['campaign_spread_pct']):.2f}%", worst["case_id"],
            ])
    return table(lines, "---|---|---:|---:|---")


def nearest_rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def cross_run_table(
    summary: list[dict[str, str]], previous: list[dict[str, str]] | None
) -> str:
    if previous is None:
        return "未提供 4 vCPU 汇总表，无法执行同配置比较。"
    previous_by = {(row["case_id"], row["method"]): row for row in previous}
    lines = [[
        "方法", "匹配配置", "绝对偏差中位值", "绝对偏差 P95", "最大偏差", "最大偏差用例"
    ]]
    for method in METHODS:
        deltas: dict[str, float] = {}
        for row in summary:
            key = (row["case_id"], method)
            if row["method"] != method or key not in previous_by:
                continue
            old = previous_by[key]
            deltas[row["case_id"]] = abs(
                int(row["median_total_ns"]) - int(old["median_total_ns"])
            ) / int(old["median_total_ns"])
        worst = max(deltas, key=deltas.get)
        values = list(deltas.values())
        lines.append([
            METHOD_LABELS[method], str(len(deltas)), f"{100*median(values):.2f}%",
            f"{100*nearest_rank(values, 0.95):.2f}%",
            f"{100*deltas[worst]:.2f}%", worst,
        ])
    return table(lines, "---|---:|---:|---:|---:|---")


def winner_consistency(
    summary: list[dict[str, str]], previous: list[dict[str, str]] | None
) -> str:
    if previous is None:
        return "未提供 4 vCPU 汇总表。"

    def winners(rows: list[dict[str, str]]) -> dict[str, str]:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row["case_id"]].append(row)
        return {
            case_id: min(items, key=lambda item: int(item["median_total_ns"]))["method"]
            for case_id, items in grouped.items()
        }

    old = winners(previous)
    current = winners([row for row in summary if row["case_id"] in old])
    changes = [(case_id, old[case_id], current[case_id]) for case_id in old if old[case_id] != current[case_id]]
    lines = [["共同配置", "4 vCPU 最快", "32 vCPU 重跑最快"]]
    for case_id, old_method, current_method in changes:
        lines.append([case_id, METHOD_LABELS[old_method], METHOD_LABELS[current_method]])
    prefix = (
        f"共同 T1/T2/T4 配置中 {len(old) - len(changes)}/{len(old)} 个最快方法一致；"
        f"变化 {len(changes)} 个。\n\n"
    )
    return prefix + table(lines, "---|---|---")


def optimal_thread_table(summary: list[dict[str, str]]) -> str:
    workloads: dict[tuple[str, str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in summary:
        workloads[(row["scale"], row["layout"], int(row["batch"]), row["method"])].append(row)
    counts: dict[str, Counter[int]] = {method: Counter() for method in METHODS}
    speedups: dict[str, list[float]] = {method: [] for method in METHODS}
    best_by_workload: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for key, rows in workloads.items():
        best = min(rows, key=lambda row: int(row["median_total_ns"]))
        t1 = next(row for row in rows if int(row["threads"]) == 1)
        counts[key[3]][int(best["threads"])] += 1
        speedups[key[3]].append(int(t1["median_total_ns"]) / int(best["median_total_ns"]))
        best_by_workload[key[:3]].append(best)

    workload_wins: Counter[str] = Counter()
    for rows in best_by_workload.values():
        workload_wins[min(rows, key=lambda row: int(row["median_total_ns"]))["method"]] += 1

    lines = [["方法", *[f"最优T{threads}" for threads in THREADS], "最佳线程相对T1中位加速", "全局胜出工作负载"]]
    for method in METHODS:
        lines.append([
            METHOD_LABELS[method], *[str(counts[method][threads]) for threads in THREADS],
            f"{median(speedups[method]):.3f}x", f"{workload_wins[method]}/24",
        ])
    return table(lines, "---|---:|---:|---:|---:|---:|---:|---:")


def phase_coverage_table(summary: list[dict[str, str]]) -> str:
    rows = []
    for row in summary:
        if row["method"] != "tblis_2.0_armsve_intra":
            continue
        phase_sum = sum(int(row[name]) for name in (
            "transform_ns", "packing_ns", "kernel_ns", "restore_ns"
        ))
        coverage = phase_sum / int(row["median_total_ns"])
        rows.append((coverage, row["case_id"]))
    lines = [["用例", "阶段中位数之和/总耗时中位数", "未归因占比"]]
    for coverage, case_id in sorted(rows)[:10]:
        lines.append([case_id, f"{100*coverage:.2f}%", f"{100*(1-coverage):.2f}%"])
    return table(lines, "---|---:|---:")


def render_analysis(
    result_dir: Path,
    summary: list[dict[str, str]],
    raw: list[dict[str, str]],
    diagnostics: list[dict[str, str]],
    previous: list[dict[str, str]] | None,
    previous_name: str | None,
) -> str:
    maximum_error = max(float(row["max_rel_error"]) for row in summary)
    return f"""# 大规模多批次多线程张量缩并实验结果分析

## 分析范围

本报告分析 32 vCPU 结果目录 `{result_dir.name}`，并与 4 vCPU 结果目录
`{previous_name or '未提供'}` 的共同 T1/T2/T4 配置逐项核对。固定表达式为：

```text
C[p,a,b,c,d] = sum(k,l) A[p,a,b,k,l] * B[p,k,l,c,d]
```

单个 batch 的逻辑矩阵为 `M=|a||b|`、`N=|c||d|`、`K=|k||l|`；整个调用的
FLOPs 为 `batch*2*M*N*K`。实验交叉 S/M/L/XL、X0/X4、B1/B4/B32 和
T1/T2/T4/T8/T32，共 120 个配置、4 种执行方法、480 个方法组和 72,000 条正式样本。
每组包含 3 个独立 campaign，每轮 warmup 10 次、记录 50 次。

“端到端时间”均来自未启用 profiler 的正式样本。PyTorch profiler、物化审计和
worker 诊断来自独立诊断调用，只用于解释执行路径，不能与正式时间直接相加。
TBLIS-inter 并发调用不使用进程全局 profiling ABI，因此阶段字段保持 N/A。

## 方法与调度

- `PyTorch-OpenBLAS`：官方 aarch64 PyTorch 2.5.0 wheel，一次执行完整 batch
  的 `torch.einsum`；OpenBLAS 内部线程固定为 1。
- `PyTorch-KML`：KPEX/KML patched PyTorch 2.5.0，一次执行完整 batch 的
  `torch.einsum`；请求 T 个 PyTorch intra-op 线程，worker 诊断用于确认实际
  多核活动，但不把具体任务分配层写成既定事实。
- `TBLIS-intra`：batch 串行，每个 contraction 使用 T 个 TBLIS 内部线程。
- `TBLIS-inter`：建立 `min(T,batch)` 个持久 outer worker，每个 contraction
  只用一个 TBLIS 线程；线程池创建不计入 operation，任务分派和同步计入。

X0 的自由轴、缩并轴分别连续；X4 同时交错 A、B 的自由轴与缩并轴。T1/T2/T4/
T8/T32 分别绑定 CPU3、CPU2-3、CPU0-3、CPU0-7、CPU0-31。机器报告 32 个
物理核、1 thread/core、单 socket、单 NUMA。当前 ECS 不暴露 DDR/uncore PMU，
因此报告没有把“算法最少字节/时间”误写成实测内存带宽。

## TBLIS 的 intra 和 inter 设置

| 策略 | 并行位置 | 实现方式 |
|---|---|---|
| `TBLIS-intra` | 单个张量缩并内部 | batch 依次执行，每个 contraction 使用 T 个 TBLIS 内部线程 |
| `TBLIS-inter` | 不同 batch 之间 | 建立 `min(T,batch)` 个持久 worker，每个 worker 执行单线程 TBLIS contraction |

因此 B1/T32 的 inter 实际只有 1 个 worker，B4/T32 只有 4 个，B32/T32 才能使用
32 个 outer worker。报告始终同时记录请求线程数、inner/outer 设置和实际活动 worker，
不能把三者都简称为“32 线程 TBLIS”。

## 结论摘要

1. **4 vCPU 的核心方法排序在共同配置上基本复现。** 72 个共同配置中 69 个的
   最快方法不变；胜出总数从 `OpenBLAS/KML/intra=7/51/14` 变为 `7/50/15`。
2. **“大单次缩并适合 TBLIS-intra”仍成立，但 32 核阈值上移。** T32 下 L/XL
   的 B1 由 intra 获胜，约比 KML 快 1.79-9.82 倍；M/B1 则因单次工作量不足以
   摊薄 32 线程开销而由 KML 获胜。4 vCPU 报告中的“M 及以上”不能原样外推。
3. **“多 batch 通常优先 KML”仍成立，但不再是全覆盖。** T32 的 16 个 B4/B32
   规模/布局配置中 KML 赢 13 个；XL/B4 的两个布局由 intra 获胜，M/X4/B32
   由 inter 以很小优势获胜。
4. **intra/inter 的边界与 4 vCPU 不一致。** B32/T32 下 inter 在 S/M/L/XL
   全部快于 intra；`inter/intra` 从 S 的约 0.002 降到 XL 的约 0.62。32 个独立
   contraction 足以占满 32 个 outer worker，而 intra 的高线程同步成本明显增加。
5. **X4 结论基本一致。** KML 在 B32/T32 的 X4/X0 惩罚约 11.9%-29.8%；
   TBLIS-inter 为 0.964-0.998x，仍显示直接 stride 路径对布局更稳健。TBLIS-intra
   的部分 T32 数据受运行时等待污染，不能仅凭其中位数判断布局优劣。
6. **尾延迟结论比 4 vCPU 更差，而不是更好。** 17/480 个方法组的 P95/中位数
   不小于 2；TBLIS-intra 的 S/B4/T2 最大超过 156x，S/B32/T2 出现可复现的
   约 192 ms P95。阶段合计仍只有毫秒级，额外时间属于同步、调度或未归因等待。
7. **最终判断是“核心结论一致，扩展边界不一致”。** KML 仍擅长批量吞吐，
   TBLIS-intra 仍擅长足够大的单次缩并，TBLIS-inter 仍适合独立 batch；但从 4
   核扩到 32 核后，任务粒度阈值、最佳线程数和长尾风险都发生了实质变化。

## 与 4 vCPU 实验的一致性

下表比较完全相同的 S/M/L/XL、X0/X4、B1/B4/B32、T1/T2/T4 配置。偏差以
4 vCPU 结果为分母，每种方法 72 个匹配配置。

{cross_run_table(summary, previous)}

{winner_consistency(summary, previous)}

四种方法的绝对偏差中位值均不超过 1.58%，说明大多数共同配置的中心位置可复现。
但最大偏差不能忽略：TBLIS-intra 的 `S_X0_B32_T2` 耗时为 4 vCPU 结果的 22.45 倍
（即增加 2144.50%）；
KML 和 OpenBLAS 的少数组也有 20%-27% 偏移。剔除两个 S/B32/T2 intra 异常后，
其余 286 个共同方法组的绝对偏差中位值为 0.55%、P95 为 8.45%、最大值为 27.18%。

### T2 定向复测

| 用例 | 数据集 | 中位数 | P95 |
|---|---|---:|---:|
| S_X0_B32_T2 | 4 vCPU | 4.955 ms | 5.118 ms |
| S_X0_B32_T2 | 32 vCPU 完整实验 | 111.213 ms | 192.148 ms |
| S_X0_B32_T2 | 32 vCPU 定向复测 | 108.006 ms | 192.149 ms |
| S_X4_B32_T2 | 4 vCPU | 4.912 ms | 5.046 ms |
| S_X4_B32_T2 | 32 vCPU 完整实验 | 84.127 ms | 192.142 ms |
| S_X4_B32_T2 | 32 vCPU 定向复测 | 8.235 ms | 136.159 ms |

X0 的高延迟中位数和 P95 在定向复测中重现；X4 的中位数不稳定，但长尾仍重现。
这说明差异不是数学输入、布局物化或单次 kernel 变慢，而是 32 vCPU 环境下
TBLIS-intra T2 小任务路径的非平稳运行时等待。详见
[`T2 定向复测`](../tblis_batch_targeted_t2_recheck_20261011/recheck.MD)。

## 总体胜出分布

{winner_table(summary)}

胜出数按每个固定线程配置的端到端中位数统计。B1 的 25 个 intra 胜出来自中大
规模和适合该任务粒度的线程数；B4/B32 仍主要由 KML 获胜。OpenBLAS 的 7 个
胜出不能解释为其多线程后端优于 KML，因为该 wheel 的 BLAS 内部线程固定为 1。

### 各方法选择最佳线程数

{optimal_thread_table(summary)}

若允许每个方法为每个工作负载单独选择线程数，24 个规模/布局/batch 工作负载中
KML 胜 13 个、TBLIS-intra 胜 10 个、TBLIS-inter 胜 1 个。intra 的最佳线程数
分散在 T4/T8/T32，说明“有 32 核就固定用 32 线程”不是有效策略。

## Batch 与线程扩展

### T1 到 T32 的端到端加速

下表是每个 batch 下 8 个规模/布局配置的 T1/T32 加速比中位值。

{speedup_table(summary, MAX_THREADS)}

B1 没有可供 TBLIS-inter 分发的独立任务，所以它仍与单线程等价；KML 的 B1
也基本不随请求线程数变化。B4 只能提供 4 个 inter worker，而 B32 才能给 KML
和 inter 足够的任务并行机会。TBLIS-intra 的中位加速掩盖了两极分化：L/XL
受益，S/M 的 T32 同步开销可能显著退化。

### T32 下随 batch 增长的吞吐

“相对 B1 吞吐”定义为 `batch*B1总耗时/当前总耗时`。对 batch 间并行路径，
B4 和 B32 的理想上限分别是 4 和 32。“B32/B1 总耗时”保留
用户实际感知到的整个调用延迟。

{throughput_table(summary, MAX_THREADS)}

KML 和 TBLIS-inter 的 B32 相对 B1 吞吐中位值分别为约 20.4x 和 28.2x；整个
B32 调用延迟却仅为 B1 的约 1.58x 和 1.13x。TBLIS-intra 的相对吞吐低于 1，
表明串行 batch 与高线程同步开销叠加后比逐项线性执行还差。OpenBLAS 仍只有
轻微吞吐改善，符合其内部单线程约束。

### T32 端到端性能

每个单元格依次为 B1/B4/B32 的 GFLOP/s，数值取 X0、X4 的算术平均。

{thread_gflops_table(summary, MAX_THREADS)}

KML 与 inter 在 B32 上形成显著的 32 核吞吐扩展，证明“补丁逐 batch 调用 KML
BLAS”不能被解释为整个 batch 串行。intra 在 XL/B1、B4 上仍可给出最高吞吐，
但在小规模 T32 中大量时间落在线程运行时等待，而不是 kernel。

## 两种 TBLIS 调度的边界

下表固定 B32/T32。`inter/intra < 1` 表示 batch 间调度更快；`inter/KML > 1`
表示 TBLIS-inter 仍慢于 KML。

{schedule_table(summary, MAX_THREADS)}

与 4 vCPU 的关键差别是：B32/T32 的 inter 在全部规模上都快于 intra。XL 仍只
快约 1.6 倍，而 S/M 因 intra 高线程等待可快两个数量级。inter 仍不等于全局最快：
除 M/X4 外，KML 在这些 B32/T32 配置上更快。

## 布局、物化与 batch 工作集

下表固定 B32/T32；方法列为 X4/X0 端到端中位耗时比。1.0 表示布局基本无影响。

{layout_table(summary, diagnostics, MAX_THREADS)}

KML 的输入物化字节随 batch 和规模线性增长，且 X4 相对惩罚稳定可见；TBLIS
包装层直接传入入口 shape/stride/data pointer，没有对应整张量 clone。可是避免
物化仍不足以保证最快：KML 可用更高的 batch 并行吞吐覆盖 clone 成本。

## 端到端 CDF 与尾延迟

以下单张总览图覆盖 120 个配置、4 种方法和 72,000 个非 profiler 样本。每行固定
规模/布局，五列为 T1/T2/T4/T8/T32；颜色区分方法，实线/长虚线/短虚线分别表示
B1/B4/B32。由于 B1 与 B32 的整个调用耗时可相差数十倍，横轴使用对数刻度，
但仍是实际端到端微秒值，而不是归一化数据。

![全部配置端到端 CDF 总览](cdf_batch_all_overview.svg)

逐用例线性轴图见 [`cdf/index.MD`](cdf/index.MD)，全部精确坐标见
[`cdf_points.tsv`](cdf_points.tsv)。

### 分布稳定性

{distribution_table(summary, raw)}

表中的 P95/中位数先在每个配置内计算，再跨同 batch 的 40 个配置取中位值和
最大值；`>2x` 则直接统计原始样本。B32 的曲线通常更陡，是因为一次 operation
包含更多计算，固定抖动的相对占比下降。这不能解释成单个 contraction 的响应
时间变稳定；本轮 S/B32/T2 还直接反驳了“batch 一定能摊薄长尾”的强表述。

### TBLIS-intra 的 T2 长尾

{tblis_tail_table(summary, 2)}

### TBLIS-intra 的 T32 长尾

{tblis_tail_table(summary, MAX_THREADS)}

T2 的 S/B4 P95/中位数超过 156x，而 B32 因每次调用包含 32 个潜在等待点，中位数
本身也可能被抬高。T32 在 S/M/L 中同样出现明显长尾和轮次漂移。

### 阶段计数与未归因时间

{phase_coverage_table(summary)}

TBLIS-intra 的阶段计数只覆盖显式 Transform/Packing/Kernel/Restore，不包含线程
运行时等待。上表中 97%-99% 的未归因占比说明 T32 小任务退化不是 kernel 变慢。
没有 futex、barrier 或调度 trace，因而只能说它与线程唤醒、同步或虚拟化调度
开销相容，不能把任何一个写成已证实根因。

## Campaign 复现性

波动定义为三个 campaign 中位数的极差除以三个中位数的总体中位值。它衡量跨轮
中心位置，不等同于单轮内部 P95 长尾。

{campaign_table(summary)}

大多数组的 campaign 中位数稳定，但 TBLIS-intra 的 T32 以及部分 KML/OpenBLAS
配置出现明显轮次偏移。最大值不能只用“虚拟化噪声”一句带过；若需要发布绝对
性能，应在隔离宿主或裸机增加轮数，并同步记录频率、抢占和 run-queue 信息。

## 正确性与证据完整性

- 120 个唯一配置、480 个方法组全部完成；每组均为 3x50 个正式样本。
- 全部方法的入口逻辑 SHA256 和 stride 一致，B1/B4 使用 B32 参考的严格前缀。
- TBLIS wrapper 输入指针与实际传给 TBLIS 的指针一致，未在计时前替换输入。
- intra/inter 的 inner/outer worker 数和亲和性与协议一致。
- OpenBLAS wheel 与 SVE/locking KML runtime 的动态链接检查通过。
- 最大相对误差为 `{maximum_error:.3e}`，低于阈值 `1e-4`。
- 总览图来自 `raw.tsv`，每条曲线严格包含 150 个样本，没有用汇总分位数反推分布。

## 口径限制

1. 当前只测试方形 `M=N` 的 4D 逻辑缩并，不能直接推广到 small-M/N/K、
   tall-and-skinny、5D/6D 或非对称输出。
2. B32 的“单缩并平均时间”是吞吐指标，不是 batch 内每项独立延迟；本实验也
   没有测动态 batching 的排队等待时间。
3. PyTorch 正式计时包含输出分配，而 TBLIS runner 复用预分配输出；这是当前
   端到端实现的真实差异，但不是纯 kernel 同口径比较。
4. TBLIS-intra 正式路径每个 batch 都调用 profiling reset/get，存在小额探针
   开销；TBLIS-inter 为避免并发污染不启用该探针。
5. PyTorch profiler 只能可靠说明事件、次数和物化字节；KML 内部 packing 与
   微内核时间不可见，不能和 TBLIS 的本地阶段字段直接横向相加。
6. 当前 TBLIS 是 `v2.0-beta2-14-geb719e7-dirty` 加本地 profiling 补丁的精确
   构建，结论不应泛化为任意上游 TBLIS 2.0 二进制。
7. ECS 没有硬件 DDR/uncore PMU，现有最低有效带宽不是实测内存控制器带宽。
8. 4 vCPU 与 32 vCPU 是不同资源配额下的实验；共同配置使用相同亲和性，但宿主
   调度、cgroup 配额和运行时看到的在线 CPU 数仍可能改变线程库行为。

## 最终判断与建议

1. **L/XL 的 B1 优先评估 TBLIS-intra，但必须调线程。** 32 核下不能沿用“M 及
   以上固定满线程”；应在 T4/T8/T32 中按单次规模选择，并同时约束 P95。
2. **B32 优先 KML patched PyTorch。** 它在 T32 的 8 个 B32 配置中赢 7 个；
   唯一例外是 M/X4，inter 仅小幅领先。XL/B32 中 KML 比 inter 快约 1.6-1.8 倍。
3. **TBLIS 多 batch 应优先 inter，而不是把每个小 contraction 开到 32 线程。**
   B32/T32 的 inter 在全部规模上优于 intra；B1/B4 的 worker 上限又分别只有 1/4，
   因此必须根据 batch 和单次 FLOPs 双维度调度。
4. **不要把 T32 阶段 Kernel GFLOP/s 当成端到端性能。** 在最差组中阶段只覆盖
   1%-3% 总时间；下一步应在 barrier、线程池、condition variable/futex 和任务
   分发周围增加 wall-clock 探针，并采集调度事件。
5. **若目标是在线服务，仍需动态到达实验。** 记录排队、batch 形成、每项完成
   时间和请求 P95/P99，再权衡批量吞吐与等待成本。

综上，32 vCPU 实验没有推翻 4 vCPU 的方法定位：TBLIS-intra 擅长足够大的单次
缩并，KML 擅长多独立任务的框架吞吐，TBLIS-inter 擅长显式 batch 间并行。
但“多大才足够大”“intra 与 inter 何时切换”以及尾延迟风险都随核心数改变。
因此结论是部分一致：方向一致，定量阈值和高线程调度结论不一致。

## 数据索引

- [`suites.MD`](suites.MD)：实验方法、完整性能表和原始汇总报告。
- [`summary.tsv`](summary.tsv)：480 个方法组的中位数、P95、吞吐和阶段统计。
- [`raw.tsv`](raw.tsv)：72,000 条非 profiler 正式样本。
- [`cdf_batch_all_overview.svg`](cdf_batch_all_overview.svg)：全部配置的单张 CDF 总览。
- [`cdf_points.tsv`](cdf_points.tsv)：全部经验 CDF 精确坐标。
- [`materialization_diagnostics.tsv`](materialization_diagnostics.tsv)：物化、分配和
  TBLIS 阶段诊断。
- [`thread_diagnostics.tsv`](thread_diagnostics.tsv)：worker、线程策略和亲和性。
- [`layout_audit.tsv`](layout_audit.tsv)：入口 shape、stride、storage 和逻辑哈希。
- [`timeline_batch.svg`](timeline_batch.svg)：代表性诊断 timeline；不同方法口径
  不完全相同，不能替代正式端到端计时。
- [`validation.log`](validation.log)：覆盖率、正确性、动态库和产物完整性检查。
- [`4 vCPU analysis.MD`](../tblis_batch_cpu0-3_20261010T075228168714Z/analysis.MD)：
  本报告的结构与结论对照基准。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--comparison-summary", type=Path)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    if (result_dir / "status.txt").read_text(encoding="utf-8").strip() != "COMPLETE":
        raise SystemExit("result directory is not COMPLETE")
    summary = read_tsv(result_dir / "summary.tsv")
    raw = read_tsv(result_dir / "raw.tsv")
    diagnostics = read_tsv(result_dir / "materialization_diagnostics.tsv")
    expected_groups = len(SCALES) * len(LAYOUTS) * len(BATCHES) * len(THREADS) * len(METHODS)
    expected_samples = expected_groups * 3 * 50
    if len(summary) != expected_groups or len(raw) != expected_samples:
        raise AssertionError(
            f"unexpected input size: summary={len(summary)}, raw={len(raw)}"
        )
    previous = read_tsv(args.comparison_summary) if args.comparison_summary else None
    previous_name = args.comparison_summary.parent.name if args.comparison_summary else None

    cdf_path = write_combined_cdf(result_dir, raw)
    svg = ET.parse(cdf_path).getroot()
    paths = svg.findall("{http://www.w3.org/2000/svg}path")
    expected_curves = len(SCALES) * len(LAYOUTS) * len(THREADS) * len(BATCHES) * len(METHODS)
    if len(paths) != expected_curves:
        raise AssertionError(f"unexpected CDF curve count: {len(paths)}")
    report = render_analysis(
        result_dir, summary, raw, diagnostics, previous, previous_name
    )
    (result_dir / "analysis.md").write_text(report, encoding="utf-8")
    print(f"wrote {cdf_path}")
    print(f"wrote {result_dir / 'analysis.md'}")


if __name__ == "__main__":
    main()
