# KML multi：32 vCPU 完整实验配置

日期：2026-10-14。当前在线、present、possible CPU 均为 0–31，进程允许全部
32 个 vCPU，cgroup CPU quota 为 -1（无限制），内存约 59 GiB。

执行状态：完整 120 配置、18,000 个正式样本已完成，校验通过。
[统一报告](/root/benchmark/tensor_contraction_redesign/results/pytorch_kml_multi_cpu0-31_20261014T101523721193Z/suites.MD)
和 [结果分析](/root/benchmark/tensor_contraction_redesign/results/pytorch_kml_multi_cpu0-31_20261014T101523721193Z/analysis.md)
包含新旧来源区分及 P95 回退，完成实验不代表所有性能验收目标都通过。

## 1. OpenBLAS 影响判断

结论：本次只重跑 PyTorch-KML。multi 由新增的 KML 专用子进程入口以绝对路径
加载；不替换安装库、软链接、公共编译配置或 OpenBLAS wheel。

运行时检查确认：

- 官方 PyTorch 2.5.0 的 BLAS_INFO=open；
- 配置 SHA256 为 `b08896f97236d9094b7db02297c10a1d7cfa88483096dcebb6a75120b47ec999`，
  与历史一致；
- 映射 wheel 自带的 OpenBLAS 和 libgomp，未加载 libkblas；
- 公共 `benchmark_tblis_batch_pytorch.py` 与历史源码 SHA256 一致；
- 用例清单的 120 行字段、顺序与历史一致，现有文件仅序列化哈希不同；
- 正式实验使用历史清单副本及校验过 SHA256 的历史 FP64 参考。

运行前后记录 OpenBLAS 实际映射库的 SHA256；若发生变化则验证失败。
不受本次改动影响不等于跨日期性能完全不变，因此统一报告明确标记旧数据来源。

## 2. 保持一致的实验要求

| 项目 | 配置 |
|---|---|
| 表达式 | `pabkl,pklcd->pabcd`，FP32 |
| 规模 M/N/K | S=256/256/64，M=512/512/128，L=1024/1024/256，XL=2048/2048/512 |
| 布局 | X0：A=abkl，B=klcd；X4：A=akbl，B=kcld |
| Batch | 1、4、32 |
| 请求线程 | 1、2、4、8、32 |
| affinity | T1=CPU3；T2=CPU2–3；T4=CPU0–3；T8=CPU0–7；T32=CPU0–31 |
| 每配置 | 3 个 campaign，每轮预热 10 次、正式记录 50 次 |
| 完整覆盖 | 120 个配置，18,000 条正式 KML 样本 |
| 正式计时 | 沿用旧 runner；包含完整 einsum、输出分配和既有输出生命周期 |
| 正式数值输入 | 沿用历史生成器和逻辑值；输入指纹与历史对齐 |
| 正确性 | 每轮计时结束后逐元素校验末次输出；原 FP64 参考、1e-4 归一化误差门限，额外检查有限数 |
| 诊断 | profiler/物化、线程活动、独立 BLAS 边界＋OMPT、代表性 Chrome traces |
| 输出 | 原始时间、汇总表、来源明确的统一表、CDF、诊断 timeline、分析报告 |

## 3. 新 KML 方法的线程策略

| 情况 | ATen 配置 | KML 内部线程 | 调度目标 |
|---|---|---|---|
| B1 | intra-op=T | T | 单次 GEMM 内部并行 |
| B4/B32 | intra-op=T | 1 | 保留当前补丁的 batch 外层并行 |

这是明确的互斥策略，不是根据结果选择的最佳策略，也不实现 hybrid。
B4 的 XL 是否值得采用内部并行，留待后续消融，不能把本次结果称为 KML
所有调度方式中的最优性能。

KML 专用环境设置 `OMP_TOOL=disabled` 避免已复现的默认工具探测崩溃；
`OMP_DYNAMIC=FALSE`、`OMP_MAX_ACTIVE_LEVELS=1` 控制线程和嵌套。
其余环境沿用旧运行器，不添加之前兼容性探针使用的被动等待策略。

## 4. 实际线程和正确性的额外验证

正式调用不加载 observer。独立诊断进程使用 BLAS 拦截和 OMPT，分别检查：

- `sgemm_` 调用次数等于 batch；
- 调用 GEMM 的线程数等于 `min(batch,T)`；
- B1 的 KML 内部 team 与请求 T 一致，T1 无需创建并行 team；
- B4/B32 每个调用线程中 KML local threads 均为 1；
- 不发生 KML 内层 team 额外扩张；
- 区分 KML 和 PyTorch 并行区，避免把 X4 复制线程误判为 GEMM 内部线程。

性能输入保留历史周期序列；另外按 batch/layout/thread 组合运行 batch 值不同的
随机非整齐小矩阵校验，使用独立 NumPy FP64 参考，不计入性能数据。

## 5. 报告口径

本次方法命名 `pytorch_2.5.0_kml_multi`，旧 `pytorch_2.5.0_kml_patched`
明确标为 locking 历史数据。OpenBLAS、TBLIS-intra、TBLIS-inter 沿用历史结果。
每行带 `source_run` 和 `measurement_epoch`，不能将旧样本称为本轮重跑。

PyTorch 无可靠 KML 内部阶段计时，主表阶段数据保留 N/A；operator/物化诊断
另外展示。CDF 横轴为整个 batch 端到端时间，纵轴为经验 CDF。
timeline 为诊断事件展示，不代表闭源 KML 纯 kernel 的拆分。

## 6. 执行入口

```bash
python3 run_kml_multi_batch.py --preflight-only
python3 run_kml_multi_batch.py
python3 summarize_kml_multi_batch.py results/<kml-multi-run-id>
```

代码、库、参考哈希和 OpenBLAS 运行前后审计保存在新目录。
运行中断时只允许同源码和库恢复；完整性或线程验证失败会中止。
