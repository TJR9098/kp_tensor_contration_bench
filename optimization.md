# PyTorch/KML 张量缩并优化方案：源码与实验依据

审查日期：2026-10-14。本文是后续实现与验证方案；所列优化尚未实施，
文中的性能数值来自已有实验数据的复算。

## 1. 判断与适用范围

最值得优先实施的是：验证 KML SVE multi 库，为少 batch 的大 GEMM 增加内部
多线程，再由计算计划选择串行、batch 间并行或单 GEMM 内部并行。布局优化应
在 `einsum` 矩阵化之前决策，先验证受限布局的分块收集方案。

这个判断针对**当前部署的 PyTorch 补丁与 SVE locking 库组合**。原始 KPEX
补丁查找 multi 库，当前部署为兼容本机而改用了 locking；不能用当前实验否定
原始补丁或 KML multi 的内部并行能力。

采用以下三种证据口径：

| 口径 | 含义 | 使用限制 |
|---|---|---|
| 已确认事实 | 源码、动态库或已有正式样本直接支持 | 限定具体版本、构建和工作负载 |
| 实测差距参考 | 当前 KML 与 TBLIS 等实现之间的耗时差距 | 不是 KML 改造后的性能承诺，也不是严格收益上限 |
| 条件估算/待验证假设 | 在明确假设下计算收益，或提出机制假设 | 必须由消融实验验证，不能算作已实现收益 |

在已测 24 个规模/布局/batch 工作负载中，允许各方法自行选择最佳线程数后，
KML 胜 13 个，TBLIS-intra 胜 10 个，TBLIS-inter 胜 1 个。其中 B32 的 8 个
工作负载中 KML 胜 7 个。优化应保护这一优势，保留已有执行路径作为对照。

## 2. 实验对象、数据来源和统计口径

### 2.1 工作负载

固定表达式为 `pabkl,pklcd->pabcd`。每个 batch 项是
`abkl,klcd->abcd` 的四维输入缩并；带 batch 轴 p 后，实际输入为五维张量。
使用 FP32，B1/B4/B32 表示独立 batch 数，T1/T2/T4/T8/T32 表示请求线程数。

| 规模 | M=ab | N=cd | K=kl |
|---|---:|---:|---:|
| S | 256 | 256 | 64 |
| M | 512 | 512 | 128 |
| L | 1024 | 1024 | 256 |
| XL | 2048 | 2048 | 512 |

X0 的单项物理轴序为 A=`abkl`、B=`klcd`；
X4 为 A=`akbl`、B=`kcld`。不同布局保持逻辑值和计算一致。

### 2.2 可追溯证据

| 证据 | 内容 |
|---|---|
| [原始样本](/root/benchmark/tensor_contraction_redesign/results/tblis_batch_cpu0-31_20261011T074035858907Z/raw.tsv) | 120 个配置 × 4 方法 × 3 campaign × 50 次，共 72,000 条正式样本 |
| [汇总表](/root/benchmark/tensor_contraction_redesign/results/tblis_batch_cpu0-31_20261011T074035858907Z/summary.tsv) | 中位数、P95、阶段指标和线程诊断 |
| [物化诊断](/root/benchmark/tensor_contraction_redesign/results/tblis_batch_cpu0-31_20261011T074035858907Z/materialization_diagnostics.tsv) | A/B clone 次数、字节数及发生位置 |
| [现有分析](/root/benchmark/tensor_contraction_redesign/results/tblis_batch_cpu0-31_20261011T074035858907Z/analysis.md) | 多线程扩展、布局、CDF 和异常复测 |
| [补丁分析](/root/benchmark/tensor_contraction_redesign/patch_analysis.md) | KPEX 补丁与部署构建的源码审计 |
| [环境记录](/root/benchmark/tensor_contraction_redesign/results/tblis_batch_cpu0-31_20261011T074035858907Z/provenance.json) | warmup、campaign、线程亲和性及运行环境 |

本轮复算确认 480 个方法配置的中位数和 P95 均与汇总表一致。每个配置合并
3×50 个正式样本；P95 使用 nearest-rank，最佳线程指在已测线程数中端到端
中位数最低的点，不代表连续搜索得到的全局最优。

## 3. 源码确认的现状

### 3.1 原始补丁与当前部署必须区分

| 项目 | 原始补丁/当前实现 |
|---|---|
| 原始 KPEX 的库搜索路径 | `matrix/kblas/multi` |
| 当前部署的库搜索路径及正式加载库 | `sve/kblas/locking` |
| 当前 BMM 并行对象 | 不同 batch，每项调用一次 GEMM |
| 当前 locking 线程接口 | `BlasSetNumThreadsLocal` 直接返回；`BlasGetNumThreads` 固定返回 1 |
| 已安装 SVE multi | 导出线程控制、`cblas_sgemm_batch` 和 pack/compute；依赖 `libomp.so` |
| 当前 SVE locking | 导出 pack/compute，不导出 `cblas_sgemm_batch` |

原始路径见 [KPEX patch](/root/benchmark/pytorch-kunpeng/kpex-current/thirdparty/pytorch-v2.5.0.patch:614)，
当前路径见 [FindKML.cmake](/root/benchmark/pytorch-kunpeng/pytorch-v2.5.0/cmake/Modules/FindKML.cmake:4)。
上述符号和 locking 接口行为已用动态符号表及反汇编核实；它们不等价于 multi
版本已经通过性能和正确性验证。

### 3.2 被测 FP32 执行路径

```text
torch.einsum
  → permute / reshape
  → bmm_impl_cpu_
  → ATen parallel_for 按 batch 分工
  → 每个 batch 调用 cpublas::gemm → sgemm_ → KML
  → view / permute 恢复输出
```

[einsum 矩阵化](/root/benchmark/pytorch-kunpeng/pytorch-v2.5.0/aten/src/ATen/native/Linear.cpp:229)
位于 BMM 之前；[BMM 循环](/root/benchmark/pytorch-kunpeng/pytorch-v2.5.0/aten/src/ATen/native/LinearAlgebra.cpp:1694)
没有进一步切分单个 GEMM 的 M/N/K。
[FP32 BLAS 边界](/root/benchmark/pytorch-kunpeng/pytorch-v2.5.0/aten/src/ATen/native/CPUBlas.cpp:272)
调用标准 `sgemm_`。

这是当前已核验用例的路径。极小矩阵、其他 dtype、其他构建选项或其他算子可能
走不同分支，不能无条件推广到所有 PyTorch GEMM。

### 3.3 静态连续分块已经存在

[ParallelOpenMP.h](/root/benchmark/pytorch-kunpeng/pytorch-v2.5.0/aten/src/ATen/ParallelOpenMP.h:25)
先建立 OpenMP team，再按如下规则分配连续区间：

```cpp
chunk_size = divup(end - begin, num_threads);
begin_tid = begin + tid * chunk_size;
f(begin_tid, min(end, begin_tid + chunk_size));
```

因此，“新增静态连续 batch 分块”不应列为独立优化。当前循环也不是每个 batch
单独创建一个并行区或 barrier。

grain 可以改变逻辑任务划分，但通常不能减少已经进入并行区的 team 成员。
如果 grain 使 [parallel_for 的串行条件](/root/benchmark/pytorch-kunpeng/pytorch-v2.5.0/aten/src/ATen/Parallel-inl.h:20)
成立，则会绕过并行区；这与精确控制一个较小 team 是不同机制。

## 4. 实验显示的优化机会

### 4.1 单 GEMM 与 batch 扩展

下表为每个 batch 下 8 个规模/布局配置的 T1/T32 加速比中位值：

| 方法 | B1 | B4 | B32 |
|---|---:|---:|---:|
| 当前 PyTorch-KML | 1.004x | 3.837x | 19.671x |
| TBLIS-inter | 0.995x | 3.929x | 28.606x |

当前 KML 的 B1 基本不扩展，与单线程 locking 库及 batch 级分工一致。

下表均为 B1，各方法自行选择最佳线程。耗时单位 ms，加速比为 KML/TBLIS：

| 规模/布局 | KML 中位数 | TBLIS-intra 中位数 | TBLIS 线程 | 比值 |
|---|---:|---:|---:|---:|
| S/X0 | 0.136956 | 0.130854 | 4 | 1.047x |
| S/X4 | 0.159027 | 0.130900 | 4 | 1.215x |
| M/X0 | 0.887652 | 0.462319 | 4 | 1.920x |
| M/X4 | 0.936155 | 0.459774 | 4 | 2.036x |
| L/X0 | 6.411864 | 1.728615 | 8 | 3.709x |
| L/X4 | 6.504189 | 1.701314 | 8 | 3.823x |
| XL/X0 | 49.127433 | 5.030086 | 32 | 9.767x |
| XL/X4 | 49.235260 | 5.015548 | 32 | 9.817x |

这支持优先探索大单次缩并内部并行。9.817x 是跨实现差距，不是 KML multi
预期加速或理论上限。

### 4.2 多 batch 的反例与边界

| Batch | 允许各方法选择最佳线程后的胜出分布 |
|---|---|
| B1 | TBLIS-intra 8/8 |
| B4 | KML 6/8，TBLIS-intra 2/8 |
| B32 | KML 7/8，TBLIS-inter 1/8 |

B4 的两个 TBLIS 胜出都来自 XL：

| 用例 | KML 中位数/ms | TBLIS-intra 中位数/ms | KML/TBLIS |
|---|---:|---:|---:|
| XL/X0/B4 | 50.031487 | 20.133290 | 2.485x |
| XL/X4/B4 | 50.787998 | 19.829722 | 2.561x |

L/B4 仍由 KML 领先。B32/T32 唯一的 TBLIS 胜出是 M/X4：
TBLIS-inter 为 1.679190 ms，KML 为 1.734376 ms，延迟约低 3.2%。
因此，batch 数与单项矩阵规模必须共同参与选择。

### 4.3 X4 的物化与布局差值

以下固定 B32/T32。X4 的 A/B 各 clone 一次，合计字节数如下；诊断定位为
进入 BMM 前的 `permute/reshape`，而不是 BMM 内部的 clone fallback。

| 规模 | KML X0/ms | KML X4/ms | X4−X0/ms | X4/X0 | X4 输入物化 | TBLIS-inter X4/X0 |
|---|---:|---:|---:|---:|---:|---:|
| S | 0.204826 | 0.266340 | 0.061514 | 1.300x | 4 MiB | 0.997x |
| M | 1.537372 | 1.734376 | 0.197004 | 1.128x | 16 MiB | 0.998x |
| L | 8.554606 | 11.030082 | 2.475476 | 1.289x | 64 MiB | 0.964x |
| XL | 56.607163 | 63.327258 | 6.720095 | 1.119x | 256 MiB | 0.986x |

如果新 X4 路径恰好达到当前 X0 延迟，条件估算为约 1.12x–1.30x 加速，
或 X4 延迟降低 10.6%–23.1%。这不是严格收益上限，也不能将全部差值归因于
clone：布局还改变了访问、缓存和后续执行条件。

256 MiB 是当前完整物化的输入字节数，不是可以承诺的 RSS 净减少量。
新实现仍可能需要收集缓冲、KML packed buffers 和并发 worker 工作空间。

### 4.4 尾延迟

S/X0/B32/T32 的数据为：

| 方法 | 中位数/ms | P95/ms | P95/中位数 |
|---|---:|---:|---:|
| KML | 0.204826 | 0.479056 | 2.339 |
| TBLIS-inter | 0.331705 | 0.348876 | 1.052 |

KML 的中位数更低，TBLIS-inter 的绝对 P95 更低。后续优化应同时比较两项；
不能通过使中位数变慢来“改善”P95/中位数比例。

T32 下 B32 相对 B1 的吞吐中位值为 KML 20.362x、TBLIS-inter 28.207x，
其中相对吞吐定义为 `B × latency(B1) / latency(B)`。
该差异包含内核、packing、内存访问和运行时因素，不等于可由调度兑现的 1.39x 收益。

## 5. 优化优先级与可行性

这里的可行性表示在当前接口和代码条件下的落地把握；收益潜力另行评价。

| 优先级 | 方向 | 可行性 | 实现办法 | 收益判断与边界 |
|---|---|---|---|---|
| P0 | 可观测性与正确性验收 | 高，细粒度线程归因需验证 | 物化计数、执行计划、BLAS 边界、不同 batch 输入 | 无直接加速，是可靠决策的前提 |
| P0 | SVE multi 原型 | 中 | 同构建比较 locking/multi，显式控制 KML 线程 | 大 B1 的收益潜力最高，倍数尚未测得 |
| P0 | serial/outer/inner 选择 | 中 | 使用 KML 自身实测曲线标定静态分桶 | 改善少 batch 大计算，保护 B32；与 multi 收益不可重复相加 |
| P1 | 布局感知与分块收集 | 中低 | 在 einsum 矩阵化之前选择 view、整输入物化或 tile 收集 | 有降低整输入临时空间的机会；端到端收益需独立验证 |
| P1 | batched API 与尾延迟 | 中 | 对比 KML batch API、现有 outer、等待与亲和性策略 | 重点验证绝对 P95；现有静态分块不是新增优化 |
| P2 | 混合并行/统一任务网格 | 低到中 | 受控 outer×inner 或 batch×输出 tile | 少 batch 超大任务可能受益；无新增收益实测 |
| P2 | 可复用权重预打包 | 中低 | 有所有权、可失效的 packed object | 需要真实复用场景，当前数据不能量化 KML 收益 |

## 6. P0：观测和验收的前置改进

### 6.1 阶段命名与测量

建议添加默认关闭或可采样的诊断事件：

```text
einsum_input_materialize：对象、次数、字节、shape/stride/storage offset
bmm_plan：serial/outer/inner、请求与实际线程配置、决策原因
bmm_outer_region：整个外层执行区间
blas_call：KML 调用边界
bmm_output_copy：输出复制
einsum_restore_view_or_copy：输出轴恢复
```

KML `sgemm_` 的边界时间应称为 `blas_call_ns` 或 `backend_ns`。没有厂商
内部探针时，KML 的纯 packing/kernel 时间继续记为 N/A。不能直接将整个
outer 区间减去 worker 时间之和得到等待时间，并行区间需要按时间轴和关键路径分析。

ATen 的 worker profiler TLS 传播有限，调用线程上记录 outer 边界比较可靠；
细粒度 worker 数据应使用验证过的探针，并检查采样本身的扰动。

现有“阶段之和只覆盖总时间 1%–3%”的异常来自 TBLIS-intra，说明该路径有
大量未归因时间，不能据此认定 KML 也存在同样比例的等待或同一根因。

### 6.2 活跃线程不是 GEMM 计算线程

[worker_diagnostic](/root/benchmark/tensor_contraction_redesign/benchmark_tblis_pytorch.py:159)
使用 `/proc/self/task` CPU ticks 的增量统计活跃线程。自旋或框架转换也会被计入，
“CPU 时间/墙钟时间”同样包含这些 CPU 消耗。

现有 XL/X0/B1/T32 KML 的正式样本中位核当量约为 0.999，但独立诊断报告
32 个 active workers、约 21.49 核。结合 locking 单线程和 B1 源码，
不能将诊断中的 32 解读为 32 个 GEMM 计算线程；差异根因尚未完成归因。

后续分别记录：请求线程数、OpenMP team 大小、调用 BLAS 的 outer worker、
实际执行 GEMM 的内部线程、进程核当量。无法可靠观测的字段保留 N/A。
诊断应与正式运行使用一致的 inference mode、预热和输出生命周期，计算区间内
记录线程活动，避免验证阶段及其他活动混入解释。

### 6.3 增加能检测 batch 索引错误的输入

[输入生成器](/root/benchmark/tensor_contraction_redesign/benchmark_tblis_batch_pytorch.py:48)
使用 `(index * 1103515245 + salt) & 0xFFFF`，序列周期为 2^16。
M/L/XL 的每个输入单 batch 元素数均为该周期的整数倍，因此不同 batch 数值相同；
S 则每 4 个 batch 重复。这不能充分检测 batch 串位或索引复用错误。

新增正确性数据集应使用可复现、batch 间不同的输入，同时保持不同方法、
布局和线程之间的同用例逻辑值一致。旧数据作为历史性能对照保留；
改变输入后使用独立的数据集标识，并重新生成 FP64 参考。

### 6.4 明确误差和数值检查

[现有校验](/root/benchmark/tensor_contraction_redesign/benchmark_tblis_batch_pytorch.py:93)
的公式为：

```text
max_i abs(result_i - reference_i) / max(abs(reference_i), 1)
```

这是带分母下限的归一化误差，参考值绝对值小于 1 时使用绝对误差尺度，
不应解释为严格逐元素相对误差。新实验继续记录该指标与 1e-4 门限，
另行显式检查所有输出和误差值为有限数，并记录最大绝对误差。

用于新并行实现的正确性集还应覆盖非整齐尺寸、边界 tile、X1 简单转置、
X2/X3/X4、X5 间隔、storage offset，以及受支持的广播场景。不支持的
dtype、布局或梯度场景走明确 fallback。

## 7. P0：SVE multi 原型与互斥调度

### 7.1 先验证库，再实现 planner

1. 保持编译器、PyTorch 源码、ABI 和 oneDNN 选项一致，建立独立 multi 变体。
   使用本机已有兼容证据的 SVE 库族；具体 multi 二进制仍需正确性和运行验证。
2. 同时配置 CMake 和运行时加载路径。[运行器](/root/benchmark/tensor_contraction_redesign/run_tblis_batch_experiment.py:59)
   当前固定加入 locking 路径，必须核实最终加载的真实库、符号及 OpenMP runtime。
3. 检查环境变量与显式线程 API 的作用关系。运行器当前设置
   `OMP_NUM_THREADS=1` 等变量，不能仅换库就假定 inner 已开启。
4. 先测同一个 multi 库的串行、outer 和 inner，再与原 locking 部署比较。
   multi 的单线程 GEMM 不保证等于 locking，B32 回归是单独的必测项。
5. 验证 `BlasSetNumThreadsLocal` 的局部语义、默认值和恢复行为；线程状态在
   调用结束和异常退出后必须恢复。不能在并发调用中用全局 setter 临时切换。
6. outer 模式将每个调用线程中的 KML inner 限为 1；inner 模式不进入 outer 并行区。
   用真实线程观测确认没有嵌套并行或超额订阅。

原型测试分为直接 SGEMM、预矩阵化 BMM、完整 einsum 三层，
分别确认库扩展、调度开销和框架端到端收益。

### 7.2 首版计算计划

| 模式 | batch 执行 | KML GEMM 线程 | 适用候选 |
|---|---|---|---|
| serial | 顺序 | 1 | 总工作量太小，或安全 fallback |
| outer | batch 间并行 | 每项 1 | 独立项较多、单项不适合内部并行 |
| inner | batch 顺序 | 每项 I | batch 少、单项足够大 |

outer 的目标有效 worker 数不超过 `min(batch,T)`，这不等于当前 OpenMP
一定只创建这些 team 成员。若需要限制 team 大小，必须实现并验证相应机制。

计划输入包括 batch、M/N/K、dtype、矩阵布局、线程预算和是否已处于并行区域。
原始高维布局及物化成本属于 einsum 层计划；BMM 已经收到矩阵化后的输入，
不能假设它可以恢复上游布局信息。

先采用离线标定的静态分桶，提供强制模式开关与保守 fallback。
TBLIS 的 T4/T8/T32 只能作为实验选点参考；KML 阈值必须由 KML 自身的
T1/T2/T4/T8/T32 曲线确定，并用未参与标定的形状验证。

### 7.3 可行性与收益边界

multi 库与线程控制接口已存在，所以无需首先重写 GEMM，工程可行性为中。
主要风险在运行时嵌套、线程状态、库加载，以及 multi 单线程性能回退。

大 B1 和 XL/B4 是优先目标；当前 TBLIS 在这些区域的优势说明有研究价值，
并未测出 KML multi 的收益。保持 outer 代码结构也不能保证保持原性能，
因为底层库已变化。

如果 multi 的 outer 性能不能满足回归门槛，应暂停统一部署或保留独立变体。
不能未经验证在同一进程中混装两个同名 BLAS 库、依赖符号覆盖实现切换。

## 8. P1：布局感知与分块收集

### 8.1 修改点和公开接口边界

X4 复制发生在 einsum 的 reshape；仅修改 BMM 的 `lda/transpose` 判断无法消除它。
对可无复制矩阵化的输入继续使用现有 BMM，对其余布局在矩阵化之前选择路径。

[KML pack 接口](/root/benchmark/pytorch-kunpeng/hpckit-components/KunpengHPCKit-kml.26.1.RC1/include/kblas.h:673)
接受 `src + ld + transpose + M/N/K`，没有高维 shape/stride、scatter 数组
或自定义读取回调。它不能直接消费任意 X4 输入。

相对地，[TBLIS packing](/root/benchmark/exo_tensor_bench_openeuler2203_arm64_20261008_v2/sources/tblis-2.0/tblis/frame/1m/packm/packm_blk_bsmtc.cxx:82)
显式构造 block-scatter 数据。因此调用 KML pack/compute 不等于已经获得
TBLIS 的直接 scatter packing 能力。

### 8.2 可实施的第一版

比较三条路径：

| 路径 | 操作 | 适用条件 |
|---|---|---|
| view + BMM | 无复制矩阵化后直接计算 | shape/stride 允许无复制 reshape |
| 完整物化 + BMM | 当前连续化路径 | 物化成本可接受、规则 GEMM 效率高 |
| tile 收集 + KML | 按高维索引收集到规则临时矩阵，再调用 GEMM 或 pack/compute | 已实现并验证的 X2/X3/X4 等受限布局 |

第三条路径首先使用普通 KML GEMM 建立正确原型，再比较显式 pack/compute 是否
有复用价值。公开 pack 接口仍需要规则矩阵输入，可能发生
“收集一次、KML 再打包一次”，还可能因输出切块重复收集同一输入。

直接把高维输入写入 KML 内部 packed 格式，需要已验证的格式契约或厂商接口；
当前公开头文件不足以支持这一假设。

先沿输出 M/N 分块并尽可能保留完整 K，减少额外归约。若后续切分 K，
需正确处理部分和、beta 和浮点累加顺序。首次只覆盖 CPU FP32 的受限场景，
其余情况保留 fallback；训练支持需要另外验证 autograd。

选择性 TBLIS 后端也是候选，但需放进同一框架封装，统一分配和调度语义后比较，
同时承担依赖、ABI 和维护成本。

### 8.3 分别验收内存和性能

先验证是否降低整输入物化和峰值工作空间，再验证端到端是否更快。
指标包括 clone 字节、收集字节、重复打包、每 worker scratch、并发总工作空间、
RSS 和中位数/P95。未暴露的后端 packed buffer 大小不能假定为零。

减少完整 clone 的目标合理；减少总搬运和降低延迟尚待验证。
X4 达到当前 X0 的情景估算见第 4.3 节，不能直接外推到尚未定量测量的 X2/X3/X5。

## 9. P1：batch API、等待策略与尾延迟

现有 ATen 静态分块作为基线。后续只比较有实际机制变化的候选：

- 小 batch/小任务提前串行，或经验证的 team 数量控制；
- OpenMP 等待策略、barrier 与线程唤醒行为；
- 专用 benchmark 中的亲和性消融，框架默认路径不擅自绑定用户线程；
- KML multi 的 `cblas_sgemm_batch` 与现有 outer 循环。

[KML batch 接口](/root/benchmark/pytorch-kunpeng/hpckit-components/KunpengHPCKit-kml.26.1.RC1/include/kblas.h:593)
在已安装 multi 库中有实际导出，locking 没有。接口存在并不保证使用 fused kernel，
也不保证优于现有循环。将指针数组构造、分配和调用成本纳入正式计时，
不要在 vendor batch 调用外再套未经控制的 outer 并行。

同一 KML 库下比较现有 outer 与 vendor batch，才能较好隔离分发变化。
还应对比原 locking 部署，避免库差异掩盖回退。

主要收益假设是改善小 B32 的绝对 P95；中位数收益未知。亲和性、等待策略
目前是诊断线索，不是已证实根因。验收同时约束中位数与绝对 P95，
P95/中位数仅作辅助描述。

## 10. P2：混合并行和权重预打包

### 10.1 混合并行

如果互斥策略仍无法充分利用少 batch 大任务，可探索
`outer_workers × inner_threads <= T`。线程预算只是必要条件，
不能自动保证实际 team、嵌套运行时及并发调用都受控。

另一候选是在一个任务系统中切分 `(batch, output-M/N tile)`，
底层继续使用单线程 KML，减少嵌套运行时风险。
它可能导致重复 packing、缓存竞争和任务过细，仍需与最佳纯 outer/inner 比较。

当前 XL/B4 的 TBLIS 优势支持探索少 batch 大任务，但没有证明 hybrid
相对最佳 KML inner 的额外收益。先完成互斥计划，再决定是否投入。

### 10.2 可复用输入预打包

KML 提供 pack/compute，适合评估跨调用复用同一权重的场景。
应使用有所有权的 packed object，明确 storage 生命周期、shape/stride、
dtype、transpose、storage offset、版本及变异失效规则。无法可靠判断变异时，
使用显式不可变对象或重新打包，不依赖不安全的裸指针缓存。

需要分别报告首次打包成本、后续调用、摊销延迟及额外驻留内存。
当前重复计时没有测量“缓存预打包后执行”的路径，不能给出这类收益。

TBLIS 最佳 B1 的阶段指标复算如下：

| 统计范围 | packing 指标/总时间 |
|---|---:|
| L/XL，两种布局 | 5.65%–8.76% |
| S/M/L/XL，两种布局 | 5.65%–10.52% |

此前 L/XL 的范围混入了 S 的 10.52%。此外，
[TBLIS profiling 实现](/root/benchmark/exo_tensor_bench_openeuler2203_arm64_20261008_v2/sources/tblis-2.0/tblis/frame/base/thread.cxx:242)
对 packing/kernel 分别取各线程累计时间的最大值；两者不保证属于同一线程，
也不是严格可相加的串行阶段。不能据此直接套用 Amdahl 公式推算预打包收益，
更不能将这些比例视为 KML 的 packing 占比。

## 11. 实施阶段与进入条件

| 阶段 | 工作 | 进入下一阶段的证据 |
|---|---|---|
| 0：基线与观测 | 固定历史结果，补 batch 不同输入、有限数检查、线程和物化观测 | 正确性覆盖有效；诊断字段含义和开销明确 |
| 1：multi 原型 | 同构建比较 locking/multi；SGEMM、BMM、einsum 三层测试 | 库加载正确，线程控制有效，B1 收益和 B32 回归均量化 |
| 2：互斥计划 | 在同一个 multi 库内标定 serial/outer/inner，再对比旧部署 | 未参与标定的形状上有效；无线程状态泄漏或失控嵌套 |
| 3：布局原型 | view、完整物化、tile 收集三路消融 | 工作空间与端到端分别达标，受限布局正确 |
| 4：尾延迟与 batch API | 同库比较 outer/batched，采集等待与亲和性证据 | 绝对 P95 改善且中位数不明显回退 |
| 5：高级方案 | 按剩余瓶颈决定 hybrid 或预打包 | 对最佳已有方案有新增收益，且真实工作负载需要 |

同编译器、同 ABI、同 oneDNN 选项的 OpenBLAS/KML 构建用于进一步分离 BLAS
与构建栈贡献；它不替代更优先的同 KML 库调度消融。

## 12. 验证矩阵与建议门槛

### 12.1 保留历史矩阵并增加有针对性的覆盖

| 维度 | 验证范围 |
|---|---|
| 历史性能矩阵 | S/M/L/XL × X0/X4 × B1/B4/B32 × T1/T2/T4/T8/T32 |
| 形状扩展 | small-M/N/K、非方形、非整齐尺寸、边界 tile |
| 布局扩展 | X1 简单转置、X2/X3/X5；高阶张量使用相应的分组/交错模式 |
| batch 扩展 | B2 及非规则 batch 数，例如 3、5、33 |
| 张量阶数 | 原有 4D/5D/6D 正确性和性能回归 |
| 数值覆盖 | batch 不同的可复现输入、零值、近零结果、不同数值尺度 |
| 并发覆盖 | 重复调用、并发调用、嵌套调用的受控 fallback、状态恢复 |
| 复用覆盖 | 仅预打包方案额外测试固定权重和权重更新 |

新增输入/布局/规模使用新标识，避免与历史结果混合。
新增矩阵分为核心回归、正确性边界和扩展性能集，不要求所有边界条件做完整笛卡尔积。

### 12.2 性能指标与公平性

正式比较以端到端中位数、绝对 P95、原始样本和 campaign 差异为主。
同时记录物化字节、工作空间、分配行为、实际计算线程、进程 CPU 消耗与误差。
硬件带宽若无法测量，记为 N/A；算法最少字节/时间不能标成实测 DDR 带宽。

历史 PyTorch [计时循环](/root/benchmark/tensor_contraction_redesign/benchmark_tblis_batch_pytorch.py:145)
每次调用 einsum 产生新输出，赋值时还涉及旧输出生命周期；
TBLIS 在 [运行器](/root/benchmark/tensor_contraction_redesign/tblis_layout_runner.cpp:717)
中预分配并复用输出。后续保留实际框架端到端指标，另设输出分配语义一致的
后端对照。不能把不同输出生命周期下的差值全部归因于调度或内核。

正式性能运行关闭细粒度诊断，与 profiler/线程诊断分开。
TBLIS 当前包含本地阶段插桩；需要判断纯后端差距时应另加关闭该插桩的对照。

### 12.3 建议工程门槛

以下百分比是建议验收目标，不是现有实验推导出的保证：

1. 正确性通过现有归一化误差 1e-4 门限和显式有限数检查；新并行路径通过
   batch 不同输入、非整齐尺寸和跨调用状态测试。
2. 原 KML 胜出的 B32 配置中位数回退目标不超过 3%，同时检查其他保护用例。
3. L/XL B1、XL B4 的改善需要跨 campaign 复现，且大于测量不确定性；
   S/X0 约 4.7% 的跨实现差距不应被当作同等强度的大收益证据。
4. 保护用例绝对 P95 回退目标不超过 5%；尾延迟优化的目标用例应有可复现的
   绝对 P95 改善，同时满足中位数约束。
5. 诊断关闭时的额外中位开销目标不超过 1%；若测量噪声大于该阈值，
   增加重复与 campaign，不用单次差值作判断。
6. 验证计算线程预算、team、嵌套行为及线程状态恢复；空闲线程总数或
   `/proc` 活跃线程数不能单独充当是否超额订阅的判据。
7. 布局方案分别报告物化、工作空间和端到端结果；只减少 clone 不足以证明加速。

先沿用每配置 3 个 campaign、每次预热 10 次和正式 50 次以便对照。
接近阈值、长尾或首次出现异常的配置，应增加独立 campaign 和样本；
最佳线程与 planner 阈值应在独立验证集上复核，避免挑选最小值造成乐观偏差。

## 13. 结论边界

当前证据足以支持“优先验证 KML multi 并建立互斥调度，再探索布局分块收集”，
但以下内容仍需原型回答：

- KML multi 的 B1 扩展能达到多少，以及单线程 outer 是否回退；
- KML 自身最佳线程数和 serial/outer/inner 切换阈值；
- 分块收集是否在降低整输入物化后仍改善端到端，或因重复 packing 变慢；
- batch API、等待策略或亲和性中哪项真正改善 KML 长尾；
- hybrid、预打包相对最佳已有路径是否还有额外收益。

已有结果是当前 ECS、库版本、部署方式、输入与计时范围下的证据。
TBLIS 版本为 `v2.0-beta2-14-geb719e7-dirty` 加本地 profiling 补丁；
OpenBLAS/KML PyTorch 的编译器、ABI 和 oneDNN 选项不同，且本实验固定
OpenBLAS 内部线程为 1。不能据此宣称 KML 优于最佳多线程 OpenBLAS，
也不能将当前 locking 部署的限制概括为所有 KML 或所有 KPEX 版本的缺陷。
