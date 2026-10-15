# KML SVE multi 兼容性与线程扩展验证

日期：2026-10-14。本报告验证当前安装的 KML 26.1.RC1 BiSheng SVE multi
变体及现有 patched PyTorch 的接入方式，不修改历史 locking 环境和实验结果。

**后续更新：** 本文保留的是当时仅有 1 个 vCPU 的验证记录。随后环境恢复为
32 vCPU，已完成 120 配置的完整 PyTorch-KML multi 实验，并通过独立 BLAS/OMPT
诊断确认单个 GEMM 的真实多线程执行。最新结果见
[32 vCPU 实验分析](/root/benchmark/tensor_contraction_redesign/results/pytorch_kml_multi_cpu0-31_20261014T101523721193Z/analysis.md)。

## 1. 已确认结果与尚未解决的问题

**SVE multi 可以在当前环境执行 FP32 GEMM，没有在已测 GEMM 中触发非法指令。**
对照库为同一 SDK 的 SVE locking。正式验证分别覆盖 CBLAS/Fortran GEMM
接口和完整 PyTorch B1 einsum；共 60 组配置、9,000 次正式计时和 60 个独立
诊断任务全部通过，结果文件与复现方法见下文。

**当前无法完成真实多核线程扩展验证。** 本轮系统只暴露 CPU 0，在线、present、
possible CPU 集合均为 `0`，进程亲和性为 `{0}`；沙箱外的 `lscpu` 只读检查也
报告 1 个 CPU。历史报告使用的 32-vCPU 配置目前不可用。

KML multi 报告 OpenMP 线程模型，但在请求 1/2/4/8/32 线程时，有效 BLAS
线程数均为 1。这些高请求线程记录是资源受限的诊断，不能据此宣称 KML multi
不能多线程，也不能将它们与 T1 的耗时比作为真正的多核扩展结果。

## 2. 运行方式和证据位置

| 项目 | 配置 |
|---|---|
| multi 动态库 | `/root/benchmark/pytorch-kunpeng/hpckit-components/KunpengHPCKit-kml.26.1.RC1/bishenglib/sve/kblas/multi/libkblas.so.1.26.1.RC1` |
| 对照库 | 同一目录层级的 `sve/kblas/locking/libkblas.so.1.26.1.RC1` |
| OpenMP runtime | BiSheng 5.2.0 的 `lib/aarch64-unknown-linux-gnu/libomp.so` |
| CPU 与 batch | 当前可用 CPU 0；所有数学运算 batch=1 |
| 请求线程数 | 1、2、4、8、32 |
| 数值类型 | FP32，独立 NumPy FP64 全输出参考 |
| 正式采样 | 每配置 3 个 campaign，每轮预热 10 次，正式记录 50 次 |
| 等待配置 | `OMP_WAIT_POLICY=PASSIVE`、`KMP_BLOCKTIME=0`、`OMP_DYNAMIC=FALSE`、`OMP_MAX_ACTIVE_LEVELS=1` |
| 正式进程的工具设置 | `OMP_TOOL=disabled`；计时不启用 OMPT |
| 诊断进程的工具设置 | `OMP_TOOL=enabled`，明确指定本次 OMPT observer |

使用 [probe_kml_multi.py](/root/benchmark/tensor_contraction_redesign/probe_kml_multi.py)
创建隔离子进程，以绝对路径先加载选中的 KML，再执行相应接口。
每个成功任务校验进程只映射目标 KML 库，避免只根据编译配置推断实际后端。

线程诊断由 [kml_probe_observer.cpp](/root/benchmark/tensor_contraction_redesign/kml_probe_observer.cpp)
在独立调用区间记录 OpenMP parallel/implicit-task 事件。它记录的是并行 team，
不会把 `/proc` 中所有活跃线程都当成 GEMM 计算线程。

| 数据集 | 内容 |
|---|---|
| [native 结果](/root/benchmark/tensor_contraction_redesign/results/kml_multi_probe_20261014T094531287120Z/report.md) | S/M/L/XL、locking/multi、全部请求线程数的 CBLAS GEMM |
| [native 原始数据](/root/benchmark/tensor_contraction_redesign/results/kml_multi_probe_20261014T094531287120Z/raw.tsv) | 正式逐次时间、进程 CPU 时间及 campaign |
| [native 线程诊断](/root/benchmark/tensor_contraction_redesign/results/kml_multi_probe_20261014T094531287120Z/thread_diagnostics.tsv) | 独立 OMPT 诊断 |
| [加载对照](/root/benchmark/tensor_contraction_redesign/results/kml_multi_probe_20261014T094531287120Z/import_audit.json) | 默认与显式 OpenMP 工具设置下的导入/GEMM 成败 |
| [PyTorch 结果](/root/benchmark/tensor_contraction_redesign/results/kml_multi_probe_20261014T094820441179Z/report.md) | XL 的 X0/X4、locking/multi、全部请求线程配置 |
| [PyTorch 原始数据](/root/benchmark/tensor_contraction_redesign/results/kml_multi_probe_20261014T094820441179Z/raw.tsv) | 3,000 次正式完整 einsum 计时 |

每次运行还保存 CPU/配额、代码和库 SHA256、输入 SHA256、实际动态库映射、
线程 API 返回值、错误日志和完整精度结果。

## 3. Native GEMM 验证

不变量：每个规模内的输入矩阵、FP32、输出尺寸、采样方式、CPU 0。
变量：locking/multi、请求线程数；当前有效 BLAS 线程并没有随请求增加。

结论：40 组正式配置共 6,000 次 GEMM 计时通过，40 个独立 OMPT 诊断任务
也通过。S/M/L/XL 的每次结果检查覆盖全部输出元素，最大归一化误差为
**1.7716594518546458e-5**，低于 1e-4 门限。

| 规模 | M | N | K | locking T1 中位数/ms | multi T1 中位数/ms |
|---|---:|---:|---:|---:|---:|
| S | 256 | 256 | 64 | 0.114659 | 0.114494 |
| M | 512 | 512 | 128 | 0.836873 | 0.839168 |
| L | 1024 | 1024 | 256 | 6.417589 | 6.401043 |
| XL | 2048 | 2048 | 512 | 50.358206 | 50.412666 |

这些数值是当前单核上的接口边界时间，包含 ctypes 调用和完整 BLAS 工作，
输出预先分配，不等于纯微内核时间。两库在本轮单线程点位接近，不能外推为
32 核条件下的速度或 B32 outer 路径回归结论。

每个 native 子进程还验证以下内容，共 160 次完整规模 Fortran ABI 校验和
640 次小矩阵边界校验：

- `sgemm_`：验证 PyTorch FP32 路径使用的 Fortran ABI，结果对照独立 FP64；
- NN、NT、TN、TT 四种转置，矩阵 M=13、N=17、K=29；
- 输入 leading dimension 加 padding，输出 leading dimension 加 padding；
- alpha=0.75、beta=-0.25，并检查输出 padding 未被覆盖；
- 所有输出为有限数，记录最大绝对误差和带分母下限的归一化误差。

误差公式为 `max(abs(result-reference)/max(abs(reference),1))`。
该指标不是严格逐元素相对误差；完整参考使用 NumPy 的独立 FP64 matmul，
元数据记录其 OpenBLAS 动态库，KML 不负责生成参考结果。

## 4. 多线程请求与实际执行的区别

不变量：当前可用处理器数为 1，每个调用只包含一个 GEMM。
变量：通过环境及 `BlasSetNumThreads`/`BlasSetNumThreadsLocal` 请求不同线程数。

结论：multi 确实是 OpenMP 变体，但本轮所有请求均退回/限制为单线程执行。

| 请求线程 | multi `BlasGetParallel` | `BlasGetNumProcs` | `BlasGetNumThreadsLocal` | native GEMM 诊断捕获的并行 team |
|---:|---:|---:|---:|---|
| 1 | 2（OpenMP） | 1 | 1 | 无 |
| 2 | 2（OpenMP） | 1 | 1 | 无 |
| 4 | 2（OpenMP） | 1 | 1 | 无 |
| 8 | 2（OpenMP） | 1 | 1 | 无 |
| 32 | 2（OpenMP） | 1 | 1 | 无 |

multi 的 OMPT 工具初始化成功，相关回调返回支持状态，但 GEMM 区间没有
parallel/implicit-task team 事件。结合库报告的有效线程数 1，本轮没有证据
表明单个 GEMM 曾使用两个或更多计算线程。

在 PyTorch X4 中，整个 einsum 的并行事件还可能来自输入复制/转换；因此即使
观察到框架的多个线程，也不能替代 KML 内部多线程的证据。

## 5. PyTorch 接入遇到的加载问题

不变量：同一 patched PyTorch、CPU 0、SVE multi 库、T1 和小矩阵 GEMM。
变量：OpenMP 工具发现设置；对照额外包含 locking 默认设置。

结论：默认工具探测下 multi 会在 `import torch` 阶段出现 SIGSEGV；显式关闭
工具探测，或指定本次 observer 后均可导入并通过小矩阵 GEMM。每种配置独立重复两次。

| 库 | 工具设置 | 两次结果 |
|---|---|---|
| locking | OMP_TOOL 未设置，OMP_TOOL_LIBRARIES 未设置 | 2/2 成功 |
| multi | OMP_TOOL 未设置，OMP_TOOL_LIBRARIES 未设置 | 2/2 SIGSEGV（返回 -11） |
| multi | `OMP_TOOL=disabled` | 2/2 导入和 GEMM 成功 |
| multi | `OMP_TOOL=enabled`，指定本次 observer | 2/2 导入和 GEMM 成功 |

这是当前构建栈的加载/工具发现兼容问题。SIGSEGV 与早期 matrix 变体的 SIGILL
不同；不能解释为 SVE multi 使用了硬件不支持的指令。具体底层根因尚未完全定位，
上述对照证明的是可复现的条件与可用绕行方式。

正式测试统一关闭 OMPT 自动探测，在需要诊断时才显式加载 observer。
本轮未替换已安装库的软链接，也未修改历史 `FindKML.cmake` 和运行器。

## 6. PyTorch 完整 B1 einsum 验证

不变量：XL（M=N=2048、K=512）、P=1、同一逻辑输入和当前 CPU 0。
变量：X0/X4、locking/multi，以及请求 T1/T2/T4/T8/T32。

结论：20 组正式配置、3,000 次正式调用和 20 个独立诊断任务全部通过；
multi 的实际动态库映射已确认，最大归一化误差同为 1.7716594518546458e-5。

| 布局 | 库 | 请求线程 | KML 报告线程 | 中位数/ms | P95/ms |
|---|---|---:|---:|---:|---:|
| X0 | locking | 1 | 1 | 50.173515 | 52.745863 |
| X0 | multi | 1 | 1 | 50.166742 | 54.067807 |
| X0 | locking | 32 | 1 | 49.949724 | 52.533927 |
| X0 | multi | 32 | 1 | 49.822890 | 51.993580 |
| X4 | locking | 1 | 1 | 50.661402 | 54.642329 |
| X4 | multi | 1 | 1 | 50.621324 | 56.474445 |
| X4 | locking | 32 | 1 | 51.272668 | 52.922456 |
| X4 | multi | 32 | 1 | 51.474508 | 53.031958 |

请求 T32 但 CPU 容量仍为 1，上表只说明该配置下完整调用可执行，不能解释为
32 核性能。X4/multi/T32 的整个 einsum 诊断捕获两个 32-thread team，
但 KML 报告仍是单线程，X0 和直接 GEMM 没有相同的并行 team 事件。
这与框架转换阶段出现并行活动相符，并不构成 KML GEMM 内部并行的证据。

本轮复算两份 raw.tsv，确认 60 个汇总组均有 3×50 条正式记录，中位数和
nearest-rank P95 与 summary.tsv 完全一致；所有任务使用预期 KML 路径，
同规模输入指纹在库、线程和 campaign 之间一致。

## 7. 本轮结果不能证明什么

- 尚未证明 SVE multi 的两个及以上 GEMM 计算线程实际工作，也没有真实多核加速比；
- 尚未证明 multi 比 locking 更快或更慢，不能把当前单核上请求 T32 的时间当作 T32 性能；
- PyTorch 接入验证仅针对 B1，未验证在 batch outer 区域内再启动 KML inner 的嵌套调度；
- 输入、输出生命周期、等待策略和资源配置与旧实验不同，不能直接合并到旧 32-vCPU 排名；
- 接口、线程模型或导出符号存在，不等价于当前资源下已执行相应的多核算法路径。

## 8. 复现及恢复多核后的下一步

```bash
cd /root/benchmark/tensor_contraction_redesign
python3 probe_kml_multi.py --engines cblas --audit-imports
python3 probe_kml_multi.py --engines torch --scales XL
```

脚本自动保存新的时间戳结果目录，默认请求 1/2/4/8/32 线程，默认每组
3 个 campaign、预热 10 次、正式 50 次。结果中请求线程、有效 BLAS 线程、CPU
容量和 OMPT 诊断分别记录；超出 CPU 容量的多核扩展字段记为 N/A。

要完成用户要求的真实多核扩展，需要当前运行环境重新提供多个可用 vCPU：
至少 2 个才能验证双核，完整 T1/T2/T4/T8/T32 需要至少 32 个可用 vCPU，且配额允许。
恢复后使用相同命令复测，首先确认 KML 有效线程数和 native GEMM 的实际 team
大于 1，再解释加速比。确认 B1 内部并行有效后，才进入 B4/B32 的互斥调度和回归验证。
