# 鲲鹏 PyTorch/KML 补丁优化与实验证据分析

## 1. 报告范围

本报告分析 KPEX 仓库针对 PyTorch v2.5.0 的 KML 补丁，并用本仓库已经完成的
两组 FP32 `torch.einsum` 实验验证其实际效果。分析对象如下：

- 基线 `pytorch_2.5.0_openblas`：官方 aarch64 PyTorch 2.5.0 wheel，
  `BLAS_INFO=open`，GCC 10.2，`USE_MKLDNN=ON`。
- 优化版本 `pytorch_2.5.0_kml_patched`：PyTorch v2.5.0 应用 KPEX 补丁，
  使用 BiSheng 5.2.0 构建，`BLAS_INFO=KML`，`USE_MKLDNN=0`。
- 补丁文件：`/root/benchmark/pytorch-kunpeng/kpex-current/thirdparty/pytorch-v2.5.0.patch`，
  KPEX 提交 `772e09f8c84e029d36e88c0af8d28bfaffdadd08`，补丁 SHA256 为
  `1c5e56b76acfd8975352ec332e684a6d7b3d3b6f3f6519ae433ae8961f096637`。
- 当前实验均为 CPU FP32。BF16、FP16、INT8、LAPACK 和 MPI 改动只做源码审计，
  不用 FP32 实验结果替它们背书。

为避免过度归因，本文使用三种证据等级：

| 等级 | 含义 |
|---|---|
| 源码确认 | 能从补丁和 PyTorch 调用路径直接确认实现机制 |
| 实验确认 | 有正式、未启用 profiler 的端到端计时和正确性结果支持 |
| 合理推断 | 机制和现象吻合，但缺少单变量消融，不能视为严格因果证明 |

## 2. 核心结论

本轮 FP32 `einsum` 真正命中的内容可以收敛为两项性能优化和一项布局适配机制：

1. **将 PyTorch 的 BLAS 后端从 OpenBLAS 换为 KML/KBLAS。** FP32 GEMM 仍经标准
   `sgemm_` ABI 调用，补丁没有另写 FP32 微内核。230 个高阶缩并配置中 KML
   版本全部更快，端到端中位加速比的几何平均为 **1.985x**；在更大的单 batch、
   单线程配置中，8/8 个配置更快，中位加速为 **1.206x**。
2. **把 CPU BMM 的不同 batch 项放到 ATen 线程池并行执行。** 这是多 batch
   扩展的主要来源。KML 版本从 T1 到 T32 的中位加速为：B1 **1.004x**、B4
   **3.837x**、B32 **19.671x**；任务数随 batch 增长的阶梯形状与补丁中的
   `parallel_for(0, bs, 1)` 直接吻合。
3. **在整个 BMM 层统一进行矩阵 stride 判定、转置映射和必要的连续化。** 简单
   连续或转置矩阵可以直接传给 BLAS；不能表示为普通/转置矩阵的输入会先
   `clone(Contiguous)`。源码上集中处理这些判断，理论上可避免逐 batch 重复路径
   开销，但本实验未单独消融这一贡献，也没有实现任意 stride 的零复制缩并。
   X4 交错布局仍发生 A、B 各一次完整物化，同时观察到 **11.9%–30.0%** 的 KML
   端到端增幅；诊断定位表明本组 X4 物化发生在 `einsum` 上游矩阵化阶段，而
   不是 `bmm_impl_cpu_` 内部的 clone fallback。

因此，准确的总体表述是：**该优化栈同时改善了鲲鹏上的 FP32 GEMM 后端，并为
PyTorch BMM 增加了 batch 任务级并行；其突出优势是多 batch 吞吐，而不是把单个
大 GEMM 自动拆成多核，也不是为 `einsum` 实现了原生 stride-aware 张量缩并。**

## 3. `einsum` 实际执行路径

### 3.1 补丁没有直接重写 `einsum`

PyTorch 2.5.0 的 `sumproduct_pair` 负责两输入 `einsum` 的成对规约。当前源码
`aten/src/ATen/native/Linear.cpp:142-145,185-190,228-232` 明确执行：

```text
torch.einsum
  -> permute 输入
  -> reshape 为三维矩阵批次
  -> at::bmm(left, right)
  -> view 输出
  -> permute 恢复输出轴序
```

对应核心代码仍是上游实现：

```cpp
left = left.permute(lpermutation).reshape_symint({lro_size, lo_size, sum_size});
right = right.permute(rpermutation).reshape_symint({lro_size, sum_size, ro_size});
Tensor result = at::bmm(left, right);
result = result.view_symint(out_size).permute(opermutation);
```

KPEX 补丁没有修改 `Linear.cpp`，所以它不是一个新的 `einsum` planner，也没有
绕开 TTGT 风格的 `permute -> reshape -> BMM -> restore`。补丁的作用点位于下游
`bmm` 和 BLAS。

### 3.2 本报告被测配置的 FP32 路径

```text
torch.einsum
  -> 上游 sumproduct_pair
  -> aten::bmm
  -> KPEX 修改后的 bmm_impl_cpu_
  -> 按 batch 的 at::parallel_for
  -> 每个 batch 调用 at::native::cpublas::gemm
  -> 标准 sgemm_ 符号
  -> 运行时 KML libkblas.so
```

这不是所有 FP32 `einsum` 的无条件路径。PyTorch 会在矩阵工作量
`contraction_size * res_rows * res_cols < 400` 时优先进入
`baddbmm_cpu_kernel`，还保留 MKLDNN/MKL 分支。本报告的被测配置超过该阈值，
KML 构建又使用 `USE_MKLDNN=0` 且不是 MKL 构建，因此进入新增的
`bmm_impl_cpu_`；结论只在这些已核验条件下成立。

运行时审计确认加载了
`KunpengHPCKit-kml.26.1.RC1/bishenglib/sve/kblas/locking/libkblas.so.1.26.1.RC1`。
`libtorch_cpu.so` 依赖 `libkblas.so.1`，其 FP32 路径在
`CPUBlas.cpp:272-317` 调用 `sgemm_`。所以“使用 KML FP32 内核”是链接结果，
而不是补丁在 `CPUBlas.cpp` 新增了另一套 FP32 算法。

当前实验运行 patched PyTorch 本体；`einsum` 不依赖 KPEX Python 扩展的自定义
`mm` 算子。因而不能把这些结果归因于 KPEX 扩展层的算子 hook。

## 4. 与 FP32 相关的实现与已验证优化

### 4.1 KML BLAS 构建接入

补丁在 `cmake/Dependencies.cmake` 增加 `BLAS=KML` 分支，并新增
`FindKML.cmake`：查找 KBLAS、写入 `BLAS_INFO=KML`，再探测 KML 提供的 BF16、
FP16、INT8 和预打包符号。FP32/FP64 沿用 PyTorch 已有 BLAS ABI，因此链接 KML
后，`sgemm_`/`dgemm_` 自然由 KBLAS 提供。

当前实际部署源码将原补丁中的 `matrix/kblas/multi` 搜索路径调整为已安装的
`sve/kblas/locking`，并重新查找 CMake cache 中的 KML 库。报告中的结果对应这份
本地部署版本，而不是未经环境适配的原始补丁文本。

源码确认的效果：

- PyTorch 能以 KML 作为 BLAS 后端完成构建；
- FP32 `bmm/mm` 最终调用 KML 的标准 `sgemm_`；
- 运行时确实加载 KML `libkblas.so`，不是仅有编译配置字符串。

### 4.2 CPU BMM 的 batch 外层并行

补丁在 `LinearAlgebra.cpp` 新增 `bmm_impl_cpu_`，并用它替换原先非 mobile CPU
路径中逐 batch 串行调用 `addmm_impl_cpu_` 的循环。关键结构为：

```cpp
at::parallel_for(0, bs, 1, [&](uint64_t start, uint64_t end) {
  for (const auto batch : c10::irange(start, end)) {
    at::native::cpublas::gemm(/* batch-offset pointers */);
  }
});
```

其语义非常明确：

- `batch=1` 只有一个外层任务，增加 ATen 线程数基本没有帮助；
- `batch=4` 最多形成 4 个独立 GEMM 任务；
- `batch=32` 才能向 32 个 worker 提供足够任务；
- 它仍然是“每个 batch 一次 GEMM”，不是一次 fused batched GEMM；
- 它没有把单个 GEMM 的 M/N/K 空间进一步切给 ATen worker。

### 4.3 BMM 级布局判断和一次性物化

`bmm_impl_cpu_` 在进入并行循环之前统一检查输出、A、B 的最后两个矩阵维 stride：

- 矩阵连续或可作为普通 BLAS 转置表示时，直接使用原存储及 `lda/ldb/ldc`；
- 输出布局不适用时，建立适合 BLAS 的临时输出，结束后 `copy_` 回结果；
- A 或 B 无法映射为普通/转置矩阵时，对相应输入执行连续化 clone；
- 通过第 0 维 batch stride 在并行循环中计算每个 GEMM 的指针偏移。

这会集中处理布局与 clone，理论上避免逐 batch 重复判断，也允许简单转置零复制；
但该部分开销没有用同 BLAS、同构建的旧 BMM 路径做单变量消融。它不能消除
`einsum` 上游 `reshape` 可能触发的复制，并且对真正交错的矩阵轴仍需要完整物化。
该 clone fallback 在源码上存在；本次 X4 诊断记录的两次 clone 位于进入 BMM
之前，故本实验没有单独量化 BMM 内部 clone fallback 的性能效果。

## 5. 实验证明一：4/5/6 维高阶缩并

结果目录：
`results/contraction_cpu0-3_20261009T163753742611Z`。

实验固定逻辑体量 `B*M*N*K=2^20`，覆盖 4D、5D、6D、不同形状、缩并维数量、
布局、batch 和线程数。每个配置执行 3 个独立 campaign，每个 campaign 预热 10
次、正式记录 50 次。OpenBLAS 与 KML 有 230 个完整配对配置，每方法 34,500 个
正式样本。

### 5.1 总体结果

| 指标 | 结果 |
|---|---:|
| 完整配对配置 | 230 |
| KML 端到端中位数胜出 | 230/230 |
| 中位加速比 | 1.932x |
| 加速比几何平均 | 1.985x |
| 加速比范围 | 1.271x–2.810x |
| KML P95 胜出 | 230/230 |
| P95 比值中位数 / 几何平均 | 1.880x / 1.943x |

加速比统一定义为 `OpenBLAS 端到端时间 / KML 端到端时间`，大于 1 表示 KML
版本更快。

### 5.2 不同张量阶数和用例族

| 分类 | 配对数 | KML 胜出 | 加速比几何平均 |
|---|---:|---:|---:|
| 4D | 63 | 63 | 2.014x |
| 5D | 77 | 77 | 1.983x |
| 6D | 90 | 90 | 1.968x |
| shape | 122 | 122 | 2.089x |
| layout | 60 | 60 | 1.825x |
| batch | 36 | 36 | 1.932x |
| layout + shape | 12 | 12 | 1.955x |

收益不局限于一个张量阶数。复杂 layout 组的几何平均收益降到 1.825x，这与数据
重排开销稀释后端 GEMM 收益的解释一致，但聚合结果本身不能单独证明该因果关系；
这组固定体量实验中 KML 版本仍全部胜出。

### 5.3 早期实验中的 batch 扩展

| Batch | T1 | T2 | T4 | KML 自身 T1→T4 | OpenBLAS 自身 T1→T4 |
|---|---:|---:|---:|---:|---:|
| B1 | 1.806x | 1.853x | 1.883x | 1.002x | 0.941x |
| B4 | 1.546x | 2.173x | 2.604x | 1.699x | 1.015x |
| B16 | 1.282x | 1.954x | 2.628x | 2.101x | 1.016x |
| B64 | 1.342x | 2.005x | 2.715x | 2.093x | 1.035x |

前三列为同一 batch/线程下 OpenBLAS/KML 端到端时间比的中位数。B1 基本不随
线程增加，而多个 batch 在 KML 版本中出现扩展；该现象已经与 batch 外层并行
机制一致，但受限于当时最多 4 个 CPU，扩展上限还不够清晰。

## 6. 实验证明二：S/M/L/XL 大规模多 batch、多线程

结果目录：
`results/tblis_batch_cpu0-31_20261011T074035858907Z`。

实验固定表达式 `pabkl,pklcd->pabcd`，交叉 4 个规模、X0/X4 两种布局、
B1/B4/B32、T1/T2/T4/T8/T32，共 120 个 OpenBLAS/KML 配对配置。每组同样有
3×50 个正式样本；每方法 18,000 个样本。T 表示 PyTorch intra-op 线程配置，
线程亲和性限制在对应 CPU 集合；实验进程同时设置 `OMP_NUM_THREADS=1`、
`OPENBLAS_NUM_THREADS=1` 等环境变量，避免基线内部 BLAS 另行扩展。

### 6.1 总体结果和线程维度

| 统计范围 | KML 胜出 | 加速比中位数 | 加速比几何平均 |
|---|---:|---:|---:|
| 全部 120 配置 | 113/120 | 1.840x | 2.349x |
| T1，24 配置 | 17/24 | 1.114x | - |
| T2，24 配置 | 24/24 | 1.840x | - |
| T4，24 配置 | 24/24 | 3.422x | - |
| T8，24 配置 | 24/24 | 3.955x | - |
| T32，24 配置 | 24/24 | 3.970x | - |

P95 指标中 KML 胜出 114/120，P95 时间比中位数为 1.858x。T1 的收益明显小于
多线程配置，说明大幅收益并非仅来自换 BLAS 库。

### 6.2 最接近后端单线程对照的 B1/T1

B1/T1 没有 batch 并行可用，8/8 配置中 KML 版本更快，加速比中位数为
**1.206x**：

| 规模/布局 | S-X0 | S-X4 | M-X0 | M-X4 | L-X0 | L-X4 | XL-X0 | XL-X4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenBLAS/KML | 1.315x | 1.362x | 1.267x | 1.182x | 1.043x | 1.090x | 1.163x | 1.230x |

该组支持“鲲鹏 PyTorch/KML 构建栈的单线程后端更快”，但仍不是纯 KML 微内核
消融，因为两个构建还同时改变了编译器、ABI、oneDNN 等选项。

### 6.3 batch 外层并行的直接证据

| 方法 | B1 T1→T32 | B4 T1→T32 | B32 T1→T32 |
|---|---:|---:|---:|
| PyTorch-OpenBLAS | 0.978x | 1.018x | 1.039x |
| PyTorch-KML patched | 1.004x | 3.837x | 19.671x |

KML 版本的扩展能力随可并行 batch 数从 1、4、32 呈清晰阶梯；B1 几乎完全不
扩展，排除了“补丁自动并行单个大缩并”的解释。B32 的 19.671x 又显著低于理想
32x，说明存在非理想扩展；缺口可能包含调度、同步、物化、带宽、频率和固定
开销等因素，当前计时不能分离各自贡献。

下表给出 KML 相对 OpenBLAS 的中位加速比：

| Batch | T1 | T2 | T4 | T8 | T32 |
|---|---:|---:|---:|---:|---:|
| B1 | 1.206x | 1.189x | 1.197x | 1.238x | 1.277x |
| B4 | 1.130x | 2.081x | 3.842x | 3.955x | 3.970x |
| B32 | 0.972x | 1.881x | 3.466x | 6.579x | 18.951x |

B32/T32 的 8 个配置全部胜出：

| 用例 | OpenBLAS 中位数 | KML 中位数 | 加速比 |
|---|---:|---:|---:|
| S-X0 | 4.085 ms | 0.205 ms | 19.942x |
| S-X4 | 3.664 ms | 0.266 ms | 13.759x |
| M-X0 | 26.044 ms | 1.537 ms | 16.941x |
| M-X4 | 26.518 ms | 1.734 ms | 15.290x |
| L-X0 | 195.927 ms | 8.555 ms | 22.903x |
| L-X4 | 198.110 ms | 11.030 ms | 17.961x |
| XL-X0 | 1763.444 ms | 56.607 ms | 31.152x |
| XL-X4 | 1779.201 ms | 63.327 ms | 28.095x |

这组数据是补丁 batch 并行有效性的最强证据。它证明“逐 batch 调用 KML BLAS”
不等于整个 batch 串行，因为不同 batch 的 GEMM 由 ATen worker 并发执行。

### 6.4 布局与物化证据

固定 B32/T32，X4 被确认发生输入物化，同时其 KML 端到端时间相对 X0 增加
11.9%–30.0%。X4 也改变了 stride 访问条件，因此不能把全部增幅严格归因于 clone：

| 规模 | X0 | X4 | X4/X0 | X4 输入物化总量 |
|---|---:|---:|---:|---:|
| S | 0.205 ms | 0.266 ms | 1.300x | 4 MiB |
| M | 1.537 ms | 1.734 ms | 1.128x | 16 MiB |
| L | 8.555 ms | 11.030 ms | 1.289x | 64 MiB |
| XL | 56.607 ms | 63.327 ms | 1.119x | 256 MiB |

独立 profiler/dispatch 审计显示：

- X0 四个规模的 `input_allocation_bytes` 均为 0；
- X4 对 A、B 各 materialize 1 次；
- S/M/L/XL 的两输入合计物化量依次为 4/16/64/256 MiB；
- X4 的 `inside_bmm_materialize_self_ns` 为 0，clone 位于 `einsum` 的上游
  `permute/reshape` 矩阵化阶段，而不是 patched BMM 内部；
- profiler 计时只用于定位路径，不能从无 profiler 的正式端到端时间中直接相减。

这证明完整 PyTorch/KML `einsum` 路径仍需为 X4 做整输入物化。补丁中的 BMM
stride 判定不能消除上游矩阵化成本，整个框架路径还没有达到 TBLIS 一类直接
消费一般 stride 的能力。

### 6.5 反例和适用边界

当前大规模实验中 KML 并非 120/120 全胜。7 个反例全部出现在 T1：

| 用例 | OpenBLAS | KML | OpenBLAS/KML | 解释 |
|---|---:|---:|---:|---|
| M-X0-B32-T1 | 27.171 ms | 29.082 ms | 0.934x | KML 慢约 7.0% |
| M-X4-B32-T1 | 28.691 ms | 30.615 ms | 0.937x | KML 慢约 6.7% |
| L-X0-B32-T1 | 202.801 ms | 210.968 ms | 0.961x | KML 慢约 4.0% |
| L-X4-B32-T1 | 211.381 ms | 219.524 ms | 0.963x | KML 慢约 3.9% |
| L-X4-B4-T1 | 25.884 ms | 26.581 ms | 0.974x | KML 慢约 2.7% |
| S-X4-B32-T1 | 4.335 ms | 4.414 ms | 0.982x | KML 慢约 1.8% |
| L-X0-B4-T1 | 25.688 ms | 25.744 ms | 0.998x | 接近噪声范围 |

B32/T1 只有 3/8 胜出，中位时间比为 0.972x。合理结论是：KML 对常见单缩并
通常有收益，而补丁最显著的优势来自多个 batch 的并行；不能宣称 KML 内核在
所有大 batch 单线程场景都优于 OpenBLAS。

## 7. 正确性与稳定性

| 数据集 | 方法 | 最大相对误差 | campaign spread 中位数 | campaign spread 最大值 |
|---|---|---:|---:|---:|
| 4/5/6D | OpenBLAS | 1.449e-6 | 3.333% | 8.672% |
| 4/5/6D | KML | 6.266e-7 | 1.675% | 5.767% |
| S/M/L/XL | OpenBLAS | 2.205e-5 | 1.711% | 19.582% |
| S/M/L/XL | KML | 2.205e-5 | 0.686% | 5.432% |

两组数据都通过各自的误差门限，说明性能收益没有破坏本实验要求的数值正确性。
不同 BLAS 的浮点归约顺序不同，不能根据最大误差更小就声称 KML 数值算法“更
准确”。campaign spread 反映本次运行的复现性，不应被写成补丁设计目标。

## 8. 补丁包含但本实验未验证的优化

| 补丁能力 | 主要源码改动 | 当前 FP32 `einsum` 是否命中 |
|---|---|---|
| BF16 KML GEMM | 增加 `bgemm_` 路径 | 否 |
| FP16 GEMM/GEMV | 增加 `hgemm16_`、`hgemv_` | 否 |
| INT8 GEMM/GEMV | 增加 `igemm_`、`igemv_`，输出 INT32 | 否 |
| INT8 预打包 | 新增 `ipacking_`、packed transpose flag | 否 |
| KML LAPACK | 新增 KML LAPACK 查找与链接 | 否 |
| MPI FP16 类型 | 使用 `MPIX_SHORT_FLOAT` | 否，当前 `USE_MPI=OFF` |
| FP32 GEMV 分流 | 声明了 `sgemv_` | 否，新增 `gemm_distribute` 只被 FP16/INT8 调用 |

当前构建探测到 `BLAS_HAS_BGEMM=1` 和 `BLAS_HAS_SBGEMM=1`，但没有探测到 FP16、
INT8 和 `ipacking` 所需符号。即使补丁文本包含这些路径，也不能视为当前二进制已
可用，更不能用 FP32 实验结果证明其性能。

## 9. 源码审计发现的风险

这些问题不否定已验证的 FP32 收益，但影响补丁推广和后续实验设计。

1. **INT8 packing 生命周期不完整。** 补丁把裸 `int8_t* ptr` 和 flag 直接加入
   `TensorBase`，未看到与 `TensorImpl/Storage` 绑定的所有权和释放路径；Tensor
   handle 复制后的状态语义也不清晰。`ipacking()` 还返回未定义 Tensor。
2. **INT8 缺少可靠 fallback。** 当 KML INT8 符号不存在或参数不满足 BLAS 条件
   时，新增 `igemm()` 没有通用实现兜底。当前构建恰好没有探测到这些符号。
3. **全局类型语义被放宽。** 补丁让 `Char` 绕过通用 `check_type`，并通过
   `const_cast` 将部分输出重绑为 Int，可能破坏 out-variant 和通用 Tensor API
   的类型约束，需专项单元测试。
4. **可能出现嵌套并行。** 一旦进入新增的 `bmm_impl_cpu_`，batch 外层就使用
   ATen 线程池；若换成内部也启用多线程的 KML 变体，可能过度订阅。部署时必须
   同时记录 ATen 线程数、BLAS 内部线程策略和 CPU affinity。
5. **重排不在并行 GEMM 循环内。** `einsum` 的 reshape/物化、BMM 输入 clone
   和结果恢复不会随 batch worker 等比例扩展，复杂 layout 仍可能受内存带宽限制。
6. **FP16 后端优先级改变。** 补丁移除原 `mkldnn_fp16_gemm` 优先路径，只有 KML
   的 H GEMM 与 H GEMV 符号同时存在才进入新路径；符号缺失时的性能需要重测。
7. **构建脚本删除 CMake cache 的非 ASCII 字节。** 这是构建兼容 workaround，
   不是运行时优化，且可能静默改变含非 ASCII 路径或配置值的内容。
8. **MPI FP16 依赖非标准扩展。** `MPIX_SHORT_FLOAT` 依赖特定 MPI 实现，当前
   构建并未启用 MPI。

## 10. 可以证明与不能证明的内容

| 结论 | 证据强度 | 判断 |
|---|---|---|
| 当前 FP32 `einsum` 会规约到 BMM | 源码确认 + profiler | 已证明 |
| FP32 BMM 最终链接并运行 KML `sgemm_` | 构建、ELF、运行时动态库审计 | 已证明 |
| 补丁按 batch 使用 ATen worker 并行 GEMM | 补丁源码 + batch/thread 阶梯扩展 | 已证明 |
| 优化栈改善 4D/5D/6D FP32 端到端性能 | 230/230 配对胜出 | 已证明 |
| 多 batch 是补丁最明显的多核收益场景 | B1/B4/B32 的 T1→T32 对照 | 已证明 |
| 补丁有效并行单次大缩并 | B1 T1→T32 仅 1.004x | 被实验否定 |
| 完整 PyTorch/KML 路径对任意 stride 零复制 | X4 在 BMM 前双输入完整物化 | 被实验否定 |
| 约 1.2x 的 B1/T1 收益全部来自 KML 微内核 | 构建变量不止 BLAS | 不能证明 |
| BF16/FP16/INT8 优化有效 | 当前只测 FP32，部分符号未探测到 | 不能证明 |
| profiler 可拆出 KML 内部 packing/kernel 时间 | PyTorch 只看到 ATen 包装层 | 不能证明 |

## 11. 归因限制

本对比不是严格的单变量补丁消融：

- 官方 wheel 使用 GCC 10.2、OpenBLAS、`USE_MKLDNN=ON`、C++11 ABI 0；
- KML 版本使用 BiSheng/clang 19.1.7、KML、`USE_MKLDNN=0`、C++11 ABI 1；
- 当前大规模实验还固定 OpenBLAS 内部线程为 1，其目的在于观察 PyTorch/BMM
  外层任务调度，不代表“最佳多线程 OpenBLAS”结果；
- 两个版本都测完整 `torch.einsum` 端到端时间，输出分配和必要的数据转换均包含
  在内，但 KML 内部 packing 和 kernel 无独立计时；
- 早期固定体量实验的约 1.985x 与大规模 B1/T1 的约 1.206x 面向不同尺寸分布，
  不能互相替代，也不构成实验矛盾。

因此，B1/T1 的改进应归因于“鲲鹏 PyTorch/KML 构建栈”；只有 batch 随线程数
呈现的阶梯扩展，才能结合源码较强地归因于 `bmm_impl_cpu_` 的外层并行改动。

## 12. 最终评价与后续优先级

该补丁对当前 FP32 张量缩并的有效优化不是一个专用张量缩并 kernel，而是两层
组合：底层使用 KML GEMM，上层在 BMM 的 batch 维并行调度独立 GEMM。它对
多 batch 工作负载很有效：B32/T32 相对官方 OpenBLAS wheel 的中位加速达到
18.951x，8 个具体配置为 13.759x–31.152x；对单 batch 则主要体现后端差异，
中位收益约 1.206x，且增加线程几乎不能加速单次缩并。

后续工作建议按以下顺序进行：

1. 构建同编译器、同 ABI、同 oneDNN 选项，仅替换 OpenBLAS/KML 的严格消融版本，
   分离 KML 内核与构建工具链的贡献。
2. 增加 `batch=1` 的单 GEMM 内部多线程方案，与当前 batch 外层方案做二维调度，
   并防止 KML 内外层过度订阅。
3. 对 `einsum` 的 `permute/reshape` 和 BMM clone 建立可观测阶段计时，优先减少
   X4 这类双输入整张量物化，而不是只继续优化 GEMM。
4. 在宣称低精度优化前，分别补齐 BF16、FP16、INT8 的符号审计、正确性、内存
   生命周期、fallback 和性能测试。
5. 将危险的 TensorBase 裸指针、全局 Char 类型放宽和 cache 字节删除从生产补丁
   中拆出，改为有所有权、范围受限且有单元测试的实现。

## 13. 证据索引

- 高阶缩并汇总：
  `results/contraction_cpu0-3_20261009T163753742611Z/summary.tsv`
- 高阶缩并原始样本：
  `results/contraction_cpu0-3_20261009T163753742611Z/raw.tsv`
- 高阶缩并实验元数据：
  `results/contraction_cpu0-3_20261009T163753742611Z/provenance.json`
- 32 vCPU 大规模汇总：
  `results/tblis_batch_cpu0-31_20261011T074035858907Z/summary.tsv`
- 32 vCPU 原始样本：
  `results/tblis_batch_cpu0-31_20261011T074035858907Z/raw.tsv`
- 布局物化诊断：
  `results/tblis_batch_cpu0-31_20261011T074035858907Z/materialization_diagnostics.tsv`
- 线程诊断：
  `results/tblis_batch_cpu0-31_20261011T074035858907Z/thread_diagnostics.tsv`
- 当前大规模分析：
  `results/tblis_batch_cpu0-31_20261011T074035858907Z/analysis.md`
- KPEX 补丁：
  `/root/benchmark/pytorch-kunpeng/kpex-current/thirdparty/pytorch-v2.5.0.patch`
- patched PyTorch `einsum` 路径：
  `/root/benchmark/pytorch-kunpeng/pytorch-v2.5.0/aten/src/ATen/native/Linear.cpp`
- patched PyTorch BMM 路径：
  `/root/benchmark/pytorch-kunpeng/pytorch-v2.5.0/aten/src/ATen/native/LinearAlgebra.cpp`
- patched PyTorch BLAS 路径：
  `/root/benchmark/pytorch-kunpeng/pytorch-v2.5.0/aten/src/ATen/native/CPUBlas.cpp`
