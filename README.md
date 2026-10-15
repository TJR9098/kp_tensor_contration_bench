# Tensor contraction redesign benchmark

This experiment compares the following implementations on the same generated
4D, 5D, and 6D FP32 contractions:

- `pytorch_2.5.0_openblas`
- `pytorch_2.5.0_kml_patched`
- `tblis_2.0_armsve`

The primary size constraint is `batch * M * N * K == 2^20`.  PyTorch phase
columns remain `NA`; its operator-level profiler measurements are written to a
separate diagnostic table.  TBLIS phase measurements come from the local
profiling ABI and represent critical-path estimates for multithreaded packing
and kernel work.

Generated results are immutable timestamped directories under `results/`.

## Environment

The batched runner defaults match the Kunpeng 920 ECS used for the published
runs and can be overridden as follows:

```bash
export PYTORCH_KUNPENG_ROOT=/root/benchmark/pytorch-kunpeng
export TBLIS_BENCH_ROOT=/root/benchmark/exo_tensor_bench_openeuler2203_arm64_20261008_v2
export TBLIS_PREFIX="$TBLIS_BENCH_ROOT/install/tblis-2.0-armsve"
```

`PYTORCH_KUNPENG_ROOT` must contain `env-native` (official PyTorch 2.5.0
OpenBLAS) and `env-kml` (the BiSheng-built KPEX/KML-patched PyTorch 2.5.0).
`TBLIS_PREFIX` must contain the ARM-optimized TBLIS 2.0 headers and shared
library.  A C++17 compiler, Python 3, GNU Make, OpenSSL development headers,
and `lscpu` are also required.

`run_tblis_batch_experiment.py` honors `PYTORCH_KUNPENG_ROOT`,
`TBLIS_BENCH_ROOT`, and `TBLIS_PREFIX`; the Makefile honors `TBLIS_PREFIX`.
The full batched experiment requires CPUs 0-31 in the process affinity mask.
The runner fails before writing samples if a dependency or CPU is unavailable.
Override the three variables above when the installations live elsewhere.

Run the full experiment and generate the report with:

```bash
make all
python3 run_experiment.py --campaigns 3
python3 summarize_results.py results/<run-id>
```

The final directory contains `suites.MD`, the primary `summary.tsv`, the
dedicated `pytorch_diagnostics_summary.tsv`, raw samples, Chrome traces, the
representative `timeline.svg`, and validation evidence in `validation.log`.

## Fixed-equation TBLIS layout and scaling experiment

The supplemental experiment on branch `tblis` keeps
`abkl,klcd->abcd` and the logical values fixed while changing only the backing
storage order and strides.  It contains:

- X0-X5 layout/materialization cases at `M=N=256, K=64`;
- batch-one S/M/L/XL contractions for X0 and X4 at T1/T2/T4;
- PyTorch profiler plus storage-identity materialization audits;
- TBLIS 2.0 phase profiling and wrapper pointer audits;
- actual worker/affinity sampling, empirical end-to-end CDFs, and timelines.

Generate and run it with:

```bash
make tblis-cases bin/tblis_layout_runner
python3 run_tblis_experiment.py --campaigns 3
python3 summarize_tblis_experiment.py results/<tblis-layout-run-id>
```

The report is `suites.MD`.  `cdf_layout_overview.svg` and
`cdf_scaling_overview.svg` summarize the empirical CDFs; `cdf/` contains one
three-method CDF per case, and `cdf_points.tsv` preserves every plotted point.
PyTorch profiler and dispatch diagnostics are never mixed into the performance
samples.  Hardware DDR bandwidth is reported as unavailable when the host does
not expose a DDR/uncore PMU; the separately named minimum effective bandwidth is
an algorithmic byte-count estimate.

## Batched TBLIS scaling experiment

The batched extension keeps the S/M/L/XL dimensions and X0/X4 layouts, adds
independent batches B1/B4/B32, and tests T1/T2/T4/T8/T32.  Alongside both PyTorch
builds it reports two TBLIS schedules:

- `tblis_2.0_armsve_intra`: batches are serial and each contraction uses T
  internal TBLIS threads;
- `tblis_2.0_armsve_inter`: a persistent pool runs up to T batches in parallel
  while each contraction uses one TBLIS thread.

Start with a one-case smoke test:

```bash
git checkout tblis
make tblis-batch-cases bin/tblis_layout_runner
python3 run_tblis_batch_experiment.py \
  --output-dir results/smoke_kml \
  --campaigns 1 \
  --case S_X0_B1_T1 \
  --method pytorch_2.5.0_kml_patched
```

This smoke test verifies the patched PyTorch/KML path only.  The summarizer is
intentionally strict and accepts only a complete 120-configuration, four-method
run.

Run all 120 configurations, four methods, and three campaigns with:

```bash
RUN_DIR=results/tblis_batch_cpu0-31_manual
python3 run_tblis_batch_experiment.py \
  --output-dir "$RUN_DIR" \
  --campaigns 3
python3 summarize_tblis_batch_experiment.py "$RUN_DIR"
python3 analyze_tblis_batch_experiment.py "$RUN_DIR"
```

The runner is resumable: rerunning the same command skips complete tasks and
rejects partial task output.  A complete run ends with `COMPLETE` in
`status.txt` after summarization.  To compare the common T1/T2/T4 cases against
an earlier 4-vCPU run, use:

```bash
python3 analyze_tblis_batch_experiment.py "$RUN_DIR" \
  --comparison-summary results/<4-vcpu-run-id>/summary.tsv
```

The batch report distinguishes whole-batch latency, average latency per
contraction, strong-scaling speedup, and contraction throughput relative to
B1.  TBLIS inter-batch phase columns are deliberately `N/A`: concurrent calls
cannot be attributed safely with the process-global profiling ABI.
`suites.MD` is the full result table, `analysis.md` gives the detailed
interpretation, and
`cdf_batch_all_overview.svg` consolidates all batch/thread CDFs into one
logarithmic-axis figure while retaining the per-case linear-axis plots.
`validation.log` must contain only `PASS`/`INFO` records before results are
used.  Raw result directories are intentionally ignored by Git because a full
run contains tens of thousands of samples and profiler traces; archive them
separately when they must be retained.

## KML SVE multi compatibility and thread probe

`probe_kml_multi.py` loads the installed SVE `locking` and `multi` libraries in
separate processes. It checks FP32 CBLAS GEMM and the Fortran `sgemm_` ABI against
an independent NumPy FP64 reference, including all output elements, transpose,
padding, and nonzero alpha/beta cases. Optional framework workers run the existing
patched PyTorch with B1 X0/X4 inputs and verify the actual loaded KML path.

The coordinator uses Python's standard library. Workers use
`<project>/env-kml/bin/python`, where `--project` defaults to
`/root/benchmark/pytorch-kunpeng`; that environment must have NumPy installed.
The existing BiSheng OpenMP runtime and OMPT header are used; `g++` builds a small
diagnostic observer. OMPT runs in separate processes from performance samples.

```bash
python3 probe_kml_multi.py \
  --project /root/benchmark/pytorch-kunpeng \
  --engines cblas --scales S M L XL \
  --thread-list 1 2 4 8 32 --audit-imports

python3 probe_kml_multi.py \
  --engines torch --scales XL \
  --thread-list 1 2 4 8 32
```

Defaults are 3 campaigns, 10 warmups and 50 measured calls per campaign. Each
invocation creates a new timestamped directory with `report.md`, `raw.tsv`,
`summary.tsv`, thread diagnostics, library hashes, per-process metadata and logs.
The selected KML is loaded before PyTorch without modifying the installed
environment or the historical runner's library selection.

The probe records available CPUs and quota before pinning. Requests beyond that
capacity are explicitly marked as compatibility diagnostics; they are not valid
multi-core scaling results. Requested threads, KML-reported threads and OMPT team
size are separate fields. Direct GEMM timings include the Python ctypes call and
the full BLAS operation, not just a microkernel. Framework outputs are allocated
per call, whereas CBLAS outputs are reused.

Formal workers set `OMP_TOOL=disabled`, `OMP_WAIT_POLICY=PASSIVE`, and
`KMP_BLOCKTIME=0`. Diagnostic workers explicitly load the supplied OMPT observer.
`--audit-imports` compares default and explicit OpenMP tool settings, preserving
any import failures in isolated subprocess logs. These settings and the changed
input generator mean the new timings must not be presented as a direct rerun of
the historical 32-vCPU results. See `kml_multi_validation.md` for the recorded
compatibility findings and resource limitations.

## Complete KML multi rerun on 32 vCPUs

`run_kml_multi_batch.py` repeats the historical 120-case batch matrix for
`pytorch_2.5.0_kml_multi` only. It reuses the original timing runner, input values,
FP64 references, output lifetime, affinity and 3x50 timing protocol. It does not
change the official OpenBLAS wheel, its process environment, or the historical
locking library selection. OpenBLAS configuration and mapped library hashes are
checked before and after the run; historical comparison rows are labelled `old`.

The policy is explicit: B1 uses KML inner=T; B4/B32 keep ATen batch parallelism
with KML inner=1. It is not a shape-tuned or hybrid scheduler. Separate diagnostics
intercept the sgemm_ boundary and use OMPT to distinguish KML and PyTorch teams,
verify batch call counts and reject nested expansion. Distinct-batch random inputs
add correctness coverage without changing the historical performance inputs.

```bash
python3 run_kml_multi_batch.py --preflight-only
python3 run_kml_multi_batch.py
# Resume using the directory printed by the new run:
python3 run_kml_multi_batch.py --output-dir results/<kml-multi-run-id>
python3 summarize_kml_multi_batch.py results/<kml-multi-run-id>
```

CPU0-31 and the historical reference directory are required. New output contains
18,000 samples in `raw.tsv`, 120 new groups in `summary.tsv`, and an explicitly
sourced 600-row `combined_summary.tsv` / `suites.MD` with historical comparisons.
`analysis.md`, CDFs, `timeline_batch.svg`, Chrome traces,
`pytorch_diagnostics_summary.tsv`, `kml_thread_diagnostics.tsv`, and
`validation.log` document results and actual thread roles. KML internal phase
columns remain N/A; estimated byte rates are not hardware DDR bandwidth.

New formal workers set `OMP_TOOL=disabled`, `OMP_DYNAMIC=FALSE`, and
`OMP_MAX_ACTIVE_LEVELS=1`. Unlike the compatibility probe, they introduce no
passive wait policy and retain the historical timing loop. These KML-only
controls and the changed library define the new method. `DATA_COMPLETE` means
all tasks finished; `COMPLETE` additionally means reporting validation passed.

## Targeted XL B4/T32 timing recheck

```bash
python3 recheck_kml_b4_t32.py
```

This repeats only `XL_X0_B4_T32` and `XL_X4_B4_T32` for locking and multi,
with each variant's previous environment and the unchanged timing/input code.
It collects 3 campaigns of 10 warmups plus 50 measurements for every case/variant,
and runs separate BLAS/OMPT diagnostics to verify four GEMM callers and one
internal BLAS thread per call. Results are saved to a new timestamped directory
with `recheck.md`, `raw.tsv`, `summary.tsv`, `campaigns.tsv`, and `cdf.svg`.
This checks reproducibility of the existing outer-batch policy; it does not mix
in a different inner-thread or hybrid policy or overwrite earlier reports.

## Explicit KML multi inter and intra schedules

```bash
python3 build_kml_schedules.py
python3 run_kml_schedules.py --preflight-only
python3 run_kml_schedules.py --output-dir results/<new-schedule-run-id>
python3 summarize_kml_schedules.py results/<new-schedule-run-id>
```

`build_kml_schedules.py` applies `kml_bmm_schedule.patch` to a generated copy of
the existing patched PyTorch `LinearAlgebra.cpp`, recompiles that translation
unit with its original BiSheng build flags, and relinks an isolated
`bin/kml_schedules/libtorch_cpu.so` using the existing object files. It does not
edit the installed PyTorch sources or replace its installed shared library.
The build requires the retained `pytorch-v2.5.0/build` object files and `ninja`.

Both methods load the same isolated library and KML SVE multi binary:

- `kml-multi-inter`: ATen parallelizes batches; each GEMM uses one KML thread,
  including B1.
- `kml-multi-intra`: the C++ BMM loop processes batches serially; each GEMM uses
  T KML threads, including B4/B32.

The process-local `KML_BMM_SCHEDULE` flag selects the BMM branch. No Python
batch loop or extra result stacking is introduced. Framework transformations
retain T threads for both methods. Formal timing uses the unchanged historical
full-einsum runner, inputs, references, affinities and 3x50 protocol, for 36,000
new samples. The runner preserves provenance and can resume only when source
and library hashes match. Independent distinct-batch tests and BLAS/OMPT
diagnostics check both strategies before and alongside formal measurements.

The report contains 240 new summary groups and 720 combined groups with explicitly
historical OpenBLAS, locking KML and TBLIS rows. It includes same-strategy and
best-strategy comparisons, P95, CDFs, profiler timelines and runtime thread
evidence. The original PyTorch library and OpenBLAS binaries are audited for
unchanged hashes; they are not rebuilt or silently replaced by this experiment.

The completed 2026-10-14 inter/intra run is under
`results/kml_inter_intra_cpu0-31_20261014T124020530797Z/`:
`suites.MD` contains the full table and profiler diagnostics; `analysis.md`
compares strategies and tails; `same_strategy_comparison.tsv` compares KML with
historical TBLIS using the same schedule. `cdf_kml_only_overview.svg` shows only
the two new KML methods, while `cdf_overview.svg` includes historical baselines.
