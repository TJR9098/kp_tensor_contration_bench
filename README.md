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
