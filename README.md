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
