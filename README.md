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
