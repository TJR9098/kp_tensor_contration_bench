#!/usr/bin/env python3
"""Run the full three-method experiment with resumable per-group outputs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTORCH_ROOT = Path("/root/benchmark/pytorch-kunpeng")
TBLIS_ROOT = Path("/root/benchmark/exo_tensor_bench_openeuler2203_arm64_20261008_v2")
METHODS = [
    "pytorch_2.5.0_openblas",
    "pytorch_2.5.0_kml_patched",
    "tblis_2.0_armsve",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def kml_library_path() -> str:
    project = PYTORCH_ROOT
    paths = [
        project / "env-kml/lib/python3.11/site-packages/torch/lib",
        project / "hpckit-components/BiShengCompiler-5.2.0-aarch64-linux-mini-hpc/lib",
        project / "hpckit-components/BiShengCompiler-5.2.0-aarch64-linux-mini-hpc/lib/aarch64-unknown-linux-gnu",
        project / "hpckit-layout/kml/bisheng/lib",
        project / "hpckit-layout/kml/bisheng/lib/sve/kblas/locking",
        project / "hpckit-layout/kml/bisheng/lib/sve512",
        project / "hpckit-layout/kml/bisheng/lib/noarch",
        project / "hpckit-components/KuTACC_26.1.RC1_Bisheng/lib",
        project / "hpckit-components/KuDNN_26.1.RC1_Bisheng/lib",
        project / "hpckit-components/KUPL_26.1.RC1_Bisheng/release/lib",
        project / "hpckit-components/Hyper-MPI_26.1.RC1_aarch64_openEuler22.03_Bisheng_hns-release/release/hmpi/lib",
    ]
    existing = os.environ.get("LD_LIBRARY_PATH")
    if existing:
        paths.append(Path(existing))
    return ":".join(map(str, paths))


def should_trace(case: dict[str, str], campaign: int) -> bool:
    if campaign != 0:
        return False
    rank = int(case["payload_rank"])
    middle_q = {4: 2, 5: 2, 6: 3}[rank]
    if int(case["contraction_dims"]) != middle_q:
        return False
    if case["families"] == "batch":
        return int(case["batch"]) == 64 and int(case["threads"]) == 4
    return (
        case["distribution"] == "D0"
        and case["layout"] in {"L0", "L3", "L4", "L5"}
        and case["shape_profile"] in {"G0", "G3"}
    )


def expected_paths(output: Path, case: dict[str, str], method: str,
                   campaign: int, trace: bool) -> list[Path]:
    stem = f"{case['case_id']}_c{campaign}"
    paths = [
        output / "raw" / method / f"{stem}.tsv",
        output / "metadata" / method / f"{stem}.json",
    ]
    if method.startswith("pytorch_"):
        paths.append(output / "diagnostics" / method / f"{stem}.tsv")
        if trace:
            paths.append(output / "traces" / method / f"{stem}.json")
    return paths


def command_for(case: dict[str, str], method: str, campaign: int,
                output: Path, trace: bool) -> tuple[list[str], dict[str, str]]:
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "GOTO_NUM_THREADS": "1",
    })
    if method == "tblis_2.0_armsve":
        return ([
            str(ROOT / "bin/tblis_contraction_runner"),
            "--manifest", str(ROOT / "cases.tsv"),
            "--case", case["case_id"], "--campaign", str(campaign),
            "--output-dir", str(output),
        ], environment)
    python = PYTORCH_ROOT / (
        "env-native/bin/python" if method == "pytorch_2.5.0_openblas"
        else "env-kml/bin/python"
    )
    if method == "pytorch_2.5.0_kml_patched":
        environment["LD_LIBRARY_PATH"] = kml_library_path()
    command = [
        str(python), str(ROOT / "benchmark_pytorch.py"),
        "--manifest", str(ROOT / "cases.tsv"), "--case", case["case_id"],
        "--method", method, "--campaign", str(campaign),
        "--output-dir", str(output),
    ]
    if trace:
        command.append("--export-trace")
    return command, environment


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--campaigns", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.campaigns < 1:
        raise SystemExit("campaigns must be positive")
    output = args.output_dir or ROOT / "results" / f"contraction_cpu0-3_{utc_stamp()}"
    if output.exists() and not (output / "status.txt").exists():
        raise SystemExit(f"existing output lacks status.txt: {output}")
    output.mkdir(parents=True, exist_ok=True)
    status = output / "status.txt"
    status.write_text("RUNNING\n", encoding="utf-8")
    shutil.copy2(ROOT / "cases.tsv", output / "case_manifest.tsv")
    shutil.copy2(TBLIS_ROOT / "install/build_metadata.txt", output / "tblis_build_metadata.txt")
    provenance = {
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": str((ROOT / "cases.tsv").resolve()),
        "methods": METHODS,
        "campaigns": args.campaigns,
        "warmup": 10,
        "repeat_per_campaign": 50,
        "logical_volume": "B*M*N*K=2^20",
        "host_affinity": sorted(os.sched_getaffinity(0)),
        "pytorch_runner": str((ROOT / "benchmark_pytorch.py").resolve()),
        "tblis_runner": str((ROOT / "bin/tblis_contraction_runner").resolve()),
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    cases = read_manifest(ROOT / "cases.tsv")
    tasks = [
        (case, method, campaign)
        for campaign in range(args.campaigns)
        for case in cases
        for method in METHODS
    ]
    random.Random(20261009).shuffle(tasks)
    if args.limit is not None:
        tasks = tasks[:args.limit]
    log_path = output / "run.log"
    completed = 0
    skipped = 0
    try:
        with log_path.open("a", encoding="utf-8") as log:
            for index, (case, method, campaign) in enumerate(tasks, 1):
                trace = should_trace(case, campaign) and method.startswith("pytorch_")
                paths = expected_paths(output, case, method, campaign, trace)
                present = [path.exists() for path in paths]
                if all(present):
                    skipped += 1
                    continue
                if any(present):
                    raise RuntimeError(
                        f"partial output for {case['case_id']} {method} campaign {campaign}: {paths}"
                    )
                command, environment = command_for(case, method, campaign, output, trace)
                log.write(json.dumps({
                    "event": "start", "index": index, "total": len(tasks),
                    "case": case["case_id"], "method": method,
                    "campaign": campaign, "trace": trace,
                }) + "\n")
                log.flush()
                process = subprocess.run(
                    command, cwd=ROOT, env=environment, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )
                log.write(process.stdout)
                log.write(json.dumps({
                    "event": "finish", "returncode": process.returncode,
                    "case": case["case_id"], "method": method,
                    "campaign": campaign,
                }) + "\n")
                log.flush()
                if process.returncode:
                    raise RuntimeError(
                        f"runner failed for {case['case_id']} {method} campaign {campaign}; see {log_path}"
                    )
                if not all(path.exists() for path in paths):
                    raise RuntimeError(f"runner did not produce all expected files: {paths}")
                completed += 1
                if completed % 10 == 0 or index == len(tasks):
                    print(json.dumps({
                        "completed_this_run": completed, "skipped": skipped,
                        "task_index": index, "task_total": len(tasks),
                        "output": str(output),
                    }), flush=True)
        if args.limit is None:
            provenance["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            (output / "provenance.json").write_text(
                json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
            )
            status.write_text("DATA_COMPLETE\n", encoding="utf-8")
        else:
            status.write_text("PARTIAL_LIMITED_RUN\n", encoding="utf-8")
        print(output)
    except Exception:
        status.write_text("FAILED_RESUMABLE\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
