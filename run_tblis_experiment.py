#!/usr/bin/env python3
"""Run the fixed-equation layout/materialization and scaling experiments."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "tblis_cases.tsv"
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
    if case["scale"] == "S" and case["threads"] == "1" and "layout" in case["families"]:
        return True
    return (
        case["scale"] in {"L", "XL"}
        and case["layout"] in {"X0", "X4"}
        and int(case["threads"]) in {1, 4}
    )


def expected_paths(output: Path, case: dict[str, str], method: str,
                   campaign: int, trace: bool) -> list[Path]:
    stem = f"{case['case_id']}_c{campaign}"
    paths = [
        output / "raw" / method / f"{stem}.tsv",
        output / "metadata" / method / f"{stem}.json",
    ]
    if campaign == 0:
        paths.extend([
            output / "materialization" / method / f"{stem}.tsv",
            output / "thread_diagnostics" / method / f"{stem}.tsv",
        ])
        if trace and method.startswith("pytorch_"):
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
    common = [
        "--manifest", str(MANIFEST), "--case", case["case_id"],
        "--campaign", str(campaign), "--output-dir", str(output),
        "--reference-dir", str(output / "references"),
    ]
    if method == "tblis_2.0_armsve":
        command = [str(ROOT / "bin/tblis_layout_runner"), *common]
    else:
        python = PYTORCH_ROOT / (
            "env-native/bin/python" if method == "pytorch_2.5.0_openblas"
            else "env-kml/bin/python"
        )
        if method == "pytorch_2.5.0_kml_patched":
            environment["LD_LIBRARY_PATH"] = kml_library_path()
        command = [
            str(python), str(ROOT / "benchmark_tblis_pytorch.py"),
            *common, "--method", method,
        ]
    if campaign == 0:
        command.append("--diagnostics")
    if trace and method.startswith("pytorch_"):
        command.append("--export-trace")
    return command, environment


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--campaigns", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case", action="append", help="run only the named case; repeatable")
    parser.add_argument("--method", action="append", choices=METHODS)
    args = parser.parse_args()
    if args.campaigns < 1:
        raise SystemExit("campaigns must be positive")
    output = args.output_dir or ROOT / "results" / f"tblis_layout_cpu0-3_{utc_stamp()}"
    if output.exists() and not (output / "status.txt").exists():
        raise SystemExit(f"existing output lacks status.txt: {output}")
    output.mkdir(parents=True, exist_ok=True)
    status = output / "status.txt"
    status.write_text("RUNNING\n", encoding="utf-8")
    shutil.copy2(TBLIS_ROOT / "install/build_metadata.txt", output / "tblis_build_metadata.txt")

    cases = read_manifest(MANIFEST)
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["case_id"] in wanted]
        missing = wanted - {case["case_id"] for case in cases}
        if missing:
            raise SystemExit(f"unknown cases: {sorted(missing)}")
    with (output / "case_manifest.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    methods = args.method or METHODS
    reference_scales = sorted({case["scale"] for case in cases})
    reference_command = [
        str(PYTORCH_ROOT / "env-native/bin/python"),
        str(ROOT / "prepare_tblis_references.py"),
        "--manifest", str(MANIFEST), "--output-dir", str(output / "references"),
    ]
    for scale in reference_scales:
        reference_command.extend(["--scale", scale])
    subprocess.run(reference_command, cwd=ROOT, check=True)

    provenance = {
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": str(MANIFEST.resolve()), "methods": methods,
        "campaigns": args.campaigns, "warmup": 10, "repeat_per_campaign": 50,
        "equation": "abkl,klcd->abcd", "dtype": "float32",
        "reference": "shared FP64 torch.mm by logical scale",
        "host_affinity": sorted(os.sched_getaffinity(0)),
        "thread_policy": {"1": [3], "2": [2, 3], "4": [0, 1, 2, 3], "8": "unsupported"},
        "hardware_dram_bandwidth": "N/A: no DDR/uncore PMU exposed to this ECS",
        "estimated_bandwidth": "minimum logical bytes divided by end-to-end time",
        "pytorch_runner": str((ROOT / "benchmark_tblis_pytorch.py").resolve()),
        "tblis_runner": str((ROOT / "bin/tblis_layout_runner").resolve()),
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    tasks = [
        (case, method, campaign)
        for campaign in range(args.campaigns)
        for case in cases
        for method in methods
    ]
    random.Random(20261010).shuffle(tasks)
    if args.limit is not None:
        tasks = tasks[:args.limit]
    log_path = output / "run.log"
    completed = 0
    skipped = 0
    try:
        with log_path.open("a", encoding="utf-8") as log:
            for index, (case, method, campaign) in enumerate(tasks, 1):
                trace = should_trace(case, campaign)
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
                if completed % 5 == 0 or index == len(tasks):
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
