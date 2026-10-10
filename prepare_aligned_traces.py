#!/usr/bin/env python3
"""Create timestamp-aligned Chrome traces for side-by-side Perfetto comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


METHODS = (
    "pytorch_2.5.0_openblas",
    "pytorch_2.5.0_kml_patched",
)
CASES = (
    "R4_Q2_G0_D0_L0_B1_T1",
    "R4_Q2_G0_D0_L3_B1_T1",
    "R4_Q2_G0_D0_L4_B1_T1",
    "R4_Q2_G0_D0_L5_B1_T1",
    "R6_Q3_G3_D0_L0_B1_T1",
    "R6_Q3_BATCH_B64_T4",
)
TARGET_ANCHOR_US = 1000.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trace_events(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, list):
        events = document
    elif isinstance(document, dict) and isinstance(document.get("traceEvents"), list):
        events = document["traceEvents"]
    else:
        raise ValueError("unsupported Chrome trace structure")
    if not all(isinstance(event, dict) for event in events):
        raise ValueError("traceEvents must contain JSON objects")
    return events


def normalize_trace(source: Path, destination: Path, method: str, case_id: str) -> dict[str, Any]:
    document = json.loads(source.read_text(encoding="utf-8"))
    events = trace_events(document)
    anchors = [
        event for event in events
        if event.get("name") == "benchmark::operation"
        and isinstance(event.get("ts"), (int, float))
    ]
    if len(anchors) != 1:
        raise ValueError(f"{source}: expected one benchmark::operation, got {len(anchors)}")

    original_anchor = float(anchors[0]["ts"])
    operation_duration = anchors[0].get("dur")
    if not isinstance(operation_duration, (int, float)):
        raise ValueError(f"{source}: benchmark::operation has no numeric duration")
    operation_end = original_anchor + float(operation_duration)
    filtered_events: list[dict[str, Any]] = []
    dropped_outside_operation = 0
    for event in events:
        timestamp = event.get("ts")
        if (
            isinstance(timestamp, (int, float))
            and not original_anchor <= float(timestamp) <= operation_end
        ):
            dropped_outside_operation += 1
            continue
        filtered_events.append(event)

    adjusted = 0
    earliest = TARGET_ANCHOR_US
    pid_prefix = f"{method}::{case_id}"
    for event in filtered_events:
        timestamp = event.get("ts")
        if isinstance(timestamp, (int, float)):
            normalized = round(TARGET_ANCHOR_US + float(timestamp) - original_anchor, 6)
            event["ts"] = normalized
            earliest = min(earliest, normalized)
            adjusted += 1
        if "pid" in event:
            event["pid"] = f"{pid_prefix}::{event['pid']}"

    if earliest < 0:
        raise ValueError(
            f"{source}: normalized trace starts at {earliest} us; increase TARGET_ANCHOR_US"
        )
    if anchors[0]["ts"] != TARGET_ANCHOR_US:
        raise RuntimeError(f"{source}: benchmark anchor did not normalize exactly")

    if isinstance(document, list):
        document = {
            "schemaVersion": 1,
            "traceEvents": filtered_events,
            "displayTimeUnit": "ms",
        }
    else:
        document["traceEvents"] = filtered_events
        document.pop("baseTimeNanoseconds", None)
        document["traceName"] = destination.name

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {
        "case_id": case_id,
        "method": method,
        "source": str(source),
        "source_sha256": sha256(source),
        "original_anchor_us": f"{original_anchor:.6f}",
        "normalized_anchor_us": f"{TARGET_ANCHOR_US:.6f}",
        "applied_shift_us": f"{TARGET_ANCHOR_US - original_anchor:.6f}",
        "earliest_event_us": f"{earliest:.6f}",
        "adjusted_event_count": adjusted,
        "dropped_outside_operation": dropped_outside_operation,
        "output": destination.name,
        "output_sha256": sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_root / "traces"
    rows: list[dict[str, Any]] = []
    for case_id in CASES:
        for method in METHODS:
            source = args.result_dir / "traces" / method / f"{case_id}_c0.json"
            destination = output_dir / f"{method}__{case_id}_c0.json"
            rows.append(normalize_trace(source, destination, method, case_id))

    manifest = output_dir / "alignment_manifest.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as archive:
        archive.add(output_dir, arcname="traces")

    print(json.dumps({
        "trace_count": len(rows),
        "target_anchor_us": TARGET_ANCHOR_US,
        "output_dir": str(output_dir),
        "archive": str(args.archive),
        "archive_sha256": sha256(args.archive),
    }, indent=2))


if __name__ == "__main__":
    main()
