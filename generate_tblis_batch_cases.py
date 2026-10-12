#!/usr/bin/env python3
"""Generate batched X0/X4 scaling cases for PyTorch and TBLIS."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCALES = {
    "S": (16, 16, 16, 16, 8, 8),
    "M": (16, 32, 16, 32, 8, 16),
    "L": (32, 32, 32, 32, 16, 16),
    "XL": (32, 64, 32, 64, 16, 32),
}
LAYOUTS = {
    "X0": ("abkl", "klcd"),
    "X4": ("akbl", "kcld"),
}
BATCHES = (1, 4, 32)
THREADS = (1, 2, 4, 8, 32)
FIELDS = [
    "case_id", "scale", "layout", "batch", "threads", "equation",
    "a_physical_order", "b_physical_order", "a", "b", "c", "d", "k", "l",
    "a_shape", "b_shape", "c_shape", "M", "N", "K", "input_elements",
    "output_elements", "storage_elements", "minimum_bytes", "storage_bytes", "flops",
]


def build_row(scale: str, layout: str, batch: int, threads: int) -> dict[str, str]:
    a, b, c, d, k, l = SCALES[scale]
    a_order, b_order = LAYOUTS[layout]
    a_elements = batch * a * b * k * l
    b_elements = batch * k * l * c * d
    output_elements = batch * a * b * c * d
    m, n, reduction = a * b, c * d, k * l
    return {
        "case_id": f"{scale}_{layout}_B{batch}_T{threads}",
        "scale": scale,
        "layout": layout,
        "batch": str(batch),
        "threads": str(threads),
        "equation": "pabkl,pklcd->pabcd",
        "a_physical_order": a_order,
        "b_physical_order": b_order,
        "a": str(a), "b": str(b), "c": str(c), "d": str(d),
        "k": str(k), "l": str(l),
        "a_shape": f"{a},{b},{k},{l}",
        "b_shape": f"{k},{l},{c},{d}",
        "c_shape": f"{a},{b},{c},{d}",
        "M": str(m), "N": str(n), "K": str(reduction),
        "input_elements": str(a_elements + b_elements),
        "output_elements": str(output_elements),
        "storage_elements": str(a_elements + b_elements + output_elements),
        "minimum_bytes": str(4 * (a_elements + b_elements + output_elements)),
        "storage_bytes": str(4 * (a_elements + b_elements + output_elements)),
        "flops": str(batch * 2 * m * n * reduction),
    }


def generate() -> list[dict[str, str]]:
    return [
        build_row(scale, layout, batch, threads)
        for scale in SCALES
        for layout in LAYOUTS
        for batch in BATCHES
        for threads in THREADS
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, delimiter="\t", fieldnames=FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    assert len(rows) == len(SCALES) * len(LAYOUTS) * len(BATCHES) * len(THREADS)
    assert {row["scale"] for row in rows} == set(SCALES)
    assert {int(row["batch"]) for row in rows} == set(BATCHES)
    assert {int(row["threads"]) for row in rows} == set(THREADS)
    assert all(row["equation"] == "pabkl,pklcd->pabcd" for row in rows)
    print(f"wrote {len(rows)} cases to {args.output}")


if __name__ == "__main__":
    main()
