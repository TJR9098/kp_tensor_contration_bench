#!/usr/bin/env python3
"""Generate the fixed-equation X-layout and single-contraction scaling cases."""

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
    "X0": ("abkl", "klcd", "none"),
    "X1": ("klab", "klcd", "none"),
    "X2": ("akbl", "klcd", "none"),
    "X3": ("abkl", "kcld", "none"),
    "X4": ("akbl", "kcld", "none"),
    "X5": ("akbl", "kcld", "A:k:2"),
}
FIELDS = [
    "case_id", "families", "scale", "layout", "threads", "equation",
    "a_physical_order", "b_physical_order", "padding", "a", "b", "c",
    "d", "k", "l", "a_shape", "b_shape", "c_shape", "M", "N", "K",
    "input_elements", "output_elements", "storage_elements", "minimum_bytes",
    "storage_bytes", "flops",
]


def product(values: tuple[int, ...] | list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def build_row(scale: str, layout: str, threads: int, families: str) -> dict[str, str]:
    a, b, c, d, k, l = SCALES[scale]
    sizes = {"a": a, "b": b, "c": c, "d": d, "k": k, "l": l}
    a_order, b_order, padding = LAYOUTS[layout]
    a_elements = a * b * k * l
    b_elements = k * l * c * d
    output_elements = a * b * c * d
    input_elements = a_elements + b_elements
    storage_elements = input_elements + output_elements
    if padding != "none":
        storage_elements += a_elements
    m, n, reduction = a * b, c * d, k * l
    return {
        "case_id": f"{scale}_{layout}_T{threads}",
        "families": families,
        "scale": scale,
        "layout": layout,
        "threads": str(threads),
        "equation": "abkl,klcd->abcd",
        "a_physical_order": a_order,
        "b_physical_order": b_order,
        "padding": padding,
        "a": str(a), "b": str(b), "c": str(c), "d": str(d),
        "k": str(k), "l": str(l),
        "a_shape": f"{a},{b},{k},{l}",
        "b_shape": f"{k},{l},{c},{d}",
        "c_shape": f"{a},{b},{c},{d}",
        "M": str(m), "N": str(n), "K": str(reduction),
        "input_elements": str(input_elements),
        "output_elements": str(output_elements),
        "storage_elements": str(storage_elements),
        "minimum_bytes": str(4 * (input_elements + output_elements)),
        "storage_bytes": str(4 * storage_elements),
        "flops": str(2 * m * n * reduction),
    }


def generate() -> list[dict[str, str]]:
    rows: dict[tuple[str, str, int], dict[str, str]] = {}

    def add(scale: str, layout: str, threads: int, family: str) -> None:
        key = (scale, layout, threads)
        if key in rows:
            existing = set(rows[key]["families"].split("+"))
            existing.add(family)
            rows[key]["families"] = "+".join(sorted(existing))
        else:
            rows[key] = build_row(scale, layout, threads, family)

    for layout in LAYOUTS:
        add("S", layout, 1, "layout")
    for scale in SCALES:
        for layout in ("X0", "X4"):
            for threads in (1, 2, 4):
                add(scale, layout, threads, "scaling")
    return list(rows.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    assert len(rows) == 28
    assert all(row["equation"] == "abkl,klcd->abcd" for row in rows)
    assert {int(row["threads"]) for row in rows} <= {1, 2, 4}
    print(f"wrote {len(rows)} cases to {args.output}")


if __name__ == "__main__":
    main()
