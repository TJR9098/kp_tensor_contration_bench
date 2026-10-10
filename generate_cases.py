#!/usr/bin/env python3
"""Generate the audited contraction case manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


X_LABELS = list("abcde")
Y_LABELS = list("ghijl")
K_LABELS = list("pqrst")
SHAPE_PROFILES = {
    "G0": (7, 7, 6),
    "G1": (8, 7, 5),
    "G2": (7, 8, 5),
    "G3": (5, 5, 10),
}
FIELDS = [
    "case_id", "families", "aliases", "payload_rank", "physical_rank",
    "contraction_dims", "shape_profile", "distribution", "layout", "batch",
    "threads", "equation", "a_labels", "b_labels", "c_labels",
    "canonical_a_labels", "canonical_b_labels", "a_shape", "b_shape",
    "c_shape", "B", "M", "N", "K", "logical_elements", "input_elements",
    "output_elements", "storage_elements", "minimum_bytes", "flops",
]


def split_exponent(exponent: int, dimensions: int, mode: str) -> list[int]:
    assert dimensions >= 1 and exponent >= dimensions
    if mode == "D0":
        quotient, remainder = divmod(exponent, dimensions)
        return [quotient + (index < remainder) for index in range(dimensions)]
    if mode == "D1":
        return [exponent - dimensions + 1] + [1] * (dimensions - 1)
    if mode == "D2":
        return [1] * (dimensions - 1) + [exponent - dimensions + 1]
    raise ValueError(mode)


def interleave(first: list[str], second: list[str]) -> list[str]:
    output: list[str] = []
    for index in range(max(len(first), len(second))):
        if index < len(first):
            output.append(first[index])
        if index < len(second):
            output.append(second[index])
    return output


def shape_text(labels: list[str], sizes: dict[str, int]) -> str:
    return ",".join(str(sizes[label]) for label in labels)


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def build_row(
    *, case_id: str, family: str, payload_rank: int, contraction_dims: int,
    shape_profile: str, distribution: str, layout: str, batch: int, threads: int,
    m_exp: int, n_exp: int, k_exp: int, include_batch_axis: bool,
) -> dict[str, str]:
    free_dims = payload_rank - contraction_dims
    x_labels = X_LABELS[:free_dims]
    y_labels = Y_LABELS[:free_dims]
    k_labels = K_LABELS[:contraction_dims]
    sizes: dict[str, int] = {}
    for labels, exponent in (
        (x_labels, m_exp), (y_labels, n_exp), (k_labels, k_exp)
    ):
        for label, part in zip(labels, split_exponent(exponent, len(labels), distribution)):
            sizes[label] = 2**part
    prefix = ["z"] if include_batch_axis else []
    if include_batch_axis:
        sizes["z"] = batch

    canonical_a = prefix + x_labels + k_labels
    canonical_b = prefix + y_labels + k_labels
    c_labels = prefix + x_labels + y_labels
    if layout in {"L0", "L4", "L5"}:
        a_labels = list(canonical_a)
        b_labels = list(canonical_b)
    elif layout == "L1":
        a_labels = prefix + x_labels + k_labels
        b_labels = prefix + k_labels + y_labels
    elif layout == "L2":
        a_labels = prefix + k_labels + x_labels
        b_labels = prefix + k_labels + y_labels
    elif layout == "L3":
        a_labels = prefix + interleave(x_labels, k_labels)
        b_labels = prefix + interleave(y_labels, list(reversed(k_labels)))
    else:
        raise ValueError(layout)

    a_shape = [sizes[label] for label in a_labels]
    b_shape = [sizes[label] for label in b_labels]
    c_shape = [sizes[label] for label in c_labels]
    m = 2**m_exp
    n = 2**n_exp
    k = 2**k_exp
    logical_elements = batch * m * n * k
    assert logical_elements == 2**20
    input_elements = batch * (m * k + n * k)
    output_elements = batch * m * n
    storage_elements = input_elements + output_elements
    if layout in {"L4", "L5"}:
        storage_elements += batch * (m * k + n * k)
    equation = f"{''.join(a_labels)},{''.join(b_labels)}->{''.join(c_labels)}"
    return {
        "case_id": case_id,
        "families": family,
        "aliases": "",
        "payload_rank": str(payload_rank),
        "physical_rank": str(payload_rank + int(include_batch_axis)),
        "contraction_dims": str(contraction_dims),
        "shape_profile": shape_profile,
        "distribution": distribution,
        "layout": layout,
        "batch": str(batch),
        "threads": str(threads),
        "equation": equation,
        "a_labels": "".join(a_labels),
        "b_labels": "".join(b_labels),
        "c_labels": "".join(c_labels),
        "canonical_a_labels": "".join(canonical_a),
        "canonical_b_labels": "".join(canonical_b),
        "a_shape": ",".join(map(str, a_shape)),
        "b_shape": ",".join(map(str, b_shape)),
        "c_shape": ",".join(map(str, c_shape)),
        "B": str(batch), "M": str(m), "N": str(n), "K": str(k),
        "logical_elements": str(logical_elements),
        "input_elements": str(input_elements),
        "output_elements": str(output_elements),
        "storage_elements": str(storage_elements),
        "minimum_bytes": str(4 * (input_elements + output_elements)),
        "flops": str(2 * logical_elements),
    }


def identity(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in (
        "payload_rank", "physical_rank", "contraction_dims", "layout", "batch",
        "threads", "equation", "a_shape", "b_shape", "c_shape",
    ))


def generate() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: dict[tuple[str, ...], dict[str, str]] = {}

    def add(row: dict[str, str]) -> None:
        key = identity(row)
        if key not in seen:
            seen[key] = row
            rows.append(row)
            return
        existing = seen[key]
        if row["case_id"] != existing["case_id"]:
            existing["aliases"] = ";".join(
                filter(None, [existing["aliases"], row["case_id"]])
            )
        families = set(existing["families"].split("+"))
        families.update(row["families"].split("+"))
        existing["families"] = "+".join(sorted(families))

    for rank in (4, 5, 6):
        for contraction_dims in range(1, rank):
            for profile, (m_exp, n_exp, k_exp) in SHAPE_PROFILES.items():
                for distribution in ("D0", "D1", "D2"):
                    add(build_row(
                        case_id=f"R{rank}_Q{contraction_dims}_{profile}_{distribution}_L0_B1_T1",
                        family="shape", payload_rank=rank,
                        contraction_dims=contraction_dims, shape_profile=profile,
                        distribution=distribution, layout="L0", batch=1, threads=1,
                        m_exp=m_exp, n_exp=n_exp, k_exp=k_exp,
                        include_batch_axis=False,
                    ))

    for rank in (4, 5, 6):
        for contraction_dims in range(1, rank):
            for layout in ("L0", "L1", "L2", "L3", "L4", "L5"):
                add(build_row(
                    case_id=f"R{rank}_Q{contraction_dims}_G0_D0_{layout}_B1_T1",
                    family="layout", payload_rank=rank,
                    contraction_dims=contraction_dims, shape_profile="G0",
                    distribution="D0", layout=layout, batch=1, threads=1,
                    m_exp=7, n_exp=7, k_exp=6, include_batch_axis=False,
                ))

    batch_q = {4: 2, 5: 2, 6: 3}
    batch_shapes = {1: 8, 4: 7, 16: 6, 64: 5}
    for rank, contraction_dims in batch_q.items():
        for batch, free_exp in batch_shapes.items():
            for threads in (1, 2, 4):
                add(build_row(
                    case_id=f"R{rank}_Q{contraction_dims}_BATCH_B{batch}_T{threads}",
                    family="batch", payload_rank=rank,
                    contraction_dims=contraction_dims, shape_profile="BFIX",
                    distribution="D0", layout="L0", batch=batch, threads=threads,
                    m_exp=free_exp, n_exp=free_exp, k_exp=4,
                    include_batch_axis=True,
                ))
    return rows


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
    assert all(int(row["logical_elements"]) == 2**20 for row in rows)
    print(f"wrote {len(rows)} unique cases to {args.output}")


if __name__ == "__main__":
    main()
