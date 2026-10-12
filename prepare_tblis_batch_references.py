#!/usr/bin/env python3
"""Stream shared FP64 references for the maximum batch of each logical scale."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path


MAX_BATCH = 32


def deterministic_tensor(torch, shape: list[int], salt: int, offset: int):
    count = 1
    for dimension in shape:
        count *= dimension
    index = torch.arange(offset, offset + count, dtype=torch.int64)
    values = ((index * 1103515245 + salt) & 0xFFFF).to(torch.float32)
    return values.div_(32768.0).sub_(1.0).reshape(shape)


def tensor_bytes(tensor) -> bytes:
    contiguous = tensor.detach().contiguous()
    return ctypes.string_at(contiguous.data_ptr(), contiguous.numel() * contiguous.element_size())


def read_scales(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        result.setdefault(row["scale"], row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scale", action="append", choices=["S", "M", "L", "XL"])
    args = parser.parse_args()
    os.sched_setaffinity(0, {0, 1, 2, 3})

    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scales = read_scales(args.manifest)
    selected = args.scale or list(scales)
    for scale in selected:
        row = scales[scale]
        output_path = args.output_dir / f"{scale}_B{MAX_BATCH}.f64"
        metadata_path = args.output_dir / f"{scale}_B{MAX_BATCH}.json"
        per_output = int(row["output_elements"]) // int(row["batch"])
        per_flops = int(row["flops"]) // int(row["batch"])
        expected_size = MAX_BATCH * per_output * 8
        if output_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                output_path.stat().st_size == expected_size
                and metadata["per_contraction_flops"] == per_flops
                and metadata["max_batch"] == MAX_BATCH
            ):
                continue
            raise RuntimeError(f"stale reference: {output_path}")

        a_shape = [int(row[name]) for name in ("a", "b", "k", "l")]
        b_shape = [int(row[name]) for name in ("k", "l", "c", "d")]
        c_shape = [int(row[name]) for name in ("a", "b", "c", "d")]
        a_elements = int(row["M"]) * int(row["K"])
        b_elements = int(row["K"]) * int(row["N"])
        digest = hashlib.sha256()
        temporary = output_path.with_suffix(".tmp")
        with temporary.open("wb") as stream, torch.inference_mode():
            for batch_index in range(MAX_BATCH):
                a = deterministic_tensor(torch, a_shape, 12345, batch_index * a_elements)
                b = deterministic_tensor(torch, b_shape, 54321, batch_index * b_elements)
                reference = torch.mm(
                    a.double().reshape(int(row["M"]), int(row["K"])),
                    b.double().reshape(int(row["K"]), int(row["N"])),
                ).reshape(c_shape)
                payload = tensor_bytes(reference)
                stream.write(payload)
                digest.update(payload)
        temporary.replace(output_path)
        metadata = {
            "scale": scale,
            "max_batch": MAX_BATCH,
            "dtype": "float64-little-endian",
            "per_batch_shape": c_shape,
            "per_batch_elements": per_output,
            "bytes": expected_size,
            "sha256": digest.hexdigest(),
            "per_contraction_flops": per_flops,
            "generator": "streamed FP64 torch.mm over independent deterministic batches",
            "torch_version": torch.__version__,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"scale": scale, "bytes": expected_size, "sha256": digest.hexdigest()}))


if __name__ == "__main__":
    main()
