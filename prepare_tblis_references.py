#!/usr/bin/env python3
"""Create one shared FP64 reference output for each logical contraction size."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path


def deterministic_tensor(torch, shape: list[int], salt: int):
    count = 1
    for dimension in shape:
        count *= dimension
    index = torch.arange(count, dtype=torch.int64)
    values = ((index * 1103515245 + salt) & 0xFFFF).to(torch.float32)
    return values.div_(32768.0).sub_(1.0).reshape(shape)


def tensor_bytes(tensor) -> bytes:
    contiguous = tensor.detach().contiguous()
    return ctypes.string_at(contiguous.data_ptr(), contiguous.numel() * contiguous.element_size())


def read_scales(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return {row["scale"]: row for row in rows}


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
        output_path = args.output_dir / f"{scale}.f64"
        metadata_path = args.output_dir / f"{scale}.json"
        expected_size = int(row["output_elements"]) * 8
        if output_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if output_path.stat().st_size == expected_size and metadata["flops"] == int(row["flops"]):
                continue
            raise RuntimeError(f"stale reference: {output_path}")

        a = deterministic_tensor(torch, [int(row[name]) for name in ("a", "b", "k", "l")], 12345)
        b = deterministic_tensor(torch, [int(row[name]) for name in ("k", "l", "c", "d")], 54321)
        with torch.inference_mode():
            reference = torch.mm(
                a.double().reshape(int(row["M"]), int(row["K"])),
                b.double().reshape(int(row["K"]), int(row["N"])),
            ).reshape([int(row[name]) for name in ("a", "b", "c", "d")])
        payload = tensor_bytes(reference)
        temporary = output_path.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(output_path)
        metadata = {
            "scale": scale,
            "dtype": "float64-little-endian",
            "shape": [int(row[name]) for name in ("a", "b", "c", "d")],
            "elements": int(row["output_elements"]),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "flops": int(row["flops"]),
            "generator": "FP64 torch.mm over canonical logical inputs",
            "torch_version": torch.__version__,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"scale": scale, "bytes": len(payload), "sha256": metadata["sha256"]}))


if __name__ == "__main__":
    main()
