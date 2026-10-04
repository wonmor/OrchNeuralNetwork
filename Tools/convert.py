#!/usr/bin/env python
"""Convert HuggingFace distilgpt2 safetensors -> Model/distilgpt2.tsw (TSW1 format).

Format:
  bytes 0-3   : ASCII magic "TSW1"
  bytes 4-7   : uint32 LE, length N of JSON header
  bytes 8..8+N: UTF-8 JSON header
  zero padding up to the first 64-byte-aligned offset from file start -> data section
  tensors     : raw float16 LE, row-major, each at a 64-byte-aligned offset
                relative to the data section start, in header order.

Usage:
  python Tools/convert.py [--src path/to/model.safetensors] [--out Model/distilgpt2.tsw]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIGN = 64

CONFIG = {
    "n_layer": 6,
    "n_head": 12,
    "n_embd": 768,
    "n_ctx": 1024,
    "n_vocab": 50257,
    "layer_norm_epsilon": 1e-5,
}

EXPECTED_SHAPES = {
    "wte.weight": (50257, 768),
    "wpe.weight": (1024, 768),
    "ln_f.weight": (768,),
    "ln_f.bias": (768,),
    "ln_1.weight": (768,),
    "ln_1.bias": (768,),
    "attn.c_attn.weight": (768, 2304),
    "attn.c_attn.bias": (2304,),
    "attn.c_proj.weight": (768, 768),
    "attn.c_proj.bias": (768,),
    "ln_2.weight": (768,),
    "ln_2.bias": (768,),
    "mlp.c_fc.weight": (768, 3072),
    "mlp.c_fc.bias": (3072,),
    "mlp.c_proj.weight": (3072, 768),
    "mlp.c_proj.bias": (768,),
}


def tensor_order(n_layer):
    names = ["wte.weight", "wpe.weight"]
    per_layer = [
        "ln_1.weight", "ln_1.bias",
        "attn.c_attn.weight", "attn.c_attn.bias",
        "attn.c_proj.weight", "attn.c_proj.bias",
        "ln_2.weight", "ln_2.bias",
        "mlp.c_fc.weight", "mlp.c_fc.bias",
        "mlp.c_proj.weight", "mlp.c_proj.bias",
    ]
    for i in range(n_layer):
        names += [f"h.{i}.{s}" for s in per_layer]
    names += ["ln_f.weight", "ln_f.bias"]
    return names


def expected_shape(name):
    key = name.split(".", 2)[2] if name.startswith("h.") else name
    return EXPECTED_SHAPES[key]


def align_up(n, a=ALIGN):
    return (n + a - 1) // a * a


def resolve_src(src):
    if src:
        return src
    from huggingface_hub import hf_hub_download
    return hf_hub_download("distilgpt2", "model.safetensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="path to model.safetensors (downloaded from HF if omitted)")
    ap.add_argument("--out", default=os.path.join(ROOT, "Model", "distilgpt2.tsw"))
    args = ap.parse_args()

    src = resolve_src(args.src)
    names = tensor_order(CONFIG["n_layer"])

    tensors = []
    with safe_open(src, "np") as f:
        keys = set(f.keys())
        for name in names:
            hf_name = "transformer." + name
            if hf_name not in keys:
                sys.exit(f"missing tensor in source: {hf_name}")
            t = f.get_tensor(hf_name)
            exp = expected_shape(name)
            if tuple(t.shape) != exp:
                sys.exit(f"{name}: shape {tuple(t.shape)} != expected {exp}")
            # Keep HF Conv1D [in, out] layout; no transpose. Row-major float16.
            t16 = np.ascontiguousarray(t.astype(np.float32).astype("<f2"))
            tensors.append((name, t16))

    # Lay out data section.
    entries = []
    offset = 0
    for name, t in tensors:
        nbytes = t.nbytes
        entries.append({
            "name": name,
            "shape": [int(s) for s in t.shape],
            "dtype": "f16",
            "offset": offset,
            "nbytes": nbytes,
        })
        offset = align_up(offset + nbytes)

    header = {
        "model": "distilgpt2",
        "license": "Apache-2.0",
        "source": "https://huggingface.co/distilgpt2",
        "config": CONFIG,
        "tensors": entries,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    data_start = align_up(8 + len(header_bytes))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as out:
        out.write(b"TSW1")
        out.write(struct.pack("<I", len(header_bytes)))
        out.write(header_bytes)
        out.write(b"\x00" * (data_start - 8 - len(header_bytes)))
        for (name, t), e in zip(tensors, entries):
            pos = out.tell() - data_start
            if pos != e["offset"]:
                sys.exit(f"layout error at {name}: {pos} != {e['offset']}")
            out.write(t.tobytes(order="C"))
            pad = align_up(out.tell() - data_start) - (out.tell() - data_start)
            out.write(b"\x00" * pad)

    size = os.path.getsize(args.out)
    print(f"wrote {args.out}: {size} bytes ({size / 1e6:.1f} MB), "
          f"{len(entries)} tensors, header {len(header_bytes)} bytes, data at {data_start}")


if __name__ == "__main__":
    main()
