#!/usr/bin/env python
"""How much does storing DistilGPT2 as float16 change its predictions?

Runs the numpy reference forward pass twice over a set of prompts: once with the original float32
safetensors weights from Hugging Face, once with the float16 weights in Model/distilgpt2.tsw expanded back
to float32 (exactly what the app does). Reports KL(p32 || p16), top-1 and top-3 agreement, and the largest
absolute change in next-token probability. Writes results/fidelity.json.

Usage: ~/.venvs/mflux/bin/python scripts/fidelity.py [--src model.safetensors]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Tools"))
import reference as R  # noqa: E402

PROMPTS = [
    "The quick brown fox", "I am going to the", "Hello, my name is", "The weather today is",
    "My favourite food is", "She opened the door and", "In 1969, astronauts first landed on the",
    "The capital of France is", "To be or not to", "Once upon a time, there was a",
    "The patient was given a dose of", "He turned the key and the engine", "Please remember to bring your",
    "The stock market fell sharply after", "A transformer is a neural network that", "Thank you for your",
    "I can't believe it's already", "The recipe calls for two cups of", "After the meeting, we decided to",
    "The best way to learn a language is", "On Monday morning the train was", "The function returns a list of",
    "Water boils at a temperature of", "My phone keeps suggesting the word", "If you look closely you can see",
    "The committee voted to approve the", "Breaking news: the president has", "Her favourite book as a child was",
    "Tomorrow I will finally", "The dog barked at the",
]


def load_fp32(src):
    from safetensors import safe_open
    W = {}
    with safe_open(src, "np") as f:
        for k in f.keys():
            if k.startswith("transformer."):
                W[k[len("transformer."):]] = f.get_tensor(k).astype(np.float32)
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "fidelity.json"))
    args = ap.parse_args()
    src = args.src
    if src is None:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download("distilgpt2", "model.safetensors")

    header, W16 = R.load_tsw(os.path.join(ROOT, "Model", "distilgpt2.tsw"))
    cfg = header["config"]
    W32 = load_fp32(src)
    tok = R.Tokenizer(os.path.join(R.MODEL_DIR, "vocab.json"), os.path.join(R.MODEL_DIR, "merges.txt"))

    # Weight-level error.
    rel = []
    for k, a in W32.items():
        if k not in W16:          # HF also stores the causal-mask buffers attn.bias / attn.masked_bias
            continue
        b = W16[k]
        rel.append(float(np.linalg.norm(a - b) / np.linalg.norm(a)))
    rows = []
    for prompt in PROMPTS:
        ids = tok.encode(prompt)
        p32 = R.softmax(R.forward(ids, W32, cfg)["logits"][-1].astype(np.float64))
        p16 = R.softmax(R.forward(ids, W16, cfg)["logits"][-1].astype(np.float64))
        kl = float(np.sum(p32 * (np.log(p32 + 1e-30) - np.log(p16 + 1e-30))))
        t32, t16 = np.argsort(-p32)[:3], np.argsort(-p16)[:3]
        rows.append({
            "prompt": prompt, "tokens": len(ids), "kl_bits": kl / np.log(2),
            "top1_agree": bool(t32[0] == t16[0]), "top3_set_agree": bool(set(t32) == set(t16)),
            "top1_token": tok.decode_token(int(t32[0])), "top1_p32": float(p32[t32[0]]), "top1_p16": float(p16[t32[0]]),
            "max_abs_dp": float(np.max(np.abs(p32 - p16))),
        })
    doc = {
        "n_prompts": len(rows),
        "weight_rel_fro_err_max": max(rel), "weight_rel_fro_err_median": float(np.median(rel)),
        "kl_bits_mean": float(np.mean([r["kl_bits"] for r in rows])), "kl_bits_max": float(np.max([r["kl_bits"] for r in rows])),
        "top1_agree": int(sum(r["top1_agree"] for r in rows)), "top3_set_agree": int(sum(r["top3_set_agree"] for r in rows)),
        "max_abs_dp": float(np.max([r["max_abs_dp"] for r in rows])),
        "prompts": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(doc, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k != "prompts"}, indent=1))


if __name__ == "__main__":
    main()
