#!/usr/bin/env python
"""Pure numpy float32 GPT-2 reference forward pass.

Reads weights ONLY from Model/distilgpt2.tsw (TSW1 format written by convert.py),
tokenizes with a from-scratch GPT-2 byte-level BPE (vocab.json + merges.txt),
and writes Model/reference.json with diagnostics the iOS app can be checked against.

Usage:
  python Tools/reference.py [--model Model/distilgpt2.tsw] [--out Model/reference.json]
"""
import argparse
import json
import os
import struct
from functools import lru_cache

import numpy as np
import regex as re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "Model")

PROMPTS = ["The quick brown fox", "I am going to the", "Hello, my name is"]


# ----------------------------------------------------------------------------
# TSW1 reader
# ----------------------------------------------------------------------------
def load_tsw(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"TSW1":
            raise ValueError(f"bad magic {magic!r}")
        (n,) = struct.unpack("<I", f.read(4))
        header = json.loads(f.read(n).decode("utf-8"))
        data_start = (8 + n + 63) // 64 * 64
        f.seek(data_start)
        blob = f.read()
    weights = {}
    for t in header["tensors"]:
        assert t["dtype"] == "f16", t
        assert t["offset"] % 64 == 0, t
        n_el = int(np.prod(t["shape"]))
        assert t["nbytes"] == n_el * 2, t
        arr = np.frombuffer(blob, dtype="<f2", count=n_el, offset=t["offset"])
        weights[t["name"]] = arr.reshape(t["shape"]).astype(np.float32)
    return header, weights


# ----------------------------------------------------------------------------
# GPT-2 byte-level BPE tokenizer
# ----------------------------------------------------------------------------
PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class Tokenizer:
    def __init__(self, vocab_path, merges_path):
        with open(vocab_path, encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_path, encoding="utf-8") as f:
            lines = f.read().split("\n")
        merges = [tuple(l.split()) for l in lines if l and not l.startswith("#version")]
        self.bpe_ranks = {m: i for i, m in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache = {}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        if len(word) < 2:
            return token
        while True:
            pairs = set(zip(word[:-1], word[1:]))
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text):
        ids = []
        for tok in re.findall(PAT, text):
            tok_u = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self.bpe(tok_u).split(" "))
        return ids

    def decode_token(self, idx):
        s = self.decoder[idx]
        return bytearray(self.byte_decoder[c] for c in s).decode("utf-8", errors="replace")

    def decode(self, ids):
        s = "".join(self.decoder[i] for i in ids)
        return bytearray(self.byte_decoder[c] for c in s).decode("utf-8", errors="replace")


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu_new(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention(x, W, i, n_head):
    """Returns (output [T,D], attn weights [H,T,T])."""
    T, D = x.shape
    hd = D // n_head
    qkv = x @ W[f"h.{i}.attn.c_attn.weight"] + W[f"h.{i}.attn.c_attn.bias"]  # [T, 3D]
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(T, n_head, hd).transpose(1, 0, 2)  # [H,T,hd]
    k = k.reshape(T, n_head, hd).transpose(1, 0, 2)
    v = v.reshape(T, n_head, hd).transpose(1, 0, 2)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(hd)  # [H,T,T]
    mask = np.tril(np.ones((T, T), dtype=bool))
    scores = np.where(mask[None], scores, np.float32(-1e10))
    attn = softmax(scores, axis=-1)
    out = (attn @ v).transpose(1, 0, 2).reshape(T, D)
    out = out @ W[f"h.{i}.attn.c_proj.weight"] + W[f"h.{i}.attn.c_proj.bias"]
    return out, attn


def mlp(x, W, i):
    h = gelu_new(x @ W[f"h.{i}.mlp.c_fc.weight"] + W[f"h.{i}.mlp.c_fc.bias"])
    return h @ W[f"h.{i}.mlp.c_proj.weight"] + W[f"h.{i}.mlp.c_proj.bias"]


def forward(ids, W, cfg):
    """Returns dict with logits, per-layer residuals, attention maps."""
    eps = cfg["layer_norm_epsilon"]
    n_head = cfg["n_head"]
    T = len(ids)
    x = W["wte.weight"][ids] + W["wpe.weight"][:T]
    residuals = [x.copy()]
    attns = []
    for i in range(cfg["n_layer"]):
        a, attn = attention(layer_norm(x, W[f"h.{i}.ln_1.weight"], W[f"h.{i}.ln_1.bias"], eps), W, i, n_head)
        x = x + a
        x = x + mlp(layer_norm(x, W[f"h.{i}.ln_2.weight"], W[f"h.{i}.ln_2.bias"], eps), W, i)
        residuals.append(x.copy())
        attns.append(attn)
    final = layer_norm(x, W["ln_f.weight"], W["ln_f.bias"], eps)
    logits = final @ W["wte.weight"].T
    return {"logits": logits, "residuals": residuals, "attns": attns}


def r6(x):
    return round(float(x), 6)


def topk_tokens(probs, tok, k):
    idx = np.argsort(-probs)[:k]
    return [{"id": int(j), "token": tok.decode_token(int(j)), "prob": r6(probs[j])} for j in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(MODEL_DIR, "distilgpt2.tsw"))
    ap.add_argument("--out", default=os.path.join(MODEL_DIR, "reference.json"))
    args = ap.parse_args()

    header, W = load_tsw(args.model)
    cfg = header["config"]
    tok = Tokenizer(os.path.join(MODEL_DIR, "vocab.json"), os.path.join(MODEL_DIR, "merges.txt"))

    results = []
    for prompt in PROMPTS:
        ids = tok.encode(prompt)
        assert tok.decode(ids) == prompt, (prompt, ids, tok.decode(ids))
        out = forward(ids, W, cfg)
        last_probs = softmax(out["logits"][-1])

        logit_lens = []
        for i, res in enumerate(out["residuals"][1:]):
            h = layer_norm(res[-1], W["ln_f.weight"], W["ln_f.bias"], cfg["layer_norm_epsilon"])
            p = softmax(h @ W["wte.weight"].T)
            logit_lens.append({"layer": i, "top": topk_tokens(p, tok, 3)})

        results.append({
            "prompt": prompt,
            "token_ids": ids,
            "tokens": [tok.decode_token(i) for i in ids],
            "top10": topk_tokens(last_probs, tok, 10),
            "logit_lens": logit_lens,
            "residual_norms": [r6(np.linalg.norm(r[-1])) for r in out["residuals"]],
            "attention_last_token": {
                "layer0_head0": [r6(v) for v in out["attns"][0][0, -1]],
                "layer5_head11": [r6(v) for v in out["attns"][5][11, -1]],
            },
        })

    doc = {
        "model": header["model"],
        "config": cfg,
        "notes": "float32 numpy forward from float16 weights in distilgpt2.tsw; gelu_new; causal attention; probs are softmax over all 50257 logits",
        "prompts": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    print(f"wrote {args.out}")

    for r in results:
        print(f"\n{r['prompt']!r} -> {r['token_ids']} {r['tokens']}")
        print("  top10:", [(t["token"], t["prob"]) for t in r["top10"]])
        print("  sum top10:", round(sum(t["prob"] for t in r["top10"]), 4))
        print("  residual norms:", r["residual_norms"])
        for ll in r["logit_lens"]:
            print(f"  L{ll['layer']}:", [(t["token"], t["prob"]) for t in ll["top"]])


if __name__ == "__main__":
    main()
