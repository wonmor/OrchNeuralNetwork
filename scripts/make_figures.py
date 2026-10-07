#!/usr/bin/env python
"""Turns results/bench_*.json and results/fidelity.json into figures/*.pdf and results/macros.tex.

Every number quoted in paper/orch-neural-network.tex is a macro defined here, so the manuscript cannot
drift from the measurements. Add a device by dropping another bench_<name>.json into results/ (see
scripts/run_benchmark.sh) and re-running this script.
"""
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# DistilGPT2 dimensions (from the weight header).
D, F, V, L, H, HD = 768, 3072, 50257, 6, 12, 64


def label_for(bench):
    d = bench["device"]
    if d["simulator"]:
        return f"iOS Simulator ({d['simulated_model']}) on {d['host_cpu']}"
    return d["machine"]


def short_label(bench):
    d = bench["device"]
    return "Simulator / M1 Max" if d["simulator"] else d["machine"]


def macs(T):
    """Multiply-accumulates for one traced forward pass, split by component."""
    blocks = T * L * (D * 3 * D + D * D + 2 * D * F)           # QKV, out-proj, two MLP matmuls
    attention = L * H * 2 * T * T * HD                           # QK^T and AV, all positions
    unembed = (L + 1) * D * V                                    # 6 logit-lens reads + the final one
    return blocks, attention, unembed


def fmt(x, nd=1):
    return f"{x:.{nd}f}"


def main():
    benches = []
    for path in sorted(glob.glob(os.path.join(RES, "bench_*.json"))):
        with open(path) as f:
            benches.append((os.path.basename(path)[6:-5], json.load(f)))
    fidelity = json.load(open(os.path.join(RES, "fidelity.json")))
    macros = []

    def M(name, value):
        macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    # ---------------- model / file ----------------
    b0 = benches[0][1]
    load = b0["load"]
    M("NumParams", f"{load['parameters']/1e6:.1f}")
    M("NumParamsExact", f"{load['parameters']:,}")
    M("NumTensors", load["tensors"])
    M("WeightFileMB", fmt(load["weight_file_bytes"] / 1e6))
    M("WeightFpThirtyTwoMB", fmt(load["parameters"] * 4 / 1e6))
    M("VocabSize", f"{V:,}")

    # ---------------- per-device table ----------------
    rows = []
    for key, b in benches:
        lat = {r["T"]: r for r in b["latency"]}
        ld = b["load"]
        delta_mb = (ld["footprint_after_load_bytes"] - ld["footprint_before_bytes"]) / 1e6
        peak_mb = (b["footprint_peak_bytes"] - ld["footprint_before_bytes"]) / 1e6
        rows.append(f"{short_label(b)} & {fmt(ld['total_ms'],0)} & {fmt(lat[1]['forward_ms']['median'])} & "
                    f"{fmt(lat[8]['forward_ms']['median'])} & {fmt(lat[16]['forward_ms']['median'])} & "
                    f"{fmt(lat[32]['forward_ms']['median'])} & {fmt(lat[48]['forward_ms']['median'])} & {fmt(delta_mb,0)} \\\\")
        tag = "Sim" if b["device"]["simulator"] else key.title()
        M(f"LoadMs{tag}", fmt(ld["total_ms"], 0))
        M(f"ExpandMs{tag}", fmt(ld["fp16_to_fp32_expand_ms"], 0))
        M(f"TokLoadMs{tag}", fmt(ld["tokenizer_ms"], 0))
        M(f"LatOne{tag}", fmt(lat[1]["forward_ms"]["median"], 0))
        M(f"LatEight{tag}", fmt(lat[8]["forward_ms"]["median"], 0))
        short = [lat[t]["forward_ms"]["median"] for t in (1, 2, 4, 8)]
        M(f"LatShortMin{tag}", fmt(min(short), 0))
        M(f"LatShortMax{tag}", fmt(max(short), 0))
        M(f"LatSixteen{tag}", fmt(lat[16]["forward_ms"]["median"], 0))
        M(f"LatFortyEight{tag}", fmt(lat[48]["forward_ms"]["median"], 0))
        M(f"LatFortyEightPninety{tag}", fmt(lat[48]["forward_ms"]["p90"], 0))
        M(f"FootprintDeltaMB{tag}", fmt(delta_mb, 0))
        M(f"FootprintPeakMB{tag}", fmt(peak_mb, 0))
        M(f"EncodeUs{tag}", fmt(b["tokenizer"]["encode_ms"]["median"] * 1000, 0))
        M(f"Repeats{tag}", int(lat[1]["forward_ms"]["n"]))
        M(f"Thermal{tag}", b["thermal_state_end"])
        M(f"DeviceLabel{tag}", label_for(b))
        M(f"OSVersion{tag}", b["device"]["os_version"])
    M("DeviceRows", "\n".join(rows))
    M("NumDevices", len(benches))

    # ---------------- oracle agreement ----------------
    ag = b0["agreement"]
    M("AgreePrompts", ag["n_prompts"])
    M("AgreeTopTenExact", ag["top10_order_exact"])
    M("AgreeLensExact", f"{ag['logit_lens_top1_agree']}/{ag['logit_lens_total']}")
    M("AgreeMaxProbErr", f"{ag['max_abs_prob_err']:.1e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}")
    M("AgreeMaxNormRel", f"{ag['max_rel_residual_norm_err']:.1e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}")
    M("AgreeMaxAttnErr", f"{ag['max_abs_attention_err']:.1e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}")

    # ---------------- fp16 fidelity ----------------
    M("FidPrompts", fidelity["n_prompts"])
    M("FidTopOne", fidelity["top1_agree"])
    M("FidTopThree", fidelity["top3_set_agree"])
    M("FidKLMeanBits", f"{fidelity['kl_bits_mean']*1e5:.1f}")      # in units of 1e-5 bits
    M("FidKLMaxBits", f"{fidelity['kl_bits_max']*1e5:.1f}")
    M("FidMaxDp", f"{fidelity['max_abs_dp']*100:.2f}")               # percentage points
    M("FidWeightErrMedian", f"{fidelity['weight_rel_fro_err_median']*1e4:.1f}")  # 1e-4
    M("FidWeightErrMax", f"{fidelity['weight_rel_fro_err_max']*1e4:.1f}")
    worst = max(fidelity["prompts"], key=lambda r: r["max_abs_dp"])
    M("FidWorstPrompt", worst["prompt"])
    M("FidWorstToken", worst["top1_token"].strip())
    M("FidWorstPthirtytwo", f"{worst['top1_p32']*100:.2f}")
    M("FidWorstPsixteen", f"{worst['top1_p16']*100:.2f}")

    # ---------------- cost decomposition ----------------
    Ts = [1, 8, 16, 32, 48]
    cost_rows = []
    for T in Ts:
        bl, at, un = macs(T)
        tot = bl + at + un
        cost_rows.append(f"{T} & {bl/1e6:.0f} & {at/1e6:.1f} & {un/1e6:.0f} & {tot/1e6:.0f} & {100*un/tot:.0f}\\% \\\\")
    M("CostRows", "\n".join(cost_rows))
    bl1, at1, un1 = macs(1)
    M("LensShareOne", f"{100*un1/(bl1+at1+un1):.0f}")
    bl48, at48, un48 = macs(48)
    M("LensShareFortyEight", f"{100*un48/(bl48+at48+un48):.0f}")
    M("UnembedMMAC", f"{D*V/1e6:.1f}")
    M("UnembedTotalMMAC", f"{un1/1e6:.0f}")
    M("BlockMMACPerToken", f"{bl1/1e6:.1f}")
    # Effective throughput at T=48 for the first device.
    lat48 = {r["T"]: r for r in b0["latency"]}[48]["forward_ms"]["median"]
    M("GflopsFortyEightSim", f"{2*(bl48+at48+un48)/lat48/1e6:.0f}")   # GFLOP/s = 2*MAC / (ms*1e-3) / 1e9

    with open(os.path.join(RES, "macros.tex"), "w") as f:
        f.write("% Generated by scripts/make_figures.py. Do not edit.\n" + "\n".join(macros) + "\n")

    # ---------------- Figure 1: latency vs T and cost decomposition ----------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    for key, b in benches:
        Tn = [r["T"] for r in b["latency"]]
        med = [r["forward_ms"]["median"] for r in b["latency"]]
        lo = [r["forward_ms"]["p10"] for r in b["latency"]]
        hi = [r["forward_ms"]["p90"] for r in b["latency"]]
        ax.plot(Tn, med, "o-", label=short_label(b))
        ax.fill_between(Tn, lo, hi, alpha=0.2)
    ax.set_xlabel("input length $T$ (tokens)")
    ax.set_ylabel("traced forward pass (ms)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 48])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 48])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("(a) Measured latency (median, p10–p90)", fontsize=10)

    ax = axes[1]
    Tf = [1, 2, 4, 8, 16, 32, 48]
    bl = [macs(T)[0] / 1e6 for T in Tf]
    at = [macs(T)[1] / 1e6 for T in Tf]
    un = [macs(T)[2] / 1e6 for T in Tf]
    x = range(len(Tf))
    ax.bar(x, bl, label="transformer blocks", color="#4C72B0")
    ax.bar(x, at, bottom=bl, label="attention scores", color="#55A868")
    ax.bar(x, un, bottom=[a + b for a, b in zip(bl, at)], label="unembedding (6 lens reads + final)", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(Tf)
    ax.set_xlabel("input length $T$ (tokens)")
    ax.set_ylabel("multiply-accumulates (millions)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    ax.set_title("(b) Analytic cost of one traced pass", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_latency.pdf"))

    # ---------------- Figure 2: fp16 fidelity per prompt ----------------
    fig, ax = plt.subplots(figsize=(10, 3.6))
    rows = fidelity["prompts"]
    ax.bar(range(len(rows)), [r["kl_bits"] * 1e5 for r in rows], color="#4C72B0")
    ax.set_ylabel(r"KL$(p_{32}\|p_{16})$, $10^{-5}$ bits", fontsize=9)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["prompt"] for r in rows], rotation=60, ha="right", fontsize=6.5)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_fidelity.pdf"))
    print("wrote", os.path.join(RES, "macros.tex"), "and figures; devices:", [k for k, _ in benches])


if __name__ == "__main__":
    main()
