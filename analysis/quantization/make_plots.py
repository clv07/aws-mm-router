"""Plots for the c7i.2xlarge quantization row (f16 / Q8_0 / Q4_K_M).

Reads results/phase1/c7i.2xlarge__*/summary.csv and results/quality/*.csv,
writes PNGs next to this file. Run from anywhere: python make_plots.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / "results"
QUANTS = ["f16", "Q8_0", "Q4_K_M"]
COLOR = {"f16": "#2a78d6", "Q8_0": "#eb6834", "Q4_K_M": "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


summary = {q: read(RESULTS / f"phase1/c7i.2xlarge__Qwen2.5-Omni-3B-{q}/summary.csv") for q in QUANTS}
UNIT = {"question": "", "tokens": " tok", "images": " img", "seconds": " s"}
LABELS = [f"{r['modality']}\n{r['size']}{UNIT.get(r['size_unit'], ' ' + r['size_unit'])}" for r in summary["Q8_0"]]


def col(q, key):
    return np.array([float(r[key]) for r in summary[q]])


def grouped_bars(ax, key, fmt, label_every=False):
    n, w = len(LABELS), 0.27
    x = np.arange(n)
    for i, q in enumerate(QUANTS):
        v = col(q, key)
        ax.bar(x + (i - 1) * w, v, w - 0.03, color=COLOR[q], label=q, edgecolor=SURFACE, linewidth=1)
        if label_every:
            for xi, vi in zip(x, v):
                ax.text(xi + (i - 1) * w, vi, fmt(vi), ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, LABELS, fontsize=8.5)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper left", ncol=3)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / name, dpi=160)
    plt.close(fig)


# 1. Decode speed
fig, ax = plt.subplots(figsize=(10, 4))
grouped_bars(ax, "decode_tok_s", "{:.1f}".format, label_every=True)
ax.set_ylabel("decode tokens / s (higher is better)")
ax.set_title("Decode speed: Q8_0 1.8× f16, Q4_K_M a further 1.2×")
ax.set_ylim(0, 20)
save(fig, "fig1_decode.png")

# 2. Time to first token
fig, ax = plt.subplots(figsize=(10, 4))
grouped_bars(ax, "ttft_s", lambda v: f"{v:.1f}" if v < 10 else f"{v:.0f}", label_every=True)
ax.set_ylabel("time to first token, s (lower is better)")
ax.set_title("Time to first token: big drop f16 → Q8_0, small drop Q8_0 → Q4_K_M")
save(fig, "fig2_ttft.png")

# 3. TTFT split: LLM prefill (fit on text runs) vs the rest (encoder, Q8_0 in every run)
fits = {}
for q in QUANTS:
    text = [r for r in summary[q] if r["modality"] in ("text", "chat")]
    fits[q] = np.polyfit([float(r["prompt_tokens"]) for r in text], [float(r["ttft_s"]) for r in text], 1)
mm = [i for i, r in enumerate(summary["Q8_0"]) if r["modality"] in ("image", "audio")]
fig, axes = plt.subplots(1, len(mm), figsize=(10, 3.6), sharey=True)
for ax, i in zip(axes, mm):
    for j, q in enumerate(QUANTS):
        r = summary[q][i]
        a, b = fits[q]
        llm = max(a * float(r["prompt_tokens"]) + b, 0)
        enc = float(r["ttft_s"]) - llm
        ax.bar(j, llm, 0.7, color=COLOR[q], edgecolor=SURFACE, linewidth=2)
        ax.bar(j, enc, 0.7, bottom=llm, color="#c9c8c2", edgecolor=SURFACE, linewidth=2)
        ax.text(j, llm / 2, f"{llm:.0f}", ha="center", va="center", fontsize=8, color="white")
        ax.text(j, llm + enc / 2, f"{enc:.0f}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(3), QUANTS, fontsize=8)
    ax.tick_params(axis="x", length=0)
    r = summary["Q8_0"][i]
    ax.set_title(f"{r['size']} {('image' if r['size'] == '1' else 'images') if r['modality'] == 'image' else 's audio'}", fontsize=10, fontweight="normal")
axes[0].set_ylabel("time to first token, s")
fig.suptitle("Where multimodal TTFT goes: colored = language-model prefill, gray = encoder (nearly constant)",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
save(fig, "fig3_ttft_split.png")

# 4. Quality
scores = read(RESULTS / "quality/scores.csv")
ppl = {r["model"].split("-")[-1]: (float(r["ppl"]), float(r["ppl_error"])) for r in read(RESULTS / "quality/perplexity.csv")}
sc = {(r["model"].split("-")[-1], r["task"]): r for r in scores}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 2.2, 1]})
ax = axes[0]
for j, q in enumerate(QUANTS):
    p, e = ppl[q]
    ax.bar(j, p, 0.7, color=COLOR[q], yerr=e, ecolor=INK2, capsize=3)
    ax.text(j, p + e + 0.2, f"{p:.2f}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(range(3), QUANTS, fontsize=8)
ax.set_title("Perplexity (wikitext-2)\nlower is better", fontsize=10)
ax.set_ylim(0, 16)
ax = axes[1]
tasks = ["chat", "summarize", "image"]
for j, q in enumerate(["Q8_0", "Q4_K_M"]):
    v = [float(sc[(q, t)]["rouge_l_vs_reference"]) for t in tasks]
    xs = np.arange(3) + (j - 0.5) * 0.36
    ax.bar(xs, v, 0.33, color=COLOR[q], label=q)
    for xi, vi in zip(xs, v):
        ax.text(xi, vi + 0.02, f"{vi:.2f}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(range(3), tasks)
ax.set_ylim(0, 1.1)
ax.legend(loc="upper right", ncol=2)
ax.set_title("ROUGE-L vs the f16 answer (1.0 = same words)\nmeasures sameness, not correctness", fontsize=10)
ax = axes[2]
for j, q in enumerate(QUANTS):
    v = 100 * float(sc[(q, "transcribe")]["wer_vs_transcript"])
    ax.bar(j, v, 0.7, color=COLOR[q])
    ax.text(j, v + 0.1, f"{v:.1f}%", ha="center", fontsize=8, color=INK2)
ax.set_xticks(range(3), QUANTS, fontsize=8)
ax.set_ylim(0, 7)
ax.set_title("Transcription WER\nlower is better", fontsize=10)
for ax in axes:
    ax.tick_params(axis="x", length=0)
save(fig, "fig4_quality.png")

# 5. Trade-off: decode speed vs perplexity, RAM in the label
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.grid(axis="x")
for q in QUANTS:
    d = col(q, "decode_tok_s")[0]
    p, e = ppl[q]
    ram = col(q, "peak_ram_gb").max()
    ax.errorbar(d, p, yerr=e, fmt="o", ms=10, color=COLOR[q], ecolor=COLOR[q], capsize=3,
                markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"{q}\n{ram:.1f} GB peak RAM", (d, p), xytext=(12, -4), textcoords="offset points",
                fontsize=9, color=INK)
ax.set_xlim(5, 20)
ax.set_ylim(9.5, 15)
ax.set_xlabel("decode tokens / s on chat (faster →)")
ax.set_ylabel("perplexity (← better)")
ax.set_title("Speed vs quality: Q8_0 is free, Q4_K_M costs quality")
save(fig, "fig5_tradeoff.png")

print("wrote", sorted(p.name for p in HERE.glob("*.png")))
