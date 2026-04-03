"""
plot_results.py — comprehensive infergo benchmark visualizations.

Generates:
  benchmark_throughput.png    — tok/s + req/s, all scenarios (CUDA/CPU × short/long)
  benchmark_latency.png       — P50 / P95 / P99 full distribution, all scenarios
  benchmark_coldstart.png     — cold-start latency (CUDA + CPU)
  benchmark_opt2_delta.png    — before vs after OPT-2 continuous batching (CUDA only)
  benchmark_gpu_cpu.png       — GPU vs CPU speedup comparison
  benchmark_embedding.png     — /v1/embeddings vs sentence-transformers
  benchmark_impact.png        — % advantage summary across all metrics
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import os

OUT = os.path.dirname(__file__)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 10,
})

# ─── Canonical data ───────────────────────────────────────────────────────────
# (backend, device, scenario, req_s, tok_s, mean_ms, p50_ms, p95_ms, p99_ms)
# Source: results_full.md — post OPT-2 (2026-04-03)

DATA = [
    # CUDA short (n=40, concurrency=4) — post OPT-2
    ("infergo",  "CUDA", "short",  3.8, 200, 1053, 1185, 1451, 1472),
    ("python",   "CUDA", "short",  2.1, 133,  469,  465,  481,  486),
    # CUDA long (n=20, concurrency=1) — post OPT-2
    ("infergo",  "CUDA", "long",   0.6, 143, 1787, 1782, 1803, 1807),
    ("python",   "CUDA", "long",   0.5, 131, 1935, 1928, 1969, 1971),
    # CPU short (n=10, concurrency=1)
    ("infergo",  "CPU",  "short",  0.2,  10, 5462, 6494, 6571, 6571),
    ("python",   "CPU",  "short",  0.1,   5,12002,12491,14083,15292),
    # CPU long (n=5, concurrency=1)
    ("infergo",  "CPU",  "long",  0.04,  10,25990,25972,26218,26218),
    ("python",   "CPU",  "long",  0.04,  11,23733,23733,23918,23918),
]

# Pre-OPT-2 CUDA data (for delta chart) — source: results_full_cuda.md (2026-04-02)
DATA_PRE_OPT2 = [
    ("infergo_pre", "CUDA", "short", 2.7, 142, 1429, 1423, 1793, 1802),
    ("infergo_pre", "CUDA", "long",  0.6, 144, 1775, 1771, 1789, 1796),
]

COLD = [
    ("infergo", "CUDA",  451),
    ("python",  "CUDA",  494),
    ("infergo", "CPU",  6450),
    ("python",  "CPU",  9897),
]

# Embedding results — source: results_embedding.md (2026-04-03, OPT-4)
EMBED = [
    ("infergo /v1/embeddings\n(HTTP, concurrency=8)", 251.8, "#2563EB"),
    ("sentence-transformers\n(in-process, batch=32)", 3307.5, "#DC2626"),
]

COLORS = {
    "infergo":     "#2563EB",   # blue
    "python":      "#DC2626",   # red
    "infergo_pre": "#93C5FD",   # light blue (pre-OPT-2)
}
ALPHA = 0.88

# ─── Helpers ─────────────────────────────────────────────────────────────────

def get(device, scenario, backend="infergo", data=DATA):
    for r in data:
        if r[0] == backend and r[1] == device and r[2] == scenario:
            return r
    return None

def label_bar(ax, bar, fmt, *, offset_frac=0.015, fontsize=8.5):
    ymax = ax.get_ylim()[1]
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + ymax * offset_frac,
            fmt.format(h), ha="center", va="bottom", fontsize=fontsize, fontweight="bold")

def pct_badge(ax, x, v_new, v_ref, *, prefix="", good_positive=True, fontsize=9, y_frac=0.88):
    if not v_ref:
        return
    pct = (v_new - v_ref) / v_ref * 100
    sign = "+" if pct >= 0 else ""
    is_good = (pct >= 0) if good_positive else (pct <= 0)
    color = "#16a34a" if is_good else "#b91c1c"
    ax.text(x, ax.get_ylim()[1] * y_frac, f"{prefix}{sign}{pct:.0f}%",
            ha="center", va="center", fontsize=fontsize, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=color, lw=0.9))

def grid(ax):
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

# ─── Figure 1: Throughput (all scenarios) ────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("infergo vs llama-cpp-python — Throughput (post OPT-2)",
             fontsize=15, fontweight="bold", y=1.01)

scenarios = [
    ("CUDA short  (n=40, concurrency=4)", "CUDA", "short"),
    ("CUDA long   (n=20, concurrency=1)", "CUDA", "long"),
    ("CPU short   (n=10, concurrency=1)", "CPU",  "short"),
    ("CPU long    (n=5,  concurrency=1)", "CPU",  "long"),
]

for ax, (title, dev, sc) in zip(axes.flat, scenarios):
    ig = get(dev, sc, "infergo")
    py = get(dev, sc, "python")
    if not ig or not py:
        continue

    toks = [ig[4], py[4]]
    reqs = [ig[3], py[3]]
    x = np.array([0, 1])
    ax2 = ax.twinx()

    # tok/s — solid bars
    b_t_ig = ax.bar(x[0] - 0.22, toks[0], 0.38, color=COLORS["infergo"],  alpha=ALPHA, zorder=3, label="tok/s")
    b_t_py = ax.bar(x[1] - 0.22, toks[1], 0.38, color=COLORS["python"],   alpha=ALPHA, zorder=3)
    # req/s — hatched bars on twin axis
    b_r_ig = ax2.bar(x[0] + 0.22, reqs[0], 0.38, color=COLORS["infergo"], alpha=0.40, hatch="///", zorder=3, label="req/s")
    b_r_py = ax2.bar(x[1] + 0.22, reqs[1], 0.38, color=COLORS["python"],  alpha=0.40, hatch="///", zorder=3)

    # value labels
    for bar, v in [(b_t_ig[0], toks[0]), (b_t_py[0], toks[1])]:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.03, str(v),
                ha="center", fontsize=9, fontweight="bold")
    for bar, v in [(b_r_ig[0], reqs[0]), (b_r_py[0], reqs[1])]:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.03, f"{v:.2f}",
                 ha="center", fontsize=9, fontweight="bold", color="gray")

    tok_pct = (toks[0] - toks[1]) / toks[1] * 100 if toks[1] else 0
    req_pct = (reqs[0] - reqs[1]) / reqs[1] * 100 if reqs[1] else 0
    t_col = "#16a34a" if tok_pct >= 0 else "#b91c1c"
    r_col = "#16a34a" if req_pct >= 0 else "#b91c1c"

    ax.text(0.5, 0.93,
            f"tok/s  {'+' if tok_pct>=0 else ''}{tok_pct:.0f}%    req/s  {'+' if req_pct>=0 else ''}{req_pct:.0f}%",
            transform=ax.transAxes, ha="center", fontsize=9.5, fontweight="bold",
            color="#1d4ed8",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#1d4ed8", lw=0.9))

    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["infergo", "llama-cpp-python"], fontsize=9.5)
    ax.set_ylabel("tok/s", fontsize=9)
    ax2.set_ylabel("req/s", fontsize=9, color="gray")
    ax.set_ylim(0, max(toks) * 1.38)
    ax2.set_ylim(0, max(reqs) * 1.38 if max(reqs) > 0 else 1)
    grid(ax)

    p_ig = mpatches.Patch(color=COLORS["infergo"], label="infergo")
    p_py = mpatches.Patch(color=COLORS["python"],  label="llama-cpp-python")
    ax.legend(handles=[p_ig, p_py], fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "benchmark_throughput.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved benchmark_throughput.png")

# ─── Figure 2: Latency distribution P50 / P95 / P99 ─────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("infergo vs llama-cpp-python — Latency Distribution (P50 / P95 / P99)",
             fontsize=15, fontweight="bold", y=1.01)

for ax, (title, dev, sc) in zip(axes.flat, scenarios):
    ig = get(dev, sc, "infergo")
    py = get(dev, sc, "python")
    if not ig or not py:
        continue

    cats = ["P50", "P95", "P99"]
    ig_vals = [ig[6], ig[7], ig[8]]
    py_vals = [py[6], py[7], py[8]]
    x = np.arange(3)
    w = 0.32

    b_ig = ax.bar(x - w/2, ig_vals, w, color=COLORS["infergo"], alpha=ALPHA, label="infergo",          zorder=3)
    b_py = ax.bar(x + w/2, py_vals, w, color=COLORS["python"],  alpha=ALPHA, label="llama-cpp-python", zorder=3)

    def fmt_ms(v):
        return f"{v}ms" if v < 10000 else f"{v/1000:.1f}s"

    for bar, v in list(zip(b_ig, ig_vals)) + list(zip(b_py, py_vals)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.018,
                fmt_ms(v), ha="center", fontsize=7.5, fontweight="bold")

    # P50 spread annotation
    p50_pct = (ig_vals[0] - py_vals[0]) / py_vals[0] * 100 if py_vals[0] else 0
    p99_pct = (ig_vals[2] - py_vals[2]) / py_vals[2] * 100 if py_vals[2] else 0
    p50_col = "#b91c1c" if p50_pct > 0 else "#16a34a"
    p99_col = "#b91c1c" if p99_pct > 0 else "#16a34a"
    note = "batching queuing" if dev == "CUDA" and sc == "short" else ("lower latency" if p50_pct < 0 else "higher latency")
    ax.text(0.5, 0.93,
            f"P50 {'+' if p50_pct>=0 else ''}{p50_pct:.0f}%  |  P99 {'+' if p99_pct>=0 else ''}{p99_pct:.0f}%",
            transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold",
            color=p50_col,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=p50_col, lw=0.9))

    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylabel("Latency (ms)", fontsize=9)
    ymax = max(ig_vals + py_vals) * 1.32
    ax.set_ylim(0, ymax)
    grid(ax)
    ax.legend(fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "benchmark_latency.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved benchmark_latency.png")

# ─── Figure 3: Cold start ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.suptitle("infergo vs llama-cpp-python — Cold Start (time to first token)",
             fontsize=13, fontweight="bold")

labels = ["CUDA", "CPU"]
ig_cs  = [451,   6450]
py_cs  = [494,   9897]
x = np.arange(2)
w = 0.32

b_ig = ax.bar(x - w/2, ig_cs, w, color=COLORS["infergo"], alpha=ALPHA, label="infergo",          zorder=3)
b_py = ax.bar(x + w/2, py_cs, w, color=COLORS["python"],  alpha=ALPHA, label="llama-cpp-python", zorder=3)

for bar, v in list(zip(b_ig, ig_cs)) + list(zip(b_py, py_cs)):
    disp = f"{v}ms" if v < 2000 else f"{v/1000:.2f}s"
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.015,
            disp, ha="center", fontsize=11, fontweight="bold")

for i, (iv, pv) in enumerate(zip(ig_cs, py_cs)):
    pct = (iv - pv) / pv * 100
    sign = "+" if pct >= 0 else ""
    color = "#16a34a" if pct < 0 else "#b91c1c"
    ax.text(i, max(iv, pv) * 1.22, f"{sign}{pct:.0f}%", ha="center",
            fontsize=13, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.1))

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13)
ax.set_ylabel("Time to first token (ms)", fontsize=10)
ax.set_ylim(0, max(py_cs) * 1.42)
grid(ax)
ax.legend(fontsize=10)

# annotations
ax.annotate("No Python startup\nor GIL overhead", xy=(0 - w/2, ig_cs[0]),
            xytext=(-0.35, ig_cs[0] + 1500),
            arrowprops=dict(arrowstyle="->", color=COLORS["infergo"]),
            fontsize=8.5, color=COLORS["infergo"])
ax.annotate("No Python interpreter\n+ no GIL", xy=(1 - w/2, ig_cs[1]),
            xytext=(0.65, ig_cs[1] + 1800),
            arrowprops=dict(arrowstyle="->", color=COLORS["infergo"]),
            fontsize=8.5, color=COLORS["infergo"])

plt.tight_layout()
plt.savefig(os.path.join(OUT, "benchmark_coldstart.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved benchmark_coldstart.png")

# ─── Figure 4: OPT-2 before/after (CUDA) ─────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("OPT-2 Impact — Continuous Batching (CUDA only)\npre-OPT-2 vs post-OPT-2 vs llama-cpp-python",
             fontsize=13, fontweight="bold", y=1.03)

for ax, sc, title in zip(axes, ["short", "long"], ["Short prompts (n=40, concurrency=4)", "Long prompts (n=20, concurrency=1)"]):
    ig_pre = get("CUDA", sc, "infergo_pre", DATA_PRE_OPT2)
    ig_post = get("CUDA", sc, "infergo")
    py = get("CUDA", sc, "python")

    # tok/s comparison
    names   = ["infergo\n(pre-OPT-2)", "infergo\n(post-OPT-2)", "llama-cpp-python"]
    toks    = [ig_pre[4] if ig_pre else 0, ig_post[4], py[4]]
    reqs    = [ig_pre[3] if ig_pre else 0, ig_post[3], py[3]]
    bar_colors = [COLORS["infergo_pre"], COLORS["infergo"], COLORS["python"]]
    x = np.arange(3)
    w = 0.34

    ax2 = ax.twinx()
    b_t = ax.bar(x - w/2, toks, w, color=bar_colors, alpha=ALPHA, zorder=3)
    b_r = ax2.bar(x + w/2, reqs, w, color=bar_colors, alpha=0.40, hatch="///", zorder=3)

    for bar, v in zip(b_t, toks):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.03, str(v),
                ha="center", fontsize=9.5, fontweight="bold")
    for bar, v in zip(b_r, reqs):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.03, f"{v:.1f}",
                 ha="center", fontsize=9.5, fontweight="bold", color="gray")

    # arrows showing improvement
    if ig_pre and ig_post:
        tok_gain = (ig_post[4] - ig_pre[4]) / ig_pre[4] * 100
        req_gain = (ig_post[3] - ig_pre[3]) / ig_pre[3] * 100 if ig_pre[3] > 0 else float("inf")
        ax.annotate("", xy=(1 - w/2, toks[1] * 0.95), xytext=(0 - w/2, toks[0] * 0.95),
                    arrowprops=dict(arrowstyle="->", color="#16a34a", lw=2))
        ax.text(0.5, 0.76,
                f"OPT-2: tok/s +{tok_gain:.0f}%  |  req/s +{req_gain:.0f}%",
                transform=ax.transAxes, ha="center", fontsize=9.5, color="#16a34a", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#16a34a", lw=1))

    # vs python
    py_tok_pct = (ig_post[4] - py[4]) / py[4] * 100
    py_req_pct = (ig_post[3] - py[3]) / py[3] * 100
    ax.text(0.5, 0.64,
            f"vs Python: tok/s +{py_tok_pct:.0f}%  |  req/s +{py_req_pct:.0f}%",
            transform=ax.transAxes, ha="center", fontsize=9, color=COLORS["infergo"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLORS["infergo"], lw=0.9))

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("tok/s", fontsize=9)
    ax2.set_ylabel("req/s (hatched)", fontsize=9, color="gray")
    ax.set_ylim(0, max(toks) * 1.55)
    ax2.set_ylim(0, max(reqs) * 1.55 if max(reqs) > 0 else 1)
    grid(ax)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "benchmark_opt2_delta.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved benchmark_opt2_delta.png")

# ─── Figure 5: GPU vs CPU speedup ────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle("GPU vs CPU Speedup — infergo & llama-cpp-python",
             fontsize=13, fontweight="bold")

for ax, sc, title in zip(axes, ["short", "long"], ["Short prompts", "Long prompts"]):
    ig_gpu = get("CUDA", sc, "infergo")
    ig_cpu = get("CPU",  sc, "infergo")
    py_gpu = get("CUDA", sc, "python")
    py_cpu = get("CPU",  sc, "python")
    if not all([ig_gpu, ig_cpu, py_gpu, py_cpu]):
        continue

    categories = ["infergo", "llama-cpp-python"]
    gpu_toks   = [ig_gpu[4], py_gpu[4]]
    cpu_toks   = [ig_cpu[4], py_cpu[4]]
    x = np.arange(2)
    w = 0.32

    b_gpu = ax.bar(x - w/2, gpu_toks, w, color=["#2563EB", "#DC2626"], alpha=ALPHA,       label="CUDA (GPU)", zorder=3)
    b_cpu = ax.bar(x + w/2, cpu_toks, w, color=["#93C5FD", "#FCA5A5"], alpha=ALPHA + 0.1, label="CPU",        zorder=3)

    for bar, v in list(zip(b_gpu, gpu_toks)) + list(zip(b_cpu, cpu_toks)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.025,
                str(v), ha="center", fontsize=9.5, fontweight="bold")

    for i, (gv, cv) in enumerate(zip(gpu_toks, cpu_toks)):
        if cv > 0:
            speedup = gv / cv
            ax.text(i, max(gv, cv) * 1.25, f"GPU {speedup:.0f}× faster",
                    ha="center", fontsize=10, fontweight="bold", color="#7c3aed",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#7c3aed", lw=0.9))

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("tok/s", fontsize=9)
    ax.set_ylim(0, max(gpu_toks) * 1.5)
    grid(ax)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "benchmark_gpu_cpu.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved benchmark_gpu_cpu.png")

# ─── Figure 6: Embedding ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.suptitle("/v1/embeddings — infergo vs sentence-transformers (all-MiniLM-L6-v2, CPU)\nOPT-4 results",
             fontsize=12, fontweight="bold")

labels, vals, cols = zip(*EMBED)
x = np.arange(len(labels))
bars = ax.bar(x, vals, 0.45, color=cols, alpha=ALPHA, zorder=3)

for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.015,
            f"{v:.1f} req/s", ha="center", fontsize=11.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Requests / second", fontsize=10)
ax.set_ylim(0, max(vals) * 1.28)
grid(ax)

ax.annotate(
    "Gap is methodology, not model speed:\ninfego uses HTTP+concurrency=8 (real-world)\nsentence-transformers runs in-process\nwith batch=32 (no network overhead).\nCosine similarity = 1.000000 — output identical.",
    xy=(0, vals[0]), xytext=(0.38, 0.68),
    xycoords=("data", "data"), textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="#555"),
    fontsize=9, color="#374151",
    bbox=dict(boxstyle="round,pad=0.4", fc="#f9fafb", ec="#d1d5db", lw=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUT, "benchmark_embedding.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved benchmark_embedding.png")

# ─── Figure 7: Impact summary ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 6.5))
fig.suptitle("infergo vs llama-cpp-python — % Advantage Summary (post OPT-2/3/4)",
             fontsize=13, fontweight="bold")

metrics = [
    # (label, % value, is_latency_metric)
    ("CUDA short\ntok/s",       +50,  False),
    ("CUDA long\ntok/s",        +9,   False),
    ("CUDA short\nreq/s",       +81,  False),
    ("CUDA long\nreq/s",        +20,  False),
    ("CUDA cold\nstart",        -9,   True),   # lower = better
    ("CPU short\ntok/s",        +100, False),
    ("CPU short\nreq/s",        +100, False),
    ("CPU cold\nstart",         -35,  True),   # lower = better
    ("CUDA short\nP50 latency", +155, True),   # higher = worse (batching queuing)
]

labels, vals, is_lat = zip(*metrics)

colors = []
for lbl, v, is_l in metrics:
    if "cold start" in lbl or "P50" in lbl:
        colors.append("#16a34a" if v < 0 else "#b91c1c")
    else:
        colors.append("#16a34a" if v > 0 else "#b91c1c")

x = np.arange(len(labels))
bars = ax.bar(x, vals, color=colors, alpha=ALPHA, zorder=3, width=0.62)

for bar, v in zip(bars, vals):
    sign = "+" if v > 0 else ""
    offset = 3 if v >= 0 else -14
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height() + offset,
            f"{sign}{v}%", ha="center", fontsize=9.5, fontweight="bold",
            color=bar.get_facecolor())

ax.axhline(0, color="black", linewidth=0.9)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.8)
ax.set_ylabel("% change vs llama-cpp-python\n(positive = infergo wins)", fontsize=10)
ax.set_ylim(min(vals) * 1.35, max(vals) * 1.25)
grid(ax)

pos_p = mpatches.Patch(color="#16a34a", alpha=ALPHA, label="infergo advantage")
neg_p = mpatches.Patch(color="#b91c1c", alpha=ALPHA, label="infergo disadvantage")
ax.legend(handles=[pos_p, neg_p], fontsize=10, loc="upper left")

# P50 note
ax.annotate(
    "Expected: continuous batching\ninterleaves 4 requests per step.\nWill narrow with PagedAttention (OPT-22).",
    xy=(x[-1], vals[-1]), xytext=(x[-1] - 2.6, vals[-1] - 50),
    arrowprops=dict(arrowstyle="->", color="#b91c1c"),
    fontsize=8.5, color="#b91c1c")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "benchmark_impact.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved benchmark_impact.png")

print("\nAll plots saved to", OUT)
