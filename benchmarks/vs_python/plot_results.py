"""
plot_results.py — visualize infergo vs llama-cpp-python benchmark results.

Generates:
  benchmark_throughput.png   — tok/s and req/s bars (GPU + CPU)
  benchmark_latency.png      — P50 / P99 latency bars (GPU + CPU)
  benchmark_coldstart.png    — cold-start comparison
  benchmark_impact.png       — % improvement summary (impact chart)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = os.path.dirname(__file__)

# ─── Raw data ────────────────────────────────────────────────────────────────

# (backend, device, scenario, req_s, tok_s, mean_ms, p50_ms, p95_ms, p99_ms)
DATA = [
    # CUDA short
    ("infergo",  "CUDA", "short",  2.7, 142, 1429, 1423, 1793, 1802),
    ("python",   "CUDA", "short",  2.1, 133,  469,  465,  481,  486),
    # CUDA long
    ("infergo",  "CUDA", "long",   0.6, 144, 1775, 1771, 1789, 1796),
    ("python",   "CUDA", "long",   0.5, 131, 1935, 1928, 1969, 1971),
    # CPU short
    ("infergo",  "CPU",  "short",  0.2,  10, 5462, 6494, 6571, 6571),
    ("python",   "CPU",  "short",  0.1,   5,12002,12491,14083,15292),
    # CPU long
    ("infergo",  "CPU",  "long",   0.04, 10,25990,25972,26218,26218),
    ("python",   "CPU",  "long",   0.04, 11,23733,23733,23918,23918),
]

COLD = [
    ("infergo", "CUDA",  456),
    ("python",  "CUDA",  494),
    ("infergo", "CPU",  6450),
    ("python",  "CPU",  9897),
]

COLORS = {"infergo": "#2563EB", "python": "#DC2626"}   # blue / red
ALPHA  = 0.88

def bar_pair(ax, x, vals, labels, width=0.35, colors=None):
    colors = colors or [COLORS[l] for l in labels]
    b0 = ax.bar(x - width/2, vals[0], width, color=colors[0], alpha=ALPHA, zorder=3)
    b1 = ax.bar(x + width/2, vals[1], width, color=colors[1], alpha=ALPHA, zorder=3)
    return b0, b1

def label_bars(ax, bars, fmt="{:.1f}", offset_frac=0.02):
    ymax = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + ymax*offset_frac,
                fmt.format(h), ha="center", va="bottom", fontsize=8.5, fontweight="bold")

def pct_label(ax, x, v_ig, v_py, ymax, above=True):
    if v_py == 0:
        return
    pct = (v_ig - v_py) / v_py * 100
    sign = "+" if pct >= 0 else ""
    color = "#16a34a" if pct >= 0 else "#b91c1c"
    y = ymax * (0.92 if above else 0.05)
    ax.text(x, y, f"{sign}{pct:.0f}%", ha="center", va="center",
            fontsize=9, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.8))

# ─── Figure 1: Throughput ─────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("infergo vs llama-cpp-python — Throughput", fontsize=14, fontweight="bold", y=1.01)

scenarios = [
    ("CUDA short", "CUDA", "short"),
    ("CUDA long",  "CUDA", "long"),
    ("CPU short",  "CPU",  "short"),
    ("CPU long",   "CPU",  "long"),
]

for ax, (title, dev, sc) in zip(axes.flat, scenarios):
    rows = {r[0]: r for r in DATA if r[1] == dev and r[2] == sc}
    ig, py = rows.get("infergo"), rows.get("python")
    if not ig or not py:
        continue

    x = np.array([0, 1])
    toks = [ig[4], py[4]]
    reqs = [ig[3], py[3]]

    ax2 = ax.twinx()
    b_toks_ig = ax.bar(x[0] - 0.2, toks[0], 0.35, color=COLORS["infergo"], alpha=ALPHA, label="tok/s", zorder=3)
    b_toks_py = ax.bar(x[1] - 0.2, toks[1], 0.35, color=COLORS["python"],  alpha=ALPHA, zorder=3)
    b_reqs_ig = ax2.bar(x[0] + 0.2, reqs[0], 0.35, color=COLORS["infergo"], alpha=0.45, hatch="//", label="req/s", zorder=3)
    b_reqs_py = ax2.bar(x[1] + 0.2, reqs[1], 0.35, color=COLORS["python"],  alpha=0.45, hatch="//", zorder=3)

    for bar, v in zip([b_toks_ig, b_toks_py], toks):
        ax.text(bar[0].get_x()+bar[0].get_width()/2, bar[0].get_height()*1.03,
                f"{v}", ha="center", fontsize=8.5, fontweight="bold")
    for bar, v in zip([b_reqs_ig, b_reqs_py], reqs):
        ax2.text(bar[0].get_x()+bar[0].get_width()/2, bar[0].get_height()*1.03,
                 f"{v:.1f}", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["infergo", "llama-cpp-python"], fontsize=9)
    ax.set_ylabel("tok/s", fontsize=9)
    ax2.set_ylabel("req/s", fontsize=9, color="gray")
    ax.set_ylim(0, max(toks)*1.25)
    ax2.set_ylim(0, max(reqs)*1.25 if max(reqs) > 0 else 1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # % delta on tok/s
    pct = (toks[0] - toks[1]) / toks[1] * 100 if toks[1] else 0
    sign = "+" if pct >= 0 else ""
    color = "#16a34a" if pct >= 0 else "#b91c1c"
    ax.text(0.5, 0.92, f"tok/s {sign}{pct:.0f}%", transform=ax.transAxes,
            ha="center", fontsize=10, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1))

    patch_ig = mpatches.Patch(color=COLORS["infergo"], label="infergo")
    patch_py = mpatches.Patch(color=COLORS["python"],  label="llama-cpp-python")
    ax.legend(handles=[patch_ig, patch_py], fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "benchmark_throughput.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved benchmark_throughput.png")

# ─── Figure 2: Latency ────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("infergo vs llama-cpp-python — Latency (P50 / P99)", fontsize=14, fontweight="bold", y=1.01)

for ax, (title, dev, sc) in zip(axes.flat, scenarios):
    rows = {r[0]: r for r in DATA if r[1] == dev and r[2] == sc}
    ig, py = rows.get("infergo"), rows.get("python")
    if not ig or not py:
        continue

    cats = ["P50", "P99"]
    ig_vals = [ig[5], ig[8]]
    py_vals = [py[5], py[8]]
    x = np.arange(len(cats))

    b_ig = ax.bar(x - 0.2, ig_vals, 0.35, color=COLORS["infergo"], alpha=ALPHA, label="infergo", zorder=3)
    b_py = ax.bar(x + 0.2, py_vals, 0.35, color=COLORS["python"],  alpha=ALPHA, label="llama-cpp-python", zorder=3)

    for bar, v in zip(list(b_ig)+list(b_py), ig_vals+py_vals):
        unit = "ms" if v < 10000 else "s"
        disp = f"{v}ms" if v < 10000 else f"{v/1000:.1f}s"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                disp, ha="center", fontsize=7.5, fontweight="bold")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Latency (ms)", fontsize=9)
    ymax = max(ig_vals+py_vals)*1.3
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)

    # P50 delta annotation
    pct = (ig_vals[0] - py_vals[0]) / py_vals[0] * 100 if py_vals[0] else 0
    sign = "+" if pct >= 0 else ""
    color = "#b91c1c" if pct >= 0 else "#16a34a"   # higher latency = red
    note = "higher latency\n(mutex queuing)" if pct > 0 else "lower latency"
    ax.text(0.5, 0.92, f"P50 {sign}{pct:.0f}%\n({note})", transform=ax.transAxes,
            ha="center", fontsize=8, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1))

plt.tight_layout()
plt.savefig(os.path.join(OUT, "benchmark_latency.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved benchmark_latency.png")

# ─── Figure 3: Cold start ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("infergo vs llama-cpp-python — Cold Start Latency", fontsize=13, fontweight="bold")

labels = ["CUDA", "CPU"]
ig_cs  = [456,  6450]
py_cs  = [494,  9897]
x = np.arange(len(labels))

b_ig = ax.bar(x - 0.2, ig_cs, 0.35, color=COLORS["infergo"], alpha=ALPHA, label="infergo", zorder=3)
b_py = ax.bar(x + 0.2, py_cs, 0.35, color=COLORS["python"],  alpha=ALPHA, label="llama-cpp-python", zorder=3)

for bar, v in zip(list(b_ig)+list(b_py), ig_cs+py_cs):
    disp = f"{v}ms" if v < 2000 else f"{v/1000:.2f}s"
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
            disp, ha="center", fontsize=10, fontweight="bold")

for i, (iv, pv) in enumerate(zip(ig_cs, py_cs)):
    pct = (iv - pv) / pv * 100
    sign = "+" if pct >= 0 else ""
    color = "#b91c1c" if pct > 0 else "#16a34a"
    ax.text(i, max(iv, pv)*1.18, f"{sign}{pct:.0f}%", ha="center",
            fontsize=11, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1))

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Time to first response (ms)", fontsize=10)
ax.set_ylim(0, max(py_cs)*1.35)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "benchmark_coldstart.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved benchmark_coldstart.png")

# ─── Figure 4: Impact summary ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("infergo vs llama-cpp-python — % Impact (infergo advantage)", fontsize=13, fontweight="bold")

metrics = [
    ("CUDA\nshort tok/s",   +7),
    ("CUDA\nlong tok/s",   +10),
    ("CUDA\nshort req/s",  +29),
    ("CUDA\nlong req/s",   +20),
    ("CUDA\ncold start",    -8),   # negative = infergo is faster (lower)
    ("CPU\nshort tok/s",  +100),
    ("CPU\nshort req/s",  +100),
    ("CPU\ncold start",   -35),
    ("P50 latency\n(CUDA short)", +206),  # infergo P50 3x higher due to mutex
]
# Note: latency higher for infergo is a disadvantage (mutex queuing)

labels, vals = zip(*metrics)
colors = ["#16a34a" if v > 0 else "#b91c1c" for v in vals]
# For latency metrics, invert the "good" direction
latency_metrics = {"P50 latency\n(CUDA short)"}
colors = []
for lbl, v in metrics:
    if "latency" in lbl.lower():
        colors.append("#b91c1c" if v > 0 else "#16a34a")  # higher latency = bad
    elif "cold start" in lbl.lower():
        colors.append("#16a34a" if v < 0 else "#b91c1c")  # lower cold start = good
    else:
        colors.append("#16a34a" if v > 0 else "#b91c1c")

x = np.arange(len(labels))
bars = ax.bar(x, vals, color=colors, alpha=ALPHA, zorder=3, width=0.6)

for bar, v in zip(bars, vals):
    sign = "+" if v > 0 else ""
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height() + (3 if v >= 0 else -12),
            f"{sign}{v}%", ha="center", fontsize=9.5, fontweight="bold",
            color=bar.get_facecolor())

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("% change vs llama-cpp-python\n(positive = infergo wins)", fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

pos_patch = mpatches.Patch(color="#16a34a", alpha=ALPHA, label="infergo advantage")
neg_patch = mpatches.Patch(color="#b91c1c", alpha=ALPHA, label="infergo disadvantage")
ax.legend(handles=[pos_patch, neg_patch], fontsize=10, loc="upper left")

# Annotation for P50 latency bar
ax.annotate("Mutex serialization:\nconcurrent requests queue,\nraising observed P50",
            xy=(x[-1], vals[-1]), xytext=(x[-1]-1.8, vals[-1]-60),
            arrowprops=dict(arrowstyle="->", color="#b91c1c"),
            fontsize=8.5, color="#b91c1c")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "benchmark_impact.png"), dpi=150, bbox_inches="tight")
plt.close()
print("saved benchmark_impact.png")

print("\nAll plots saved to", OUT)
