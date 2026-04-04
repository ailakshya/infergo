"""Generate benchmark plots from bench_results.json."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

with open("bench_results.json") as f:
    data = json.load(f)

LABELS = {
    "native_bindings":   "infergo native\n(ctypes)",
    "http_openai_client":"infergo server\n(OpenAI SDK)",
    "core_python_urllib":"infergo server\n(urllib)",
    "infergo_cli":       "infergo server\n(Go CLI)",
    "llama_cpp_python":  "llama-cpp-\npython",
}

COLORS = {
    "native_bindings":   "#4C72B0",
    "http_openai_client":"#55A868",
    "core_python_urllib":"#64B5CD",
    "infergo_cli":       "#CCB974",
    "llama_cpp_python":  "#C44E52",
}

ORDER = ["native_bindings","http_openai_client","core_python_urllib","infergo_cli","llama_cpp_python"]

def get(metric, concurrency):
    return {r["name"]: r[metric] for r in data if r["concurrency"] == concurrency}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("5-Way Inference Benchmark — LLaMA 3 8B Q4 (RTX 5070 Ti)", fontsize=13, fontweight="bold")

# ── Plot 1: P50 latency at c=1 and c=4 ───────────────────────────────────────
ax = axes[0]
p50_c1 = get("p50_ms", 1)
p50_c4 = get("p50_ms", 4)
x = np.arange(len(ORDER))
w = 0.35
bars1 = ax.bar(x - w/2, [p50_c1[n] for n in ORDER], w, label="c=1",
               color=[COLORS[n] for n in ORDER], alpha=0.9)
bars2 = ax.bar(x + w/2, [p50_c4[n] for n in ORDER], w, label="c=4",
               color=[COLORS[n] for n in ORDER], alpha=0.45, hatch="//")
ax.set_xticks(x)
ax.set_xticklabels([LABELS[n] for n in ORDER], fontsize=8)
ax.set_ylabel("Latency P50 (ms)")
ax.set_title("P50 Latency")
ax.legend(fontsize=8)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7)

# ── Plot 2: Throughput req/s ──────────────────────────────────────────────────
ax = axes[1]
rps_c1 = get("req_per_s", 1)
rps_c4 = get("req_per_s", 4)
bars1 = ax.bar(x - w/2, [rps_c1[n] for n in ORDER], w, label="c=1",
               color=[COLORS[n] for n in ORDER], alpha=0.9)
bars2 = ax.bar(x + w/2, [rps_c4[n] for n in ORDER], w, label="c=4",
               color=[COLORS[n] for n in ORDER], alpha=0.45, hatch="//")
ax.set_xticks(x)
ax.set_xticklabels([LABELS[n] for n in ORDER], fontsize=8)
ax.set_ylabel("Throughput (req/s)")
ax.set_title("Throughput")
ax.legend(fontsize=8)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

# ── Plot 3: tok/s ─────────────────────────────────────────────────────────────
ax = axes[2]
tps_c1 = get("tok_per_s", 1)
tps_c4 = get("tok_per_s", 4)
bars1 = ax.bar(x - w/2, [tps_c1[n] for n in ORDER], w, label="c=1",
               color=[COLORS[n] for n in ORDER], alpha=0.9)
bars2 = ax.bar(x + w/2, [tps_c4[n] for n in ORDER], w, label="c=4",
               color=[COLORS[n] for n in ORDER], alpha=0.45, hatch="//")
ax.set_xticks(x)
ax.set_xticklabels([LABELS[n] for n in ORDER], fontsize=8)
ax.set_ylabel("Tokens / second")
ax.set_title("Token Throughput")
ax.legend(fontsize=8)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig("bench_plot.png", dpi=150, bbox_inches="tight")
print("Saved bench_plot.png")
