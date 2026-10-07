"""Bar graphs from the nsys cuda_gpu_kern_sum CSVs in profiling/<test>/.

Metric: total GPU kernel time / 1000 iterations = ms per iteration (lower is better).
Run from the repo root: python profiling/make_graphs.py
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
ITERS = 1000
COMMON = "800x800, 1000 spp, max depth 8, RTX 4060 Laptop, Release build"


def ms_per_iter(test):
    with open(ROOT / test / "kern_cuda_gpu_kern_sum.csv") as f:
        total_ns = sum(float(r["Total Time (ns)"]) for r in csv.DictReader(f))
    return total_ns / 1e6 / ITERS


def grouped(fname, title, groups, tests, labels, subtitle):
    """groups: x-axis categories; tests[i][j] = test folder for group i, series j."""
    fig, ax = plt.subplots(figsize=(6, 4))
    n = len(labels)
    width = 0.8 / n
    for j, label in enumerate(labels):
        xs = [i + (j - (n - 1) / 2) * width for i in range(len(groups))]
        ys = [ms_per_iter(tests[i][j]) for i in range(len(groups))]
        bars = ax.bar(xs, ys, width, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=2)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel("GPU time per iteration (ms)")
    ax.set_title(f"{title}\n(lower is better)")
    ax.margins(y=0.15)
    ax.legend()
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ROOT / fname, dpi=150)
    print(f"wrote {fname}")


if __name__ == "__main__":
    grouped(
        "graph_mis.png",
        "NEE + MIS on vs off",
        ["Cornell (open)"],
        [["cornell_mis_off", "cornell_mis_on"]],
        ["MIS off", "MIS on"],
        f"{COMMON}; compaction on, material sort off, RR on",
    )
    grouped(
        "graph_compaction.png",
        "Stream compaction: open vs closed Cornell box",
        ["Open", "Closed"],
        [
            ["cornell_compact_off", "cornell_compact_on"],
            ["cornell_closed_compact_off", "cornell_closed_compact_on"],
        ],
        ["Compaction off", "Compaction on"],
        f"{COMMON}; MIS on, material sort off, RR on",
    )
    grouped(
        "graph_sort.png",
        "Material sorting on vs off",
        ["Cornell (5 materials)", "Disney showcase"],
        [
            ["cornell_sort_off", "cornell_sort_on"],
            ["disney_showcase_sort_off", "disney_showcase_sort_on"],
        ],
        ["Sort off", "Sort on"],
        f"1000 spp, Release build; MIS on, compaction on, RR on. Cornell 800x800 depth 8; Disney 1920x1080, depth 12",
    )
