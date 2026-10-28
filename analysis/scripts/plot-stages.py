"""Plots analysis/data/stage-timing.csv as two stacked-bar figures, one bar per run:
img/stage_timing.png has every run; img/stage_timing_zoom.png leaves out the runs whose kernels
take over 100 ms, so the small ones can be read, and prints the stage times inside the bars."""
import csv
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"

STAGES = ["generate", "intersect", "sort", "shade", "compact", "gather", "display", "other"]
COLORS = {
    "generate": "#b9b7b1",
    "intersect": "#2a78d6",
    "sort": "#eb6834",
    "shade": "#3aa76d",
    "compact": "#e0b33a",
    "gather": "#7d6fc4",
    "display": "#d9d7d2",
    "other": "#000000",
}

all_runs = collections.OrderedDict()
with open("analysis/data/stage-timing.csv") as f:
    for row in csv.DictReader(f):
        all_runs.setdefault((row["scene"], row["config"]), {})[row["stage"]] = float(row["ms_per_iter"])


def draw(runs, path, title, label_segments):
    labels = [f"{scene}\n{config.replace('-no-', ',' + chr(10) + 'no-')}" for scene, config in runs]
    x = list(range(len(runs)))

    fig, ax = plt.subplots(figsize=(max(9.0, 1.1 * len(runs)), 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    top = max(max(r.get("kernels", 0.0), r.get("host", 0.0)) for r in runs.values())

    bottom = [0.0] * len(runs)
    for stage in STAGES:
        heights = [runs[k].get(stage, 0.0) for k in runs]
        ax.bar(x, heights, bottom=bottom, color=COLORS[stage], width=0.62, label=stage, zorder=3)
        if label_segments:
            for i, (b, h) in enumerate(zip(bottom, heights)):
                if h >= 0.035 * top:
                    ax.text(i, b + h / 2, f"{h:.1f}", ha="center", va="center", fontsize=7.5,
                            color="white" if stage in ("intersect", "sort", "shade", "gather") else INK,
                            zorder=5)
        bottom = [b + h for b, h in zip(bottom, heights)]

    # the host's wall-clock time per iteration is a tick above each stack; the gap between the
    # tick and the top of the stack is time the GPU spends idle: allocation, sync, launches
    for i, k in enumerate(runs):
        kernels = runs[k].get("kernels", 0.0)
        host = runs[k].get("host")
        if host is not None:
            ax.hlines(host, i - 0.31, i + 0.31, color=INK, lw=1.4, zorder=4)
            ax.text(i, host + 0.015 * top, f"{host:.1f} ms host" + chr(10) + f"{kernels:.1f} ms kernels",
                    color=INK, fontsize=7.5, ha="center", va="bottom", linespacing=1.3)
        else:
            ax.text(i, kernels + 0.015 * top, f"{kernels:.1f} ms", color=INK, fontsize=8.5, ha="center", va="bottom")

    ax.set_ylim(0, top * 1.14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5 if len(runs) < 9 else 7.4)
    ax.set_ylabel("ms per iteration", color=INK_SOFT, fontsize=10, labelpad=8)
    ax.set_title(title, color=INK, fontsize=12.5, pad=16, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=len(STAGES), loc="upper center", bbox_to_anchor=(0.5, -0.16))

    ax.grid(axis="y", color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0, labelsize=9.5)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


draw(all_runs, "img/stage_timing.png", "Where each iteration spends its time", False)

small = collections.OrderedDict((k, v) for k, v in all_runs.items() if v.get("kernels", 0.0) <= 100.0)
draw(small, "img/stage_timing_zoom.png", "The same, without the two slowest runs", True)
