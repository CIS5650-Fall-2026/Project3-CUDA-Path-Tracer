from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for index, name in enumerate(("open", "closed")):
    lines = (root / "img" / f"compaction-{name}.txt").read_text(encoding="utf-8-sig").splitlines()
    timings = {0: [], 1: []}
    counts = [320 * 320]
    reading_counts = False
    for line in lines:
        if line == "bounce,active_paths":
            reading_counts = True
        elif line and line[0].isdigit():
            values = line.split(",")
            if reading_counts:
                counts.append(int(values[1]))
            else:
                timings[int(values[1])].append(float(values[2]))
    axes[0].plot(range(len(counts)), counts, marker="o", label=name.title())
    means = [statistics.mean(timings[mode]) for mode in (0, 1)]
    for mode, mean in enumerate(means):
        bar = axes[1].bar(index + (mode - 0.5) * 0.35, mean, width=0.35,
                         color=("#8b9cad", "#309a83")[mode],
                         label=("Off", "On")[mode] if index == 0 else None)
        axes[1].bar_label(bar, fmt="%.2f", padding=3)
    print(f"{name}: off={means[0]:.3f} ms, on={means[1]:.3f} ms")
axes[0].set(title="Paths remaining after each bounce", xlabel="Bounce", ylabel="Active paths")
axes[0].legend()
axes[0].grid(alpha=0.2)
axes[1].set(title="Compaction time comparison", ylabel="Milliseconds per sample",
            xticks=[0, 1], xticklabels=["Open", "Closed"], ylim=(0, 17))
axes[1].legend(title="Compaction")
fig.suptitle("RTX 3070 / 320 x 320 / depth 12")
fig.tight_layout()
fig.savefig(root / "img" / "compaction-comparison.png", dpi=160)
