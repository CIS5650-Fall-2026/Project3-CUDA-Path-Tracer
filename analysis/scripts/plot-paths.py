"""Plots analysis/data/toggles-paths.csv (sorting on, AA on) as img/paths_alive_per_bounce.png."""
import csv
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"
OPEN_C = "#2a78d6"
CLOSED_C = "#eb6834"

series = collections.defaultdict(dict)
with open("analysis/data/toggles-paths.csv") as f:
    for row in csv.DictReader(f):
        if row["sort"] == "1" and row["aa"] == "1":
            series[(row["scene"], row["compaction"])][int(row["bounce"])] = int(row["paths_alive"])

BORN = 640_000
bounces = list(range(0, 8))


def curve(scene):
    d = series[(scene, "1")]
    return [BORN] + [d[b] for b in range(1, 8)]


open_y = curve("open")
closed_y = curve("closed")

fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.axhline(BORN, color=INK_SOFT, lw=1.2, ls=(0, (4, 3)), alpha=0.55, zorder=1)
ax.text(-0.15, BORN + 12_000, "no compaction: all 640,000 threads launch every bounce",
        color=INK_SOFT, fontsize=8.5, va="bottom", ha="left")

for y, color in ((closed_y, CLOSED_C), (open_y, OPEN_C)):
    ax.plot(bounces, y, color=color, lw=2, marker="o", ms=5,
            mec=SURFACE, mew=1.5, zorder=3, solid_capstyle="round")

ax.text(7.25, closed_y[-1], "closed box" + chr(10) + f"{closed_y[-1] // 1000}k alive, {round(100 * closed_y[-1] / BORN)}%", color=INK,
        fontsize=10, ha="left", va="center", linespacing=1.4)
ax.text(7.25, open_y[-1], "open box" + chr(10) + f"{open_y[-1] // 1000}k alive, {round(100 * open_y[-1] / BORN)}%", color=INK,
        fontsize=10, ha="left", va="center", linespacing=1.4)

ax.set_xlim(-0.3, 9.75)
ax.set_ylim(0, 725_000)
ax.set_xticks(bounces)
ax.set_xticklabels(["start" if b == 0 else str(b) for b in bounces])
ax.set_yticks([0, 160_000, 320_000, 480_000, 640_000])
ax.set_yticklabels(["0", "160k", "320k", "480k", "640k"])
ax.set_xlabel("bounce", color=INK_SOFT, fontsize=10, labelpad=8)
ax.set_ylabel("paths still alive", color=INK_SOFT, fontsize=10, labelpad=8)
ax.set_title("Stream compaction removes far less work in a sealed room",
             color=INK, fontsize=12.5, pad=16, loc="left")

ax.grid(axis="y", color=GRID, lw=1, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK_SOFT, length=0, labelsize=9.5)

fig.tight_layout()
fig.savefig("img/paths_alive_per_bounce.png", facecolor=SURFACE)
print("wrote img/paths_alive_per_bounce.png")
