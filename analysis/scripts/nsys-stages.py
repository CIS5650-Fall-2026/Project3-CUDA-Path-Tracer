"""Sums the per-kernel CSVs that run-nsys.sh leaves in analysis/data/ into stages,
divided by the iteration count, and writes analysis/data/stage-timing.csv. Two extra
rows per run: `kernels`, the sum of every stage, and `host`, the wall-clock ms/iteration
read off the kernel timeline; the gap between them is time the GPU sits idle

    python analysis/scripts/nsys-stages.py 300
"""
import csv
import glob
import os
import re
import sys

iters = int(sys.argv[1])

# kernel name fragment -> stage, first match wins. thrust's sort_by_key with a custom
# comparator is a cub merge sort (three kernels, one of them named Partition), and
# thrust::partition is a cub select plus two for_each copies
STAGES = [
    ("generateRayFromCamera", "generate"),
    ("computeIntersections", "intersect"),
    ("MergeSort", "sort"),
    ("shadeMaterial", "shade"),
    ("for_each", "compact"),
    ("Select", "compact"),
    ("CompactInit", "compact"),
    ("finalGather", "gather"),
    ("sendImageToPBO", "display"),
]

rows = []
for path in sorted(glob.glob("analysis/data/nsys-*_cuda_gpu_kern_sum.csv")):
    m = re.match(r"nsys-(.+?)-(.+)_cuda_gpu_kern_sum\.csv", os.path.basename(path))
    scene, config = m.group(1), m.group(2)
    totals = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Name"]
            stage = next((s for frag, s in STAGES if frag in name), None)
            if stage is None:
                print(f"{path}: unmapped kernel counted as other: {name[:80]}")
                stage = "other"
            totals[stage] = totals.get(stage, 0.0) + float(row["Total Time (ns)"])
    for stage, ns in sorted(totals.items()):
        rows.append((scene, config, stage, ns / 1e6 / iters))
    rows.append((scene, config, "kernels", sum(totals.values()) / 1e6 / iters))

    # wall-clock time per iteration: sendImageToPBO runs once per iteration, so the spacing
    # of its launches is the iteration period as the host experiences it
    trace = path.replace("_cuda_gpu_kern_sum.csv", "_cuda_gpu_trace.csv")
    if os.path.exists(trace):
        starts = []
        with open(trace, newline="") as f:
            for row in csv.DictReader(f):
                if "sendImageToPBO" in row["Name"]:
                    starts.append(float(row["Start (ns)"]))
        if len(starts) > 1:
            rows.append((scene, config, "host", (starts[-1] - starts[0]) / 1e6 / (len(starts) - 1)))

with open("analysis/data/stage-timing.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scene", "config", "stage", "ms_per_iter"])
    for scene, config, stage, ms in rows:
        w.writerow([scene, config, stage, f"{ms:.3f}"])
print("wrote analysis/data/stage-timing.csv")
for r in rows:
    print(*r)
