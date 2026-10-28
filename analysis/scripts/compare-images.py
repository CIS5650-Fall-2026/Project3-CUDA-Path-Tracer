# Per-pixel comparison between two renders of the same scene.
# Source of the MAD / RMS / region-mean numbers quoted in the README.
#
# Usage:
#   python analysis/compare-images.py img/cornell_diffuse_5000samp.png img/REFERENCE_cornell.5000samp.png
#   python analysis/compare-images.py img/cornell_specular_5000samp.png img/cornell_specular_5000samp_no_compaction.png

import math
import sys

import numpy as np
from PIL import Image

GRID = 4  # region-mean check uses a GRID x GRID tiling


def main(path_a, path_b):
    a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.float64)
    b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        sys.exit(f"size mismatch: {a.shape} vs {b.shape}")

    diff = np.abs(a - b)
    print(f"A: {path_a}  global mean {a.mean():.3f}")
    print(f"B: {path_b}  global mean {b.mean():.3f}")
    print(f"mean absolute difference: {diff.mean():.2f} / 255")
    print(f"RMS difference:           {math.sqrt((diff ** 2).mean()):.2f} / 255")

    h, w = a.shape[0] // GRID, a.shape[1] // GRID
    worst = 0.0
    for i in range(GRID):
        for j in range(GRID):
            ra = a[i * h:(i + 1) * h, j * w:(j + 1) * w].mean()
            rb = b[i * h:(i + 1) * h, j * w:(j + 1) * w].mean()
            worst = max(worst, abs(ra - rb) / max(ra, 1e-9) * 100)
    print(f"worst {GRID}x{GRID} region mean difference: {worst:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__ or "usage: compare-images.py A.png B.png")
    main(sys.argv[1], sys.argv[2])
