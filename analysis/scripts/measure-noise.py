"""Compares brightness and noise of two renders of the same scene

Pixel values are the 0 to 255 numbers stored in the PNG, not radiance
Noise is the mean absolute difference between horizontally adjacent pixels, which
is near zero on a smooth surface and grows with the grain

usage, from the repo root:
  python analysis/scripts/measure-noise.py
  python analysis/scripts/measure-noise.py A.png B.png
"""
import sys

import numpy as np
from PIL import Image

DEFAULTS = [
    "img/direct_off_depth2_100samp.png",
    "img/direct_on_depth2_100samp.png",
]

# regions for the Cornell box at 800x800, as (y0, y1, x0, x1), each on one flat surface
REGIONS = {
    "floor": (600, 720, 250, 550),
    "back wall": (300, 500, 300, 500),
    "ceiling": (60, 160, 250, 550),
}


def main():
    paths = sys.argv[1:3] if len(sys.argv) >= 3 else DEFAULTS
    print(f"{'image':<40}{'region':<12}{'mean':>8}{'noise':>8}")
    for path in paths:
        img = np.asarray(Image.open(path).convert("RGB")).astype(float)
        print(f"{path:<40}{'whole frame':<12}{img.mean():>8.2f}{'':>8}")
        for name, (y0, y1, x0, x1) in REGIONS.items():
            patch = img[y0:y1, x0:x1]
            noise = np.abs(np.diff(patch, axis=1)).mean()
            print(f"{'':<40}{name:<12}{patch.mean():>8.2f}{noise:>8.2f}")


if __name__ == "__main__":
    main()
