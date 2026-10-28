"""Reads pixel values across the grout lines of the bump test renders

Pixel values are the 0 to 255 numbers stored in the PNG, not radiance

usage, from the repo root:
  python analysis/scripts/measure-bump.py
  python analysis/scripts/measure-bump.py OFF.png BEFORE.png AFTER.png
"""
import sys

import numpy as np
from PIL import Image

DEFAULTS = [
    "img/bump_off_5000samp.png",
    "img/blooper/bump_light_leak.png",
    "img/bump_on_5000samp.png",
]
LABELS = ["bump off", "before fix", "after fix"]

# All regions are for scenes/bump/bump_on.json at 800x800: the back wall covers
# x 270 to 530 and y 270 to 530, four tiles each way

# a strip 30 px wide through the middle of the top-left tile, sampled every 2 rows
# across the first horizontal grout line
ROW_STRIP_X = (290, 320)
ROWS = range(322, 350, 2)

# a strip 30 px tall through the same tile, sampled every 2 columns across the
# first vertical grout line
COL_STRIP_Y = (290, 320)
COLS = range(324, 348, 2)


def load(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(float)


def row_profile(img):
    x0, x1 = ROW_STRIP_X
    return [round(img[y, x0:x1].mean()) for y in ROWS]


def col_profile(img, channel):
    y0, y1 = COL_STRIP_Y
    return [round(img[y0:y1, x, channel].mean()) for x in COLS]


def main():
    paths = sys.argv[1:4] if len(sys.argv) >= 4 else DEFAULTS
    images = [load(p) for p in paths]

    print("across a horizontal grout line, mean of R, G and B")
    print(f"{'row':<12}" + "".join(f"{y:>5}" for y in ROWS))
    for label, img in zip(LABELS, images):
        print(f"{label:<12}" + "".join(f"{v:>5}" for v in row_profile(img)))

    print("\nacross a vertical grout line, red and green separately")
    print(f"{'column':<12}" + "".join(f"{x:>5}" for x in COLS))
    for label, img in zip(LABELS, images):
        print(f"{label + ' R':<12}" + "".join(f"{v:>5}" for v in col_profile(img, 0)))
        print(f"{label + ' G':<12}" + "".join(f"{v:>5}" for v in col_profile(img, 1)))

    print("\nmean pixel value of the whole frame")
    for label, path, img in zip(LABELS, paths, images):
        print(f"{label:<12}{img.mean():6.2f}   {path}")


if __name__ == "__main__":
    main()
