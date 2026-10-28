# Cuts one region out of a render and blows it up with nearest-neighbor sampling,
# so the pixel grid stays visible in side-by-side comparisons
#
#   python analysis/scripts/crop.py in.png out.png x0 y0 x1 y1 [scale]

import sys
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
x0, y0, x1, y1 = (int(v) for v in sys.argv[3:7])
scale = int(sys.argv[7]) if len(sys.argv) > 7 else 2

img = Image.open(src).crop((x0, y0, x1, y1))
img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
img.save(dst)
print(dst, img.size)
