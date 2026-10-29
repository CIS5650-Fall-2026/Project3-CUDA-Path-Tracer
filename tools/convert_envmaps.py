"""Converts the sky maps of the glTF research scenes into lat-long EXRs.

PBRT-v4 stores an image infinite light as a square equal-area octahedral map
(Clarberg 2008, "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD"),
and the research scenes keep the PBRT light as metadata: the glTF's
extras.pbrt.light_sources names the map ("source", relative to the scene's
source/ folder) and the light's transform ("pbrt_transform"). The renderer
reads lat-long maps only, so this script resamples each map once, with the
light's rotation baked in, and writes <map>.latlong.exr next to the
original. The glTF loader (src/scene/gltf_loader.cpp) looks for that file.

The lat-long layout is the one src/render/environment.h reads: rows top to
bottom, v = acos(d.y) / pi, u = 0.5 + atan2(d.z, d.x) / (2 pi), for a world
direction d (glTF axes, +Y up).

Usage, from the repository root:
    python tools/convert_envmaps.py [SCENES_DIR]
SCENES_DIR defaults to scenes/assets/gltf-research-scenes/scenes. Needs numpy
and OpenEXR (pip install OpenEXR).
"""

import json
import sys
from pathlib import Path

import numpy as np
import OpenEXR

DEFAULT_SCENES = Path(__file__).resolve().parent.parent / "scenes/assets/gltf-research-scenes/scenes"


def read_pfm(path):
    """A PFM image as float32 (height, width, 3), rows top to bottom."""
    with open(path, "rb") as f:
        kind = f.readline().strip()
        width, height = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4")
    channels = 3 if kind == b"PF" else 1
    image = data.reshape(height, width, channels)[::-1]  # PFM stores the bottom row first
    return np.repeat(image, 3, axis=2) if channels == 1 else image.astype(np.float32)


def read_exr(path):
    """An EXR's RGB channels as float32 (height, width, 3), rows top to bottom."""
    with OpenEXR.File(str(path)) as f:
        channels = f.channels()
        if "RGB" in channels:
            return channels["RGB"].pixels.astype(np.float32)
        if "RGBA" in channels:
            return channels["RGBA"].pixels[..., :3].astype(np.float32)
        raise ValueError(f"{path}: no RGB channels ({', '.join(channels)})")


def write_exr(path, image):
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, {"RGB": np.ascontiguousarray(image, dtype=np.float32)}) as f:
        f.write(str(path))


def sphere_to_square(d):
    """PBRT-v4's EqualAreaSphereToSquare for unit directions d (..., 3), with
    the exact arctangent where PBRT uses a polynomial fit. Returns u, v in
    [0, 1]; the upper hemisphere (+z) fills the center diamond."""
    x, y, z = np.abs(d[..., 0]), np.abs(d[..., 1]), np.abs(d[..., 2])
    r = np.sqrt(np.clip(1.0 - z, 0.0, None))
    a = np.maximum(x, y)
    b = np.minimum(x, y)
    b = np.where(a == 0.0, 0.0, b / np.where(a == 0.0, 1.0, a))
    phi = np.arctan(b) * 2.0 / np.pi
    phi = np.where(x < y, 1.0 - phi, phi)
    v = phi * r
    u = r - v
    lower = d[..., 2] < 0.0
    u, v = np.where(lower, 1.0 - v, u), np.where(lower, 1.0 - u, v)
    u = np.copysign(u, d[..., 0])
    v = np.copysign(v, d[..., 1])
    return 0.5 * (u + 1.0), 0.5 * (v + 1.0)


def octahedral_texels(image, x, y):
    """image[y, x] for texel indices up to one past the square's edges. Each
    edge of the equal-area square folds onto itself mirrored about its
    middle, so the texel past an edge is the one just inside it at the
    mirrored position (PBRT-v4's WrapMode::OctahedralSphere)."""
    n = image.shape[0]
    past = x < 0
    x, y = np.where(past, -x - 1, x), np.where(past, n - 1 - y, y)
    past = x >= n
    x, y = np.where(past, 2 * n - 1 - x, x), np.where(past, n - 1 - y, y)
    past = y < 0
    x, y = np.where(past, n - 1 - x, x), np.where(past, -y - 1, y)
    past = y >= n
    x, y = np.where(past, n - 1 - x, x), np.where(past, 2 * n - 1 - y, y)
    return image[y, x]


def bilerp(image, u, v):
    """Bilinear lookup at texture coordinates u, v (texel centers at
    (i + 0.5) / n), wrapping across the square's edges."""
    n = image.shape[0]
    x = u * n - 0.5
    y = v * n - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    top = (1.0 - fx) * octahedral_texels(image, x0, y0) + fx * octahedral_texels(image, x0 + 1, y0)
    bottom = (1.0 - fx) * octahedral_texels(image, x0, y0 + 1) + fx * octahedral_texels(image, x0 + 1, y0 + 1)
    return (1.0 - fy) * top + fy * bottom


def latlong_directions(width, height):
    """The world direction at the center of every lat-long texel, (height, width, 3)."""
    u, v = np.meshgrid((np.arange(width) + 0.5) / width, (np.arange(height) + 0.5) / height)
    theta = v * np.pi
    phi = (u - 0.5) * 2.0 * np.pi
    return np.stack([np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)], axis=-1)


def to_latlong(square, light_from_world):
    """Resamples an equal-area square map into a lat-long map with twice its
    texel count (n x n -> 2n x n), which keeps every lat-long texel smaller
    than a square texel, so bilinear lookups lose no detail."""
    n = square.shape[0]
    world = latlong_directions(2 * n, n)
    light = world @ light_from_world.T
    light /= np.linalg.norm(light, axis=-1, keepdims=True)
    u, v = sphere_to_square(light)
    return bilerp(square, u, v).astype(np.float32)


def mean_luminance_square(image):
    # Every texel of the equal-area square covers the same solid angle.
    return float((image @ [0.2126, 0.7152, 0.0722]).mean())


def mean_luminance_latlong(image):
    # A lat-long row covers solid angle in proportion to sin(theta).
    height = image.shape[0]
    weight = np.sin((np.arange(height) + 0.5) / height * np.pi)[:, None]
    return float(((image @ [0.2126, 0.7152, 0.0722]) * weight).sum() / (weight.sum() * image.shape[1]))


def infinite_lights(gltf_path):
    """The (map path, light-from-world rotation) of every image infinite light
    the file lists."""
    model = json.loads(gltf_path.read_text(encoding="utf-8"))
    sources = model.get("extras", {}).get("pbrt", {}).get("light_sources", [])
    for light in sources:
        if light.get("pbrt") != "infinite" or "source" not in light:
            continue
        # PBRT's Transform directive lists the matrix column by column: it is
        # the transpose of the 16 numbers read row by row. Its upper 3x3 takes
        # a light-space direction to the world.
        world_from_light = np.array(light.get("pbrt_transform", np.eye(4).ravel()), dtype=np.float64).reshape(4, 4).T
        light_from_world = np.linalg.inv(world_from_light[:3, :3])
        yield gltf_path.parent.parent / "source" / light["source"], light_from_world


def main():
    scenes = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SCENES
    if not scenes.is_dir():
        sys.exit(f"no scenes folder at {scenes}")
    done = {}
    for gltf in sorted(scenes.glob("*/gltf/*.gltf")):
        for source, light_from_world in infinite_lights(gltf):
            out = source.with_suffix(".latlong.exr")
            if out in done:
                if not np.allclose(done[out], light_from_world):
                    print(f"{gltf.name}: {source} is used with another rotation by an earlier file; kept that one")
                continue
            done[out] = light_from_world
            if not source.is_file():
                print(f"{gltf.parent.parent.name}: {source.name} is not in the download, skipped")
                continue
            square = read_pfm(source) if source.suffix.lower() == ".pfm" else read_exr(source)
            if square.shape[0] != square.shape[1]:
                print(f"{source}: {square.shape[1]}x{square.shape[0]} is not square, so not an equal-area map; skipped")
                continue
            latlong = to_latlong(square, light_from_world)
            write_exr(out, latlong)
            print(f"{gltf.parent.parent.name}: {source.name} {square.shape[1]}x{square.shape[0]} -> {out.name} "
                  f"{latlong.shape[1]}x{latlong.shape[0]}, mean luminance {mean_luminance_square(square):.5f} -> "
                  f"{mean_luminance_latlong(latlong):.5f}")


if __name__ == "__main__":
    main()
