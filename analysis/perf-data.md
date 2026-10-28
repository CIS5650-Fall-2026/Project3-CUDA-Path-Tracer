# Performance Data

This file records how every measured number in the README was taken and what the raw readings were. Sizes that come from the code, such as struct and texture sizes, are not measurements and are not here. Interpretation lives in the README.

## Setup

- GPU: NVIDIA GeForce RTX 3090 Ti, 24 GB, driver 616.56
- CPU: AMD Ryzen 9 5950X, 128 GB, Windows 11 Home
- Resolution 800x800 in every run, except the gray kitchen in `run-bvh.sh` and `run-nsys.sh`, which is 400x400, and `run-bvh-res.sh`, which lists its resolutions
- Timing comes from `PERF_LOG` in `src/perf.h`: host-side chrono around `pathtrace()`, printed as an average over each block of 100 iterations, plus a per-stage split from CUDA events recorded between the kernels
- Paths-alive counts are printed once, from iteration 1

Times are comparable inside one script run. Every number in the README was measured on 2026-09-30, after the last code change, so the few comparisons the README makes across scripts are between runs of the same day on the same build. Blocks of 100 iterations inside one run spread by about 0.5 to 1.6 ms, and by 4 to 5 ms for the two slowest cells of `run-toggles.sh`, so effects under about 5% are below what this method can resolve.

Every toggle is at its default in the scripts that do not vary it: everything on except `DIRECT_LIGHTING`, which is off.

Every script runs from the repo root, writes a CSV to `analysis/data/` and the raw console output to `analysis/logs/` (not committed). Scripts that edit `src/pathtrace.cu` or a scene file restore it on exit, including on Ctrl-C, but they do not rebuild afterwards, so the binary left in `build/` is the last configuration that ran. Rebuild before rendering.

| Script | Varies | Rebuilds | Output |
|---|---|---|---|
| `run-toggles.sh` | sorting, compaction, anti-aliasing, open/closed box | yes | `toggles-timing.csv`, `toggles-paths.csv` |
| `run-aabb.sh` | mesh AABB culling, triangle count | yes | `aabb-timing.csv` |
| `run-refraction.sh` | sphere material, open/closed box | no | `refraction-timing.csv` |
| `run-depth.sh` | trace depth, open/closed box | no | `depth-timing.csv` |
| `run-texture.sh` | wall material: plain, image, procedural | no | `texture-timing.csv` |
| `run-bump.sh` | bump on/off on procedural tiles | no | `bump-timing.csv` |
| `run-direct.sh` | direct lighting on/off, trace depth | yes | `direct-timing.csv` |
| `run-direct-depth.sh` | direct lighting on/off at several trace depths, noise of the renders | yes | `direct-depth-noise.txt` |
| `run-bvh.sh` | BVH on/off, triangle count, kitchen | yes | `bvh-timing.csv` |
| `run-bvh-res.sh` | BVH on/off at the same resolution, kitchen at 400 and 800 | yes | `bvh-res-timing.csv` |
| `run-bvh-tune.sh` | BVH leaf size and depth limit | no | `bvh-tune-timing.csv` |
| `run-specular.sh` | Phong exponent of the mirror sphere | no | `specular-timing.csv` |
| `run-dof.sh` | lens radius on the kitchen | no | `dof-timing.csv` |
| `run-nsys.sh` | per-stage kernel time under Nsight Systems, eleven configurations | yes | `nsys-*_cuda_gpu_kern_sum.csv`, `stage-timing.csv` |

To reproduce every number in the README, run every script in the table, in one session. The scripts that rebuild leave the binary in their last configuration, so run those first, then `cmake --build build --config Release`, then the ones that do not rebuild. Each script has its own section below with its command.

`run-perf.sh` and `run-aa.sh` measured subsets of `run-toggles.sh` in earlier, separate runs and are superseded by it; their CSVs are no longer cited.

## run-toggles.sh: sorting, compaction and anti-aliasing

```bash
bash analysis/scripts/run-toggles.sh
python analysis/scripts/plot-paths.py
```

- Scenes: `core/cornell.json` (open), `core/cornell_closed.json` (closed)
- 1000 iterations, depth 8. One build per combination of sorting and compaction with anti-aliasing on, and one more build with sorting off, compaction on and anti-aliasing off, which is run on the open box only
- Run on 2026-09-30; ten blocks of 100 iterations per row, the first dropped as warm-up, so each average is over nine blocks

| Scene | Sorting | Compaction | AA | avg ms/iteration | range | fps |
|---|---|---|---|---|---|---|
| open | on | on | on | 47.16 | 45.15 - 49.20 | 21.2 |
| open | on | off | on | 62.13 | 61.79 - 62.50 | 16.1 |
| open | off | on | on | 21.96 | 21.13 - 22.72 | 45.5 |
| open | off | off | on | 19.81 | 19.63 - 20.06 | 50.5 |
| open | off | on | off | 21.86 | 21.18 - 22.58 | 45.7 |
| closed | on | on | on | 80.84 | 78.30 - 83.62 | 12.4 |
| closed | on | off | on | 65.91 | 65.62 - 66.59 | 15.2 |
| closed | off | on | on | 38.42 | 37.84 - 38.99 | 26.0 |
| closed | off | off | on | 23.42 | 23.21 - 23.63 | 42.7 |

Paths alive after each bounce, iteration 1, compaction on (with compaction off every bounce reads 640,000):

| Bounce | open, sort on | open, sort off | open, sort off, AA off | closed, sort on | closed, sort off |
|---|---|---|---|---|---|
| 1 | 522,877 | 522,877 | 523,164 | 613,895 | 613,895 |
| 2 | 362,280 | 361,753 | 362,150 | 600,904 | 601,062 |
| 3 | 280,193 | 279,407 | 279,680 | 590,148 | 590,382 |
| 4 | 223,201 | 223,014 | 222,899 | 579,851 | 580,131 |
| 5 | 180,479 | 180,514 | 180,366 | 569,868 | 570,172 |
| 6 | 146,774 | 146,970 | 146,600 | 560,132 | 560,577 |
| 7 | 119,716 | 119,919 | 119,912 | 550,769 | 550,991 |

`analysis/scripts/plot-paths.py` turns the sort-on, AA-on columns of `toggles-paths.csv` into `img/paths_alive_per_bounce.png`.

## run-aabb.sh: mesh bounding-box culling

```bash
bash analysis/scripts/run-aabb.sh
```

- Scene: `mesh/suzanne.json` with `suzanne_4k.gltf` and `suzanne_16k.gltf` from `scenes/models/`
- 100 iterations, depth 8, compaction, sorting and anti-aliasing on
- Toggle: `MESH_AABB_CULL`; the script also forces `BVH` off for the run, since the BVH root box makes the per-mesh box redundant
- Run on 2026-09-30; one reading per row

| Model | Triangles | Culling | ms/iteration | fps |
|---|---|---|---|---|
| suzanne_4k | 3,936 | on | 274.72 | 3.6 |
| suzanne_4k | 3,936 | off | 365.39 | 2.7 |
| suzanne_16k | 15,744 | on | 1020.36 | 1.0 |
| suzanne_16k | 15,744 | off | 1396.48 | 0.7 |

Cross-check against the BVH-off rows of `run-bvh.sh`, run in the same session: suzanne_4k 285.57 against 274.72 ms (+3.9%), suzanne_16k 1073.98 against 1020.36 ms (+5.3%).

## run-refraction.sh: mirror versus glass

```bash
bash analysis/scripts/run-refraction.sh
```

- Scenes: `core/cornell.json` and `refraction/glass_open.json`, `core/cornell_closed.json` and `refraction/glass_closed.json`. Each pair differs only in the sphere's material
- 300 iterations, depth 8, all toggles at their defaults
- Run on 2026-09-30; three readings per row

| Scene | Material | Box | readings, ms/iteration | avg |
|---|---|---|---|---|
| cornell | mirror | open | 43.86, 44.71, 43.73 | 44.10 |
| glass_open | glass | open | 45.07, 45.48, 44.37 | 44.97 |
| cornell_closed | mirror | closed | 78.96, 77.84, 78.20 | 78.33 |
| glass_closed | glass | closed | 78.62, 77.73, 78.95 | 78.43 |

## run-depth.sh: trace depth

```bash
bash analysis/scripts/run-depth.sh
```

- Scenes: `refraction/glass_open.json`, `refraction/glass_closed.json`, with `DEPTH` overridden
- 300 iterations, all toggles at their defaults
- Run on 2026-09-30; three readings per row

| Scene | Depth | readings, ms/iteration | avg |
|---|---|---|---|
| glass_open | 8 | 44.81, 45.20, 43.97 | 44.66 |
| glass_open | 32 | 67.07, 68.57, 68.64 | 68.09 |
| glass_closed | 8 | 78.53, 78.98, 80.02 | 79.18 |
| glass_closed | 32 | 259.48, 260.29, 259.31 | 259.69 |

## run-texture.sh: plain versus image versus procedural

```bash
bash analysis/scripts/run-texture.sh
```

- Scenes: `texturing/tiles_plain.json`, `texturing/tiles_image.json`, `texturing/tiles_procedural.json`. The open Cornell box with all five walls sharing one material, so most hits on every bounce sample it
- The image is `scenes/textures/tiles_8x8.png`, 2048 x 2048, generated with the same tile count, grout width and colors as the procedural material
- 300 iterations, depth 8, all toggles at their defaults
- Run on 2026-09-30, bilinear sampling; three readings per row

| Scene | Wall material | readings, ms/iteration | avg |
|---|---|---|---|
| tiles_plain | flat color | 43.83, 44.53, 43.66 | 44.01 |
| tiles_image | image lookup | 44.33, 45.00, 44.37 | 44.57 |
| tiles_procedural | computed tiles | 43.96, 44.52, 43.63 | 44.04 |

## run-bump.sh: bump on versus off

```bash
bash analysis/scripts/run-bump.sh
```

- Scenes: `texturing/tiles_procedural.json` and `bump/tiles_bump.json`, identical except for `"BUMP": 1.0`
- 300 iterations, depth 8, all toggles at their defaults
- Run on 2026-09-30; three readings per row

| Scene | Bump | readings, ms/iteration | avg |
|---|---|---|---|
| tiles_procedural | off | 44.00, 44.64, 43.72 | 44.12 |
| tiles_bump | on | 43.38, 43.99, 43.14 | 43.50 |

## measure-bump.py: pixel values across the grout

```bash
python analysis/scripts/measure-bump.py
```

- Images: `img/bump_off_5000samp.png`, `img/blooper/bump_light_leak.png` (before the below-surface fix), `img/bump_on_5000samp.png` (after). All from `bump/bump_on.json` or `bump/bump_off.json`, 800x800, 5000 spp
- Values are the 0 to 255 numbers stored in the PNG, averaged over a strip 30 px wide. The light itself is clipped at 255
- Noise is about 1 at 5000 spp

Across the first horizontal grout line, mean of R, G and B, by image row:

| row | 322 | 324 | 326 | 328 | 330 | 332 | 334 | 336 | 338 | 340 | 342 | 344 | 346 | 348 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bump off | 48 | 48 | 49 | 48 | 49 | 31 | 31 | 30 | 30 | 50 | 50 | 51 | 51 | 51 |
| before fix | 47 | 47 | 48 | 37 | 24 | 31 | 29 | 30 | 29 | 49 | 48 | 50 | 50 | 50 |
| after fix | 48 | 48 | 48 | 41 | 31 | 31 | 30 | 30 | 30 | 53 | 52 | 50 | 50 | 51 |

Rows 328 to 330 are the bottom edge of the upper tile, facing away from the light. Rows 340 to 342 are the top edge of the lower tile, facing it.

Across the first vertical grout line, by image column:

| column | 324 | 326 | 328 | 330 | 332 | 334 | 336 | 338 | 340 | 342 | 344 | 346 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bump off R | 56 | 57 | 57 | 58 | 36 | 36 | 36 | 39 | 60 | 61 | 61 | 62 |
| bump off G | 49 | 50 | 50 | 51 | 31 | 32 | 32 | 35 | 54 | 55 | 56 | 56 |
| before fix R | 55 | 55 | 48 | 53 | 35 | 35 | 36 | 38 | 34 | 55 | 60 | 60 |
| before fix G | 48 | 48 | 52 | 54 | 31 | 31 | 32 | 33 | 21 | 46 | 55 | 54 |
| after fix R | 55 | 56 | 53 | 56 | 35 | 35 | 36 | 38 | 42 | 57 | 61 | 60 |
| after fix G | 48 | 49 | 62 | 58 | 31 | 31 | 32 | 33 | 24 | 47 | 55 | 55 |

Column 328 is the right edge of the left tile, facing the green wall. Column 340 is the left edge of the right tile, facing the red wall.

Mean pixel value of the whole frame: bump off 27.27, before fix 26.43, after fix 26.95.

## run-direct.sh: direct lighting on versus off

```bash
bash analysis/scripts/run-direct.sh
```

- Scenes: `lighting/small_light_depth2.json` and `lighting/small_light_depth8.json`. `core/cornell_diffuse.json` with the light scaled from 3 x 0.3 x 3 to 1 x 0.3 x 1 and its emittance raised from 5 to 45
- Toggle: `DIRECT_LIGHTING`
- 300 iterations, all other toggles at their defaults
- Run on 2026-09-30; three readings per row

| Scene | Direct lighting | readings, ms/iteration | avg |
|---|---|---|---|
| small_light_depth2 | on | 20.19, 19.78, 20.19 | 20.05 |
| small_light_depth2 | off | 20.31, 19.94, 20.22 | 20.16 |
| small_light_depth8 | on | 45.53, 47.08, 45.23 | 45.95 |
| small_light_depth8 | off | 46.03, 46.44, 45.28 | 45.92 |

Paths alive after each bounce at depth 8, iteration 1, same with the toggle on or off: 529,113, 371,972, 292,150, 236,567, 194,431, 161,321, 133,863.

## measure-noise.py: brightness and noise of the direct lighting renders

```bash
python analysis/scripts/measure-noise.py img/direct_off_depth2_100samp.png img/direct_on_depth2_100samp.png
python analysis/scripts/measure-noise.py img/direct_off_depth8_100samp.png img/direct_on_depth8_100samp.png
```

- Images rendered by hand from the two scenes above at 100 spp, once with `DIRECT_LIGHTING` 0 and once with 1
- Values are the 0 to 255 numbers stored in the PNG
- Noise is the mean absolute difference between horizontally adjacent pixels inside a region that lies on one flat surface

| Image | Region | mean | noise |
|---|---|---|---|
| depth 2, off | whole frame | 13.73 | |
| | floor | 29.77 | 46.77 |
| | back wall | 33.05 | 46.88 |
| | ceiling | 1.33 | 2.61 |
| depth 2, on | whole frame | 13.57 | |
| | floor | 30.12 | 4.92 |
| | back wall | 33.83 | 5.62 |
| | ceiling | 0.84 | 0.67 |
| depth 8, off | whole frame | 38.04 | |
| | floor | 64.51 | 77.75 |
| | back wall | 74.67 | 81.05 |
| | ceiling | 34.05 | 49.06 |
| depth 8, on | whole frame | 37.78 | |
| | floor | 64.25 | 77.43 |
| | back wall | 74.37 | 80.47 |
| | ceiling | 33.73 | 48.47 |

## run-direct-depth.sh: noise of direct lighting against trace depth

```bash
bash analysis/scripts/run-direct-depth.sh
```

- Scene: temporary copies of `lighting/small_light_depth8.json` with `DEPTH` set to 2, 3, 4, 5, 6 and 8, 800x800, 100 iterations, so 100 spp
- Toggle: `DIRECT_LIGHTING`, one build for each state; each render is compared with `measure-noise.py` (the regions and the noise measure are described under that script's section above)
- Run on 2026-09-30; one render per row; raw output in `analysis/data/direct-depth-noise.txt`, renders in `build/direct-depth-<depth>-<0|1>-100samp.png`
- The depth 2 and 8 rows reproduce the numbers of the section above to the second decimal, so the renders are repeatable

| Depth | Direct lighting | whole frame mean | floor mean | floor noise | back wall noise | ceiling noise | floor noise relative to the floor mean |
|---|---|---|---|---|---|---|---|
| 2 | off | 13.73 | 29.77 | 46.77 | 46.88 | 2.61 | 1.57 |
| 2 | on | 13.57 | 30.12 | 4.92 | 5.62 | 0.67 | 0.16 |
| 3 | off | 23.24 | 44.10 | 62.47 | 63.53 | 27.72 | 1.42 |
| 3 | on | 21.95 | 40.41 | 52.36 | 51.87 | 7.02 | 1.30 |
| 4 | off | 29.20 | 52.77 | 69.95 | 72.01 | 37.15 | 1.33 |
| 4 | on | 28.38 | 52.30 | 64.74 | 65.18 | 30.32 | 1.24 |
| 5 | off | 33.04 | 58.28 | 74.77 | 76.82 | 43.25 | 1.28 |
| 5 | on | 32.51 | 57.64 | 72.44 | 73.46 | 39.73 | 1.26 |
| 6 | off | 35.38 | 60.64 | 76.57 | 78.83 | 46.61 | 1.26 |
| 6 | on | 34.98 | 60.20 | 75.13 | 76.84 | 44.66 | 1.25 |
| 8 | off | 38.04 | 64.51 | 77.75 | 81.05 | 49.06 | 1.21 |
| 8 | on | 37.78 | 64.25 | 77.43 | 80.47 | 48.47 | 1.21 |

## run-bvh.sh: BVH on versus off

```bash
bash analysis/scripts/run-bvh.sh
```

- Scenes: `mesh/suzanne.json` with `suzanne_4k.gltf` and `suzanne_16k.gltf`, and `kitchen/kitchen_v0.json` (the gray-material kitchen, five models, 400x400)
- Toggle: `BVH`; off leaves `MESH_AABB_CULL` on, so the comparison is against one box per mesh
- 300 iterations, depth 8 in all three scenes, all other toggles at their defaults
- Run on 2026-09-30; three readings per row

| Scene | Triangles | BVH | readings, ms/iteration | avg |
|---|---|---|---|---|
| suzanne_4k | 3,936 | on | 48.83, 49.27, 48.05 | 48.72 |
| suzanne_4k | 3,936 | off | 282.63, 285.93, 288.14 | 285.57 |
| suzanne_16k | 15,744 | on | 51.74, 50.66, 50.74 | 51.05 |
| suzanne_16k | 15,744 | off | 1088.22, 1072.25, 1061.47 | 1073.98 |
| kitchen_v0 | 25,653 | on | 36.03, 35.44, 35.38 | 35.62 |
| kitchen_v0 | 25,653 | off | 307.21, 308.89, 310.85 | 308.98 |

BVH node counts printed at load: suzanne_4k 2,047, suzanne_16k 8,191, sink 4,095, glass 3,327, cutting board 2,047, lemon 2,047, spoon 4,095. Leaf size 4, `bvhMaxDepth` 24, never reached.

## run-bvh-res.sh: BVH on versus off at one resolution

```bash
bash analysis/scripts/run-bvh-res.sh
```

- Scenes: temporary copies of `mesh/suzanne.json` (with `suzanne_16k.gltf`, 800x800) and `kitchen/kitchen_v0.json` at 400x400 and at 800x800, so the real scenes are never edited
- Toggle: `BVH`; off leaves `MESH_AABB_CULL` on, so the comparison is against one box per mesh
- 300 iterations, depth 8, all other toggles at their defaults
- Run on 2026-09-30; three readings per row

| Scene | Resolution | Triangles | BVH | readings, ms/iteration | avg |
|---|---|---|---|---|---|
| suzanne_16k | 800x800 | 15,744 | on | 50.62, 51.03, 49.87 | 50.51 |
| suzanne_16k | 800x800 | 15,744 | off | 1015.03, 1026.86, 1024.70 | 1022.20 |
| kitchen_v0_400 | 400x400 | 25,653 | on | 35.23, 35.94, 35.58 | 35.58 |
| kitchen_v0_400 | 400x400 | 25,653 | off | 306.61, 307.77, 308.68 | 307.69 |
| kitchen_v0_800 | 800x800 | 25,653 | on | 100.04, 98.91, 98.93 | 99.29 |
| kitchen_v0_800 | 800x800 | 25,653 | off | 1074.43, 1086.09, 1081.66 | 1080.73 |

## run-bvh-tune.sh: BVH leaf size and depth limit

```bash
bash analysis/scripts/run-bvh-tune.sh
```

- Scenes: temporary copies of `mesh/suzanne.json` (with `suzanne_16k.gltf`, 800x800) and `kitchen/kitchen_v0.json` (400x400), each with the optional `"BVH"` key added, so the real scenes are never edited and nothing is rebuilt
- The settings are `LEAF_SIZE` and `MAX_DEPTH`, defaults 4 and 24
- One sweep per setting: the leaf size at 1, 2, 4, 8, 16, 32 with the depth limit at 24, and the depth limit at 6, 8, 10, 12 with the leaf size at 4
- 300 iterations, trace depth 8, all other toggles at their defaults
- Run on 2026-10-01; three readings per row; the node count is the sum over the meshes of a scene, printed by the build

| Scene | Leaf size | Depth limit | BVH nodes | readings, ms/iteration | avg |
|---|---|---|---|---|---|
| suzanne_16k | 4 | 24 | 8,191 | 51.47, 51.86, 51.52 | 51.62 |
| suzanne_16k | 1 | 24 | 31,487 | 51.35, 52.69, 54.31 | 52.78 |
| suzanne_16k | 2 | 24 | 16,383 | 51.77, 51.18, 50.43 | 51.13 |
| suzanne_16k | 8 | 24 | 4,095 | 52.95, 52.43, 52.09 | 52.49 |
| suzanne_16k | 16 | 24 | 2,047 | 56.51, 54.71, 56.91 | 56.04 |
| suzanne_16k | 32 | 24 | 1,023 | 63.51, 62.60, 61.90 | 62.67 |
| suzanne_16k | 4 | 6 | 127 | 140.10, 135.99, 132.50 | 136.20 |
| suzanne_16k | 4 | 8 | 511 | 72.37, 71.55, 72.84 | 72.25 |
| suzanne_16k | 4 | 10 | 2,047 | 54.73, 56.42, 56.13 | 55.76 |
| suzanne_16k | 4 | 12 | 8,191 | 52.64, 51.85, 49.89 | 51.46 |
| kitchen_v0 | 4 | 24 | 15,611 | 34.91, 35.87, 35.44 | 35.41 |
| kitchen_v0 | 1 | 24 | 51,301 | 34.15, 34.72, 35.35 | 34.74 |
| kitchen_v0 | 2 | 24 | 29,947 | 35.69, 35.53, 35.31 | 35.51 |
| kitchen_v0 | 8 | 24 | 8,187 | 34.82, 35.94, 37.38 | 36.05 |
| kitchen_v0 | 16 | 24 | 4,091 | 42.03, 41.93, 41.84 | 41.93 |
| kitchen_v0 | 32 | 24 | 2,043 | 42.84, 44.70, 45.88 | 44.47 |
| kitchen_v0 | 4 | 6 | 635 | 57.93, 58.08, 57.71 | 57.91 |
| kitchen_v0 | 4 | 8 | 2,555 | 39.88, 40.84, 39.86 | 40.19 |
| kitchen_v0 | 4 | 10 | 10,235 | 36.42, 36.35, 36.57 | 36.45 |
| kitchen_v0 | 4 | 12 | 15,611 | 36.65, 37.92, 37.40 | 37.32 |

The kitchen rows at leaf size 4 with the depth limit at 12 and at 24 build the same 15,611 nodes and read 37.32 and 35.41 ms, a 5.4% difference between runs of one tree.

## run-specular.sh: Phong exponent

```bash
bash analysis/scripts/run-specular.sh
```

- Scenes: `core/cornell.json` (perfect mirror) and `specular/specular_5000.json`, `specular_500.json`, `specular_50.json`, which are copies of it with `EXPONENT` on the sphere
- No toggle; the scenes are copied to temporary files with `ITERATIONS` set to 300 and deleted afterwards
- Run on 2026-09-30; three readings per row

| Exponent | readings, ms/iteration | avg |
|---|---|---|
| mirror | 44.08, 44.71, 43.63 | 44.14 |
| 5000 | 43.54, 42.73, 42.80 | 43.02 |
| 500 | 43.89, 44.40, 43.51 | 43.93 |
| 50 | 43.94, 44.43, 43.55 | 43.97 |

## run-dof.sh: lens radius

```bash
bash analysis/scripts/run-dof.sh
```

- Scene: `kitchen/kitchen_v1.json`, the final kitchen, 800x800, depth 16
- `LENS_RADIUS` edited in place to 0.09 and 0.0, restored on exit; `FOCAL_DISTANCE` 9.0 in both
- 300 iterations; run on 2026-09-30; three readings per row

| Lens radius | readings, ms/iteration | avg |
|---|---|---|
| 0.09 | 206.13, 206.03, 205.63 | 205.93 |
| 0.0 | 204.84, 204.80, 205.06 | 204.90 |

## run-nsys.sh: per-stage time under Nsight Systems

```bash
bash analysis/scripts/run-nsys.sh
python analysis/scripts/plot-stages.py
```

- Scenes: `core/cornell.json`, `core/cornell_closed.json` (800x800, depth 8), `kitchen/kitchen_v0.json` (400x400, depth 8) and `kitchen/kitchen_v1.json`, the final kitchen (800x800, depth 16)
- Configurations: every toggle at its default, with each of `SORT_BY_MATERIAL` and `STREAM_COMPACTION` off and both off for the two Cornell boxes, `BVH` off for the gray kitchen, and the final kitchen at the defaults; the script rebuilds for each
- One profiled run per configuration, no repeats, so a single row carries the usual few percent of noise
- `nsys profile --trace=cuda --sample=none`, 300 iterations. `nsys stats --report cuda_gpu_kern_sum` gives total time and launch count per kernel; `nsys-stages.py` maps kernel names to stages (the merge sort kernels to `sort`, the cub select and for_each kernels to `compact`) and divides by 300
- The wall-clock time per iteration (`host`) is the spacing between launches of `sendImageToPBO`, read from `cuda_gpu_trace`; the app's console output does not come through nsys on Windows
- Run on 2026-09-30

| Scene | Config | generate | intersect | sort | shade | compact | gather | kernels | host |
|---|---|---|---|---|---|---|---|---|---|
| cornell | all-on | 0.11 | 8.18 | 12.19 | 0.75 | 3.73 | 0.12 | 25.11 | 49.78 |
| cornell | no-sort | 0.11 | 8.25 | | 0.84 | 3.74 | 0.08 | 13.05 | 25.58 |
| cornell | no-compaction | 0.11 | 15.96 | 25.41 | 0.83 | | 0.13 | 42.46 | 66.22 |
| cornell | no-sort-no-compaction | 0.11 | 16.93 | | 1.27 | | 0.05 | 18.39 | 22.51 |
| cornell_closed | all-on | 0.12 | 18.05 | 23.55 | 1.46 | 7.14 | 0.19 | 50.52 | 81.53 |
| cornell_closed | no-sort | 0.12 | 18.26 | | 1.55 | 7.21 | 0.05 | 27.21 | 41.42 |
| cornell_closed | no-compaction | 0.12 | 19.46 | 25.50 | 1.47 | | 0.20 | 46.77 | 68.32 |
| cornell_closed | no-sort-no-compaction | 0.12 | 20.10 | | 1.65 | | 0.05 | 21.93 | 25.64 |
| kitchen_v0 | all-on | 0.03 | 13.09 | 5.98 | 0.40 | 1.61 | 0.02 | 21.14 | 36.41 |
| kitchen_v0 | no-bvh | 0.03 | 280.32 | 6.06 | 0.40 | 1.63 | 0.02 | 288.47 | 303.71 |
| kitchen_v1 | all-on | 0.19 | 103.52 | 38.65 | 2.99 | 12.00 | 0.19 | 157.57 | 210.08 |

`sendImageToPBO` is 0.01 to 0.02 ms in every row and left out. Per-kernel launch counts per iteration in the open box at the defaults: `computeIntersections` 8, `shadeMaterial` 8, merge sort 84 + 84 + 8, partition 8 + 8 + 8 + 8.

The wall-clock times against the same Cornell configurations without the profiler, from `run-toggles.sh`:

| Scene | Sorting | Compaction | Without profiler | Under nsys | Difference |
|---|---|---|---|---|---|
| open | on | on | 47.16 | 49.78 | +2.6 |
| open | on | off | 62.13 | 66.22 | +4.1 |
| open | off | on | 21.96 | 25.58 | +3.6 |
| open | off | off | 19.81 | 22.51 | +2.7 |
| closed | on | on | 80.84 | 81.53 | +0.7 |
| closed | on | off | 65.91 | 68.32 | +2.4 |
| closed | off | on | 38.42 | 41.42 | +3.0 |
| closed | off | off | 23.42 | 25.64 | +2.2 |

Host-side API time in the same run, from `nsys stats --report cuda_api_sum`, ms per iteration:

| Call | cornell all-on | cornell no-compaction | cornell no-sort |
|---|---|---|---|
| `cudaStreamSynchronize` | 26.81 (40 calls) | 38.05 (8) | 8.49 (32) |
| `cudaDeviceSynchronize` | 8.49 (18) | 17.05 (18) | 8.67 (18) |
| `cudaFree` | 2.96 (24) | 2.18 (8) | 1.55 (16) |
| `cudaLaunchKernel` | 2.63 (227) | 2.09 (219) | 1.10 (51) |
| `cudaMalloc` | 1.37 (24) | 0.74 (8) | 0.73 (16) |
| `cudaGLMapBufferObject` | 1.04 (1) | 1.04 (1) | 0.99 (1) |
| `cudaMemcpy` | 1.00 (1) | 1.02 (1) | 0.96 (1) |

`cuda_gpu_mem_time_sum` lists nine device-to-host copies per iteration in the open box run at the defaults, 0.76 ms in all, and eight memsets, 0.23 ms; eight of the copies are thrust reading back the partition's result inside the `cudaStreamSynchronize` rows, and only the image copy is a `cudaMemcpy` call of the app's own.

The synchronize rows are time the host spends waiting for kernels, so they overlap the kernel time and are not overhead on their own. The allocation, launch, map and copy rows are.

## Image comparisons

`analysis/scripts/compare-images.py` reports per-pixel differences between two renders:

```bash
python analysis/scripts/compare-images.py img/cornell_diffuse_5000samp.png img/REFERENCE_cornell.5000samp.png
python analysis/scripts/compare-images.py img/cornell_specular_5000samp.png img/cornell_specular_5000samp_no_compaction.png
```

| Pair | mean abs difference | other |
|---|---|---|
| diffuse render vs course reference | 2.13 / 255 | RMS 3.21, worst 4x4 region mean within 0.07% |
| compaction on vs off | 1.51 / 255 | global means equal to three decimals |

## One-off measurements

The toggles are the `#define`s at the top of [`src/pathtrace.cu`](../src/pathtrace.cu). Iteration count and trace depth are `ITERATIONS` and `DEPTH` in the scene JSON.
