# Performance results (RTX 5070 Laptop GPU, Release, CUDA 13.3, 2026-10-05)

Times are per-iteration averages over 200 iterations (20 for the no-BVH runs, 10 for the 20 spp NEE runs), measured with CUDA events. Raw console output is in `raw_logs.txt`.
Total = intersect + sort + shade + compact (path-tracing kernels only).

## 1. Stream compaction (thrust::partition)
| Scene | Compaction | Intersect | Shade | Compact | Total |
|---|---|---|---|---|---|
| cornell_glass_closed (closed, 800x800) | on  | 2.88 | 3.84 | 12.71 | 19.43 |
| cornell_glass_closed (closed)          | off | 5.10 | 6.52 | 0     | 11.62 |
| bunny_test_diffuse (open, 1000x800)    | on  | 2.76 | 3.23 | 5.69  | 11.68 |
| bunny_test_diffuse (open)              | off | 12.03 | 4.57 | 0    | 16.60 |

Alive paths per bounce (iteration 10, compaction on):
- closed: 640000, 632847, 624233, 429588, 365286, 309871, 261963, 220702, 0
- open:   800000, 426018, 76583, 4577, 1104, 339, 133, 61, 40, 23, 14, 6, 0

## 2. Material sort (thrust::sort_by_key on intersections)
| Scene | Sort | Intersect | Sort | Shade | Compact | Total |
|---|---|---|---|---|---|---|
| cornell         | off | 1.39 | 0     | 1.92 | 9.13 | 12.45 |
| cornell         | on  | 1.53 | 35.88 | 2.06 | 9.32 | 48.78 |
| helmets_textured | off | 3.45 | 0     | 1.33 | 7.64 | 12.42 |
| helmets_textured | on  | 3.62 | 37.61 | 1.54 | 7.99 | 50.75 |

## 3. Russian roulette (from depth 3)
| Scene | RR | Intersect | Shade | Compact | Total |
|---|---|---|---|---|---|
| cornell        | on  | 1.39   | 1.92 | 9.13  | 12.45 |
| cornell        | off | 1.68   | 2.23 | 9.79  | 13.70 |
| scifi_corridor (1920x1080, mostly closed) | on  | 79.29  | 4.49 | 22.11 | 105.89 |
| scifi_corridor | off | 192.71 | 9.03 | 40.08 | 241.82 |

Alive paths per bounce, cornell: RR on 640000, 522801, 362469, 196098, 135034, 93970, 65608, 45905 / RR off 640000, 522801, 362469, 280287, 223378, 180546, 146730, 119493
Alive paths per bounce, corridor: RR on 2073600, 1855360, 1687172, 225935, 115491, 62067, 34641, 20111, 11784, 6924 / RR off 2073600, 1855360, 1687172, 1340229, 1133515, 964793, 840563, 742792, 660478, 589380

## 4. BVH on/off (no-BVH path still uses mesh bounding-box culling)
| Scene | BVH | Intersect | Shade (incl. NEE shadow rays) | Total |
|---|---|---|---|---|
| cornell_mesh (torus knot) | on  | 4.09    | 3.84    | 17.60 |
| cornell_mesh              | off | 154.86  | 169.21  | 337.22 |
| bunny_test_diffuse (69k tris) | on  | 2.76    | 3.23    | 11.68 |
| bunny_test_diffuse            | off | 1296.30 | 1161.85 | 2466.22 |

## 5. SAH vs midpoint split
| Scene | Split | Nodes | Build (ms) | Intersect |
|---|---|---|---|---|
| bunny_test_diffuse | SAH      | 43501 | 33.1 | 2.76 |
| bunny_test_diffuse | midpoint | 44817 | 14.8 | 3.17 |
| helmets_textured | SAH      | 10195 + 15655 | 11.4 + 14.4 | 3.45 |
| helmets_textured | midpoint | 10759 + 16127 | 3.5 + 5.3   | 3.96 |

## 6. Environment NEE + MIS (20 spp)
| Scene | Env NEE | Shade | Total | RMSE vs 5000 spp ref | High-freq noise |
|---|---|---|---|---|---|
| bunny_sunset (glass bunny) | on  | 2.51 | 15.21 | 0.0387 | 0.0464 |
| bunny_sunset               | off | 0.84 | 12.25 | 0.0477 | 0.0529 |
| bunny_test_diffuse (diffuse bunny) | on  | 3.21 | 12.37 | - | 0.0228 |
| bunny_test_diffuse                 | off | 0.78 | 10.43 | - | 0.0310 |
Images: ../img/cmp_env_nee.jpg, ../img/cmp_env_nee_zoom.jpg

## Measured earlier
- Mesh bbox culling (no BVH): 137.7 ms with culling vs 193.6 ms without
- Geom passed by reference: intersect 6.36 -> 4.27 ms
- Area-light NEE @20 spp: relative noise 0.41 vs 0.97 (~2.4x lower), mean 0.1381 vs 0.1365
- Glass NaN fix: intersect 702 ms -> 4.0 ms, 18% NaN rays per bounce before fix
- Bloom: 104.5 -> 108.1 ms/frame (~3.6 ms)
- Procedural vs file texture (helmets, shade): 1.10 vs 1.134 ms
- Normal mapping (helmets, shade, same build): 1.53 -> 1.62 ms with maps on; adding tangent fields to the intersection struct: 1.134 -> ~1.33 ms
