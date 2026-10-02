# Implementation and configuration

`generateRayFromCamera` creates camera rays, and `computeIntersections` finds their surface hits. The BSDF kernel in `pathtraceExtensions.cuh` handles light scattering; `finalGather` accumulates the result, and `sendImageToPBO` sends it to the preview. Hemisphere sampling accepts explicit Halton samples. Intersection helpers in `geometry.h` compute surface points, object motion, and entry/exit metadata.

All toggles live in the scene's optional `Renderer` JSON object. Missing settings use the defaults below. Change the scene and rerun to compare modes; toggles are not interactive UI controls. `scenes/gallery.json` is a complete example.

| Setting | Default | Meaning |
| --- | --- | --- |
| `ANTIALIASING` | `true` | Sample within each pixel; disabled mode samples its center. |
| `MATERIAL_SORTING` | `true` | Sort material IDs and gather paths/intersections together before shading. |
| `RUSSIAN_ROULETTE` / `RR_DEPTH` | `true` / `3` | Start stochastic termination after this many scattering events. |
| `REFRACTION` | `true` | Enable dielectric scattering; disabled glass is shaded as diffuse. |
| `DEPTH_OF_FIELD` | `true` | Use the camera aperture; zero aperture remains a pinhole. |
| `DIRECT_LIGHTING` | `true` | Sample an emissive object at diffuse vertices and use MIS. |
| `SAMPLER` / `SEED` | `"halton"` / `1` | Digit-scrambled, rotated Halton or `"random"` independent hash samples. |
| `MOTION_BLUR` | `true` | Sample shutter time; disabled mode freezes at shutter opening. |
| `COMPACTION` | `true` | Compact paths after shading; disabled mode retains the full buffer. |
| `SHARED_COMPACTION` | `true` | Shared-memory hierarchical scan; false selects Thrust `remove_if`. |
| `MESH_CULLING` | `true` | Check each mesh's object-space bounding box before its triangles. |
| `POSTPROCESS` | `false` | Enable exposure, bloom, Reinhard tone mapping, and gamma. |
| `EXPOSURE` / `GAMMA` | `0` / `2.2` | Exposure in stops and output gamma. |
| `BLOOM_STRENGTH` / `BLOOM_THRESHOLD` / `BLOOM_RADIUS` | `0.15` / `1` / `6` | Linear bright-pass settings; radius is 0–32 pixels. |

Camera settings include `APERTURE` (world-space aperture **radius**, default 0), `FOCAL_DISTANCE` (distance along camera view to the focus plane, default eye-to-lookAt distance), and `SHUTTER_OPEN`/`SHUTTER_CLOSE` (default 0/0). `FOVY` is a vertical **half-angle**. Camera resolution and depth are validated; supported depth is 1–64 intersection events.

## Light transport

**Diffuse and reflective materials.** `Diffuse` uses cosine-weighted hemisphere sampling and multiplies throughput by albedo. `Specular` is an ideal mirror. Emission is two-sided, and an emitter hit terminates the path. Paths store throughput separately from accumulated radiance, so terminating or compacting a path cannot discard its earlier direct-light contributions. The finite depth limit remains a truncation of the infinite transport solution.

**Russian roulette.** After `RR_DEPTH`, survival probability is the maximum throughput component, adjusted for refractive eta scaling and clamped to [0.05, 0.95]. A survivor divides throughput by its probability; a terminated path retains its radiance. This makes the roulette step unbiased relative to the same finite-depth integrator. The eta adjustment prevents temporary radiance scaling inside glass from causing excessive termination.

**Refraction.** Use `{"TYPE":"Glass","RGB":[0.99,0.99,0.99],"IOR":1.5}`; `Refractive` is an alias. Entry/exit orientation comes from outward geometric normals. Schlick Fresnel selects reflection versus transmission, Snell's law sets the transmitted direction, and total internal reflection forces reflection. Transmitted radiance includes the eta-squared factor, which cancels on exit through a matching interface. RGB is a per-event tint, not distance-dependent absorption. The implementation assumes isolated air/dielectric boundaries, not nested or overlapping media. Glass meshes need closed surfaces and outward winding. Rough specular lobes and dispersion are not implemented; `ROUGHNESS` on existing specular scenes is not used.

**Direct lighting.** At each diffuse vertex that can still reach another surface, choose one light uniformly and sample its area. A visibility ray checks for occlusion. Cube faces are chosen in proportion to transformed area; spheres include the nonuniform-scale area Jacobian; mesh triangles are chosen uniformly with the corresponding triangle-area PDF. PDFs include the light-selection probability and area-to-solid-angle conversion. Power-heuristic MIS weights both explicit light samples and subsequent BSDF-sampled emitter hits, avoiding doubled emission. Delta paths retain full emitter contribution. Shadow rays use the same shutter time as the camera path. Glass blocks ordinary shadow rays; refracted light paths are still found by BSDF tracing, with potentially high variance in caustics.

## Camera and samples

**Depth of field.** Sample a disk with square-root radial mapping to make aperture area uniform. Intersect the pinhole ray with a plane perpendicular to camera view at `FOCAL_DISTANCE`, then aim the sampled aperture ray at that point. Objects on the plane remain focused; nearer/farther objects blur. This is an ideal thin-lens model, not a multi-element lens simulation.

**Better sample sequences.** `sampling.h` maps pixel ID, iteration, seed, and dimension to a sample without mutable RNG state. Halton dimensions use distinct prime bases, bijective affine digit permutations per pixel/dimension/digit, and a per-pixel Cranley–Patterson rotation. Zero digits are scrambled too. Camera dimensions and eight dimensions per bounce are reserved, so toggling features does not accidentally shift the other sample streams. This is digit scrambling, not full nested Owen scrambling. Independent hash samples provide a baseline. The measured diffuse-scene error comparison is in [PERFORMANCE.md](PERFORMANCE.md); no sequence is claimed to win on every integrand.

**Motion blur.** An object's optional `VELOCITY:[vx,vy,vz]` is world-space translation per shutter-time unit. Its position at time t is the JSON transform plus `VELOCITY*t`. A camera ray samples the shutter interval once and carries that time through every bounce and shadow query. Bounds are checked after subtracting the same translation, so moving meshes remain correctly culled. This supports linear object translation, not rotation, deformation, or camera motion during exposure.

## Geometry and path scheduling

**OBJ loading.** Use an object with `TYPE:"obj"`, `FILE:"meshes/prism.obj"`, and the usual material and transform fields. Paths resolve relative to the scene JSON. The loader accepts position indices, slash-form face tokens, positive and negative indices, triangles, and simple planar polygons. Ear clipping handles concave polygons. Group/material directives, UVs, and supplied vertex normals are ignored: the JSON assigns one material to the whole object and rendering uses flat geometric normals. Malformed indices and degenerate/self-intersecting polygons are rejected. Positive, nonzero scale is required to preserve outward winding. This is OBJ polygon geometry support, not texture/MTL support.

Each mesh has an object-space AABB. Rays that miss it skip all triangle intersection checks. Rays that pass it still traverse triangles linearly; there is no BVH. Local ray directions are not renormalized, preserving the world ray parameter under nonuniform transforms. Spheres and cubes use the same exact hit-point/orientation convention in `geometry.h`.

**Shared-memory stream compaction.** `StreamCompaction::Efficient::compact` in `stream_compaction/efficient.cu` supports both integer arrays and device-resident path records. Each 256-thread block scans 512 alive flags using a Blelloch up-sweep/down-sweep in shared memory. Block totals are scanned recursively; offsets are added before scattering whole path records into a separate buffer. Padding handles partial blocks. This is a stable, work-efficient multi-block implementation with linear total work. Scratch buffers are allocated once. All-dead and empty arrays are handled. Before compaction, terminated paths save their radiance as output color in a separate per-pixel path buffer; `finalGather` accumulates that buffer exactly once after the bounce loop.

Material sorting first sorts compact indices by material ID, then gathers both paths and intersections into matching contiguous arrays. This avoids sorting a large tuple that exceeded CUB's static shared-memory limit on the NVIDIA T1000 used for the recorded runs. RNG keys use persistent pixel IDs, so reordering does not change samples.

## Display and persistence

**Final-image processing.** The completed-ray radiance buffer is averaged, thresholded, blurred with horizontal and vertical Gaussian passes, and combined with the original signal. Exposure, Reinhard tone mapping, and gamma follow. Preview and PNG use the same display buffer. HDR, raw output, and checkpoints retain unprocessed radiance; post-processing never feeds back into the estimator. The filter is screen-space bloom, not a denoiser. Radius and threshold changes have predictable screen-space effects and do not simulate lens scattering.

**Restartable rendering.** `--checkpoint FILE` selects a destination, `--checkpoint-every N` saves periodically, and `--resume FILE` resumes. Headless completion, interactive completion, Escape, and C save checkpoints. A checkpoint contains the completed sample count, actual camera pose/basis, raw float accumulation, scene/settings fingerprint, format version, and checksum. Scene geometry and mesh bounds are reloaded/reconstructed from the unchanged source files rather than storing a second copy. The fingerprint includes mesh vertex contents, so changed geometry is rejected.

Restoration validates the entire file before changing accumulation. The image target and iteration count may change; rendering settings and scene content must match. Samples are regenerated from the stored count and stable sample keys, yielding bit-identical split versus uninterrupted rendering on the same build/device. Checkpoints are taken between samples, not halfway through a bounce. Writes use a temporary sibling file and atomic replacement; failed writes leave the prior destination intact. The version-1 float payload assumes the same native byte order/float representation. File checksums detect accidental corruption, not adversarial modification, and arbitrary power-loss durability is not guaranteed.

## GPU versus a hypothetical CPU and future improvements

| Feature | GPU behavior / CPU comparison | Further optimization |
| --- | --- | --- |
| Roulette | Less surviving work, but divergent termination and compaction overhead. A CPU benefits immediately from shorter paths without GPU queue overhead. | Tune start depth and compact only when enough work has terminated. |
| Refraction | Many independent glass paths parallelize well; mixed reflection/transmission diverges. A CPU avoids warp divergence but traces far fewer rays concurrently. | Separate material kernels, nested-medium tracking, better caustic sampling. |
| Depth of field | A few arithmetic operations per primary ray, easily parallel. A CPU pays the same operation count per ray. | Better aperture sampling, specialized pinhole/lens kernels. |
| Direct lighting | Extra shadow intersections benefit from ray parallelism but increase geometry traffic. CPU traversal has better cache flexibility with fewer concurrent rays. | BVH, light-importance selection, batched shadow queues. |
| Halton | Deterministic parallel samples need no RNG-state buffers; prime-base divisions and scrambling cost more than hashing. A CPU may hide this cost behind expensive traversal. | Precomputed permutations or a vetted Sobol direction table and scrambling scheme. |
| Motion blur | Time-dependent translation is cheap per intersection; rays become less coherent. A CPU uses the same math with fewer coherence constraints. | Motion-aware acceleration bounds and animated transform interpolation. |
| Post-processing | Independent pixels and separable filtering suit the GPU; CPU filtering adds bandwidth and possible transfers. | Shared-memory filter tiles, downsampled bloom pyramid, fewer display updates. |
| Checkpoints | Accumulation is already copied to the CPU each sample. File I/O is CPU work; a GPU cannot accelerate the disk. | Avoid per-sample host copies, asynchronous staging, optional lossless compression. |
| Compaction | Work-efficient scan parallelizes large queues; small queues can lose to launch/synchronization overhead. A CPU's linear stable filter is inexpensive for small arrays. | Ballot-based local ranks, fewer host count transfers, adaptive compaction. |
| OBJ/culling | Thousands of independent ray/triangle intersection checks exploit GPU throughput; linear traversal wastes work for large meshes. CPU construction/loading is appropriate and happens once. | Per-mesh BVHs, global BVH, indexed vertex reuse and smooth shading. |
