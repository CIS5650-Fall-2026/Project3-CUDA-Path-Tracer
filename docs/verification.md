# Validation and feature coverage

## Scope

Core rendering includes diffuse and ideal specular transport, segmented intersection/shading kernels, stream compaction, toggleable material sorting, and stochastic antialiasing. The extension adds exact-material wavefront queues, roulette termination, two original procedural mesh families, and two original analytic textures.

| Implemented feature | Listed value | Evidence |
|---|---:|---|
| Original OBJ parser and triangle rendering | 2 | Original generated meshes; malformed-input and transform tests |
| CPU-built BVH, iterative GPU traversal | 6 | Exact brute-force agreement; three mesh-complexity benchmarks |
| Fresnel reflection and refraction | 2 | Dielectric edge checks; glass/mirror comparison |
| File and procedural textures; mesh bump mapping | 6 | UV/normal diagnostics; texture and bump comparisons |
| Restartable path tracing | 5 | Lossless sums and BVH state; fresh-process exact continuation |
| Thin-lens depth of field | 2 | Zero-aperture identity; focused/defocused comparison |
| Direct-light sampling | 2 | Cube, transformed-sphere and triangle light tests; MIS comparison |
| Per-material wavefront queues and specialized kernels | 6 | Queue conservation and exact legacy equivalence; scheduling comparisons |
| Two new complex procedural shapes and two new textures | 4 | Fluted body and swept knot; marble and staggered-cell renders |
| Russian roulette | 1 | Survival compensation, statistical agreement and open/closed timing |
| **Total listed feature value** | **36** | |

The assignment requires at least 10 feature points before additional work can earn a bonus, capped at +25/100 at the graders' discretion. This portfolio has 36 nominal listed points: 10 required and 26 additional, giving a one-point margin toward the cap. Listed values do not guarantee the awarded bonus; in particular the new procedural work must be assessed separately from the existing texture feature. Previous Project 2 scan work is reused with provenance and is not counted again.

## Reproduce the checks

```sh
ctest --test-dir build --output-on-failure
python3 scripts/validate_render.py --binary build/bin/cis565_path_tracer_headless --output results/extension-validation
python3 scripts/generate_assets.py --profile results/extension-host/procedural-generation.json --profile-only
python3 scripts/profile_mesh_load.py --build build --output results/extension-mesh-load
python3 scripts/run_experiments.py --binary build/bin/cis565_path_tracer_headless --output results/extension-experiments
python3 scripts/verify_experiments.py results/extension-experiments
python3 scripts/plot_experiments.py results/extension-experiments --procedural-cpu-profile results/extension-mesh-load/summary.json
python3 scripts/render_gallery.py --binary build/bin/cis565_path_tracer_headless --output results/extension-gallery
python3 scripts/verify_preview.py --output results/extension-preview
```

The validation, experiment, scene-load, gallery and preview output directories must not already exist. Run GPU measurements sequentially with no competing GPU workload. The graphical check needs the preview executable and an active NVIDIA OpenGL display; its driver records the actual actions and saved images.

The validation script checks raw finite/nonnegative accumulation and exact scheduling equivalence; separate-process checkpoint continuation; incompatible configuration, version, corruption and asset-change rejection; truncated-depth lighting behavior; agreement of independent NEE/BSDF lighting estimates; and roulette's paired statistical agreement. Output logs and metadata accompany every render.

## Validation system

GPU evidence uses an AWS g5.xlarge workstation with AMD EPYC 7R32 (4 vCPUs), 16 GiB provisioned host RAM (16,554,987,520 bytes visible to the OS), NVIDIA A10G (23,028 MiB reported GPU memory), NVIDIA driver 580.178.04, CUDA 12.8.93, Ubuntu 24.04.4, GCC 13.3, and CMake 3.28.3. Release builds target CUDA architecture 86. The preview uses the NVIDIA OpenGL driver on the same GPU. The measured CPU and host-memory report is retained in [system.log](../results/extension-rebuild/system.log).

The local host geometry pass uses Apple M5, 24 GiB RAM, macOS 27.0.1, Apple clang 21.0.0, C++17 and `-O3 -DNDEBUG`: 21 test groups, all 24 scene files, 6,989,243 assertions, zero failures. The retained [geometry log](../results/extension-host/geometry-tests.log) and [host specifications](../results/extension-host/system.log) describe a CPU correctness check; they are separate from GPU performance evidence.

## Recorded results

Renderer build-input SHA-256: `d1ca1c603f9fcec1d353fd3d30b61a8025b9d3da5ad2294bf24f66e4ab0923aa`. Local and remote input hashes agree. The compatibility identifier additionally records compiler/toolkit versions, Release configuration, architecture, and a compiler-flags fingerprint. Sampler version is **2**. The baseline revision field identifies the inherited starter; the content hash identifies this implementation.

| Check | Result | Retained evidence |
|---|---|---|
| Release build and CTest | Both executables build; 3/3 tests pass | [Build, configuration and test logs](../results/extension-rebuild/) |
| Local geometry and all repository scenes | 21 groups, 24 scenes, 6,989,243 assertions; zero failures | [Geometry log](../results/extension-host/geometry-tests.log) |
| New procedural mesh geometry | All six tessellations pass serialized geometry, seam and topology checks | [Mesh validation](../results/extension-host/procedural-mesh-validation.json) |
| Existing generated assets and scenes | All 22 are byte-identical to the previous completed revision | [Baseline comparison](../results/extension-host/old22-baseline-comparison.json) |
| Device equivalence and edge checks | 34/34 pass, including 35 queue-conservation fixtures | [GPU report](../results/extension-validation/gpu-validation.stdout.txt) |
| Actual renders, checkpoint rejection and estimator checks | All 55 checks pass | [Validation summary](../results/extension-validation/summary.json) |
| Benchmark matrix and exact scheduling comparisons | All 48 variants and 288 processes verified; exact sums across 168 processes in 12 groups | [Exact comparison](../results/extension-experiments/exact-comparison.json) |
| CUDA memory validation | Zero errors in both scenes and modes | [Four memcheck records](../results/extension-rebuild/sanitizer-summary.json) |
| CUDA race validation | Zero hazards, errors, or warnings in both scenes and modes | [Four racecheck records](../results/extension-rebuild/sanitizer-summary.json) |
| CPU procedural generation and scene loading | Six tessellations measured separately on AWS; all generated mesh validations pass | [Generation profile](../results/extension-host/procedural-generation.json), [load/BVH profile](../results/extension-mesh-load/summary.json) |
| Gallery and continuation | All 23 gallery records retained; 16 → 512 continuation equals uninterrupted sums byte for byte | [Gallery settings](../results/extension-gallery/gallery.json), [resumed metadata](../results/extension-gallery/restart-resumed.json) |
| Graphical preview integration | Preview/headless and Space restoration exact; six saves and all three independent camera checks pass | [Preview summary](../results/extension-preview/summary.json) |

The sequential [measurement driver](../results/extension-rebuild/measurement-driver-summary.json) records successful generation, scene-load, experiment, exact-comparison, plot, gallery and preview stages with their commands, elapsed times and logs. All rendering evidence identifies the source hash above.

### Scheduling and continuation contract

The actual-render fixture uses 17 × 13 pixels, 17 samples, seed 947 and depth 7. It exercises all four wavefront/roulette combinations. Every valid comparison disables compaction, selects brute-force traversal, enables sorting in unified mode, or combines those changes without changing the complete raw sum file. Wavefront and sorted unified mode are mutually exclusive; `--sort on requires --wavefront off` is rejected explicitly. Instrumented and uninstrumented runs must also retain identical sums, while recorded queue and roulette counters satisfy conservation checks.

Each configuration has a separate-process 5 → 17 continuation. Resumption supplies only the scene, target samples, output prefix and checkpoint; seed, resolution and renderer configuration must be inherited exactly. One case retains `rr_start=11` above its depth of 7. The checkpoint format is **2**, with strict rejection of version 1; the raw accumulation format and JSON metadata format retain their independent version 1 identifiers.

Rejection cases change seed, resolution, NEE, wavefront, roulette or roulette start; supply conflicting sorting, invalid roulette bounds or noncanonical boolean bytes; corrupt the checkpoint; and change an asset byte. The altered asset is restored byte for byte before later work. Roulette start is ignored for rendering when roulette is off, and roulette on/off must be byte-identical when no eligible continuation depth is reached.

The fixed benchmark design has 48 variants, each with three warmups and three measured processes: 288 invocations. Its original 26 variants explicitly disable wavefront and roulette. Twelve variants compare unsorted unified, sorted unified and wavefront schedules on uniform, mixed, closed and mesh-heavy workloads at 512 × 336, 32 samples and depth 16, with roulette off. Four compare roulette on/off in open and closed scenes at the same resolution and sample target and depth 32. Six compare the two new shapes with plain, marble and cell materials at 160 × 106, 16 samples and depth 10, with roulette on.

Verification confirms the full 48-variant matrix, all 288 invocation settings and common physical inputs within each scheduling comparison. Its 12 exact groups cover the eight original compaction, sorting, BVH and checkpoint groups plus the four new wavefront groups: 168 process outputs agree exactly within their groups. Roulette changes the sampling process, so its on/off comparisons use statistical evidence rather than byte identity. Raw files retain finite nonnegative float sums, sample counts and checksums.

### Memory and race checks

Memory/race checks use 65 × 33 pixels, three samples and depth 8 with roulette enabled, in both sorted unified and wavefront modes. Each mode is checked on both the original workbench and the new procedural still life, covering mesh traversal, glass, file/procedural textures, bump, and direct lighting. All eight runs pass and their lossless sums match across modes and tools for each scene. The non-square, non-block-multiple size exercises multiple compaction blocks. The [exact sanitizer comparison](../results/extension-rebuild/sanitizer-exact-comparison.json) records the file hashes.

### Transport estimator checks

Lighting tests use 8,192 samples at 48 × 32 and depth 4, with roulette off to isolate the estimators. Cube, nonuniformly transformed sphere and triangle emitters each require the NEE/BSDF image means to differ by less than 5%. Their measured relative differences are 0.073163% (cube), 0.093751% (sphere) and 0.285169% (mesh). All three separate depth-one tests agree exactly, ensuring direct lighting does not add an untraced transport vertex.

Roulette's acceptance test uses the fixed seeds 947, 104729, 130363, 155921, 196613, 262147, 314159 and 524287. Each seed produces both settings in a closed diffuse chamber and a glass workbench, at 64 × 48, 8,192 samples, depth 32 and roulette start 3. All 32 renders and all seed pairs are retained. For each scene, the absolute mean of the eight paired differences must satisfy `abs(mean(on-off)) <= max(4*SE, 1%*mean(off)) + 1e-6`, where `SE` is the paired sample standard error. The summary retains both sample variances, means, paired estimates and the computed threshold. No seed is selected or discarded after observation. Both scenes pass:

| Scene | Mean, RR off | Mean, RR on | Relative mean error | Paired standard error | Absolute error / allowed error |
|---|---:|---:|---:|---:|---:|
| Closed diffuse chamber | 0.043943833 | 0.043944558 | 0.001651% | 1.304595e-6 | 7.254832e-7 / 4.404383e-4 |
| Glass workbench | 0.286834067 | 0.286827642 | 0.002240% | 1.623656e-5 | 6.425238e-6 / 2.869341e-3 |

### Procedural geometry and CPU setup

The fluted body has 1,600, 6,272 and 24,832 triangles across its three tessellations; the swept knot has 1,024, 4,096 and 16,384. The serialized meshes have complete finite positions, normals and UVs; unit normals; positive, outward, noncollapsed triangles; canonical periodic seams with separate UV endpoints; and closed two-manifold edge incidence. The body preserves flat cap normals and radial cap UVs. Welded Euler characteristics are 2 for the body and 0 for the knot.

CPU generation profiling creates only temporary OBJ outputs, reads the canonical assets without rewriting them, and records both generated and canonical sizes and SHA-256 hashes. One warmup and three measured runs per mesh separately time construction/write and serialized validation. All six AWS-generated temporary files have different byte hashes from the canonical Mac-generated assets; both hashes and sizes are retained, all six generated meshes pass geometry validation, and the renderer uses the unchanged canonical assets. The profiler documents platform math-library formatting differences as a possible cause; cross-platform byte identity is not claimed.

Scene-load profiling separately measures OBJ parsing, transforms, camera setup and the initial BVH, then a BVH rebuild from loaded primitives. It also uses one warmup and three measured runs with files warm in the OS cache. The AWS medians below use the recorded EPYC host, CPython 3.12.3 for generation and GCC 13.3/C++17 Release code for loading. These are CPU setup measurements, not GPU geometry-generation timings.

| Mesh | Triangles | Construction/write (ms) | Validation (ms) | Scene load including BVH (ms) | Separate BVH rebuild (ms) |
|---|---:|---:|---:|---:|---:|
| Fluted body, low | 1,600 | 17.329 | 28.982 | 7.196 | 0.911 |
| Fluted body, medium | 6,272 | 65.624 | 114.702 | 28.513 | 4.484 |
| Fluted body, high | 24,832 | 258.427 | 473.043 | 116.399 | 21.922 |
| Swept knot, low | 1,024 | 38.105 | 18.726 | 4.182 | 0.527 |
| Swept knot, medium | 4,096 | 141.692 | 73.305 | 16.789 | 2.657 |
| Swept knot, high | 16,384 | 543.436 | 304.380 | 68.503 | 13.281 |

Generation and validation operate on temporary files; scene loading and BVH rebuilding operate on canonical assets. The separately measured rebuild is not added to the scene-load value, which already includes an initial BVH build.

The retained Mac checks also verify profiler isolation, deterministic regeneration and controlled mismatch detection. Their original absolute paths are historical execution metadata. The [artifact index](../results/extension-host/procedural-artifact-index.json) maps every cross-report reference to its bundled copy with a byte size and SHA-256. Those Mac QA timings are separate from the AWS performance profile.

## Images and measurements

- [Gallery settings](../results/extension-gallery/gallery.json) and [copied-image hash manifest](../img/render-manifest.json).
- [Every experiment invocation](../results/extension-experiments/runs.csv), [GPU/loop medians and ranges](../results/extension-experiments/timing-summary.json), and [single-iteration phase/count records](../results/extension-experiments/single-iteration.csv).
- [AWS CPU generation profile](../results/extension-host/procedural-generation.json) and [CPU scene-load/BVH measurements](../results/extension-mesh-load/summary.json).
- Each prefix retains PNG, lossless `.sum.bin`, `.json` settings/compatibility, stdout/stderr, and benchmark CSV where requested. Checkpoints used for continuation and cadence measurements are retained.

The gallery retains 23 records: the original 16, six shape/texture comparisons, and the combined procedural cover. Covers use 960 × 640 and 2,048 samples; the six new individual comparisons use 640 × 424 and 512 samples. All use seed 947, depth 10, wavefront and roulette on, apart from explicit feature/debug overrides. `img/workbench.png` is copied from the new `procedural-cover.png`; the original workbench render remains in the gallery evidence. The copied-image manifest records the source render and hashes. The 16 → 512 restart comparison equals its uninterrupted 512-sample counterpart byte for byte; both raw files have SHA-256 `fc3fa296f37bc0ca8c9997756447d2209f45154c43b1c81a3e3a3dc0fc85df4d`.

The graphical integration check passes with wavefront and roulette on, roulette start 3. The 16-sample preview PNG equals headless output byte for byte. Horizontal-only orbit, vertical zoom and screen-plane pan each change the camera image independently, compared with a fresh original-camera image at the same sample count. Space, S and Escape deliver successfully while frames advance; six unique saves are retained, Escape exits with code 0 and the post-Space image exactly equals a fresh 9-sample original-camera headless render. The [preview summary](../results/extension-preview/summary.json) and [action log](../results/extension-preview/actions.json) record the actual outcomes.

## Evidence interpretation

Each benchmark comparison holds seed, camera, depth, resolution and sample target fixed. Three warmups precede three measured processes per variant. Overall CUDA-event and render-loop elapsed times include the instrumented renderer's synchronization and CPU scheduling gaps; render-loop wall time additionally includes checkpoint serialization when enabled. Whole-process timings include setup and file output. These quantities are reported separately.

The wavefront shading aggregate is retained for compatibility. Its diffuse, mirror, dielectric, emission and terminal timers are a breakdown of that aggregate, so they are not added to it a second time. Queue-host time includes count/offset/live-count downloads, validation and launch submission; it overlaps overall elapsed time and is shown separately from GPU phases. Procedural texture comparisons include changed albedo and roulette survival, so they describe whole-render costs rather than isolated texture evaluation.

Feature-cost comparisons describe these scenes and settings; they are not general claims about all hardware or workloads. The README's CPU-renderer discussion is qualitative. Actual CPU generation, loading and BVH setup measurements are identified separately.

## Integrity acceptance

> Using third-party code without its approval, including using another student's code, is an academic integrity violation, and will, at minimum, result in you receiving an F for the semester.

This is a hard completion constraint. Never use another student's code. Never copy or adapt code from the internet. The sole explicitly authorized third-party baseline is the official starter and its unchanged existing dependencies, retaining their credits and licenses. Reuse from our own Project 1 and Project 2 is limited to our authored implementation and requires revision/path provenance. No new third-party implementation, downloaded model, noise library or dependency is permitted. The baseline, reuse and independently authored source/assets are documented in [provenance.md](provenance.md), `starter-manifest.json` and `change-manifest.json`.

After all source, documentation, images and measurements are complete, a clean local commit is hashed as a whole deliverable. Two independent reviewers each conduct two full audits of that same snapshot. Reports are stored outside the repository; any change requires a new hash and all four reviews again. Preflight reviews do not count.
