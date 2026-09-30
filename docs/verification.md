# Validation and feature coverage

## Scope

Core rendering includes diffuse and ideal specular transport, segmented intersection/shading kernels, stream compaction, toggleable material sorting, and stochastic antialiasing.

| Implemented feature | Listed value | Evidence |
|---|---:|---|
| Original OBJ parser and triangle rendering | 2 | Original generated meshes; malformed-input and transform tests |
| CPU-built BVH, iterative GPU traversal | 6 | Exact brute-force agreement; three mesh-complexity benchmarks |
| Fresnel reflection and refraction | 2 | Dielectric edge checks; glass/mirror comparison |
| File and procedural textures; mesh bump mapping | 6 | UV/normal diagnostics; texture and bump comparisons |
| Restartable path tracing | 5 | Lossless sums and BVH state; fresh-process exact continuation |
| Thin-lens depth of field | 2 | Zero-aperture identity; focused/defocused comparison |
| Direct-light sampling | 2 | Cube, transformed-sphere and triangle light tests; MIS comparison |
| **Total listed feature value** | **25** | |

This portfolio includes the required minimum feature work. Listed values do not guarantee the discretionary grade bonus. Previous Project 2 scan work is reused with provenance and is not counted again.

## Reproduce the checks

```sh
ctest --test-dir build --output-on-failure
python3 scripts/validate_render.py --binary build/bin/cis565_path_tracer_headless
python3 scripts/run_experiments.py --binary build/bin/cis565_path_tracer_headless
python3 scripts/render_gallery.py --binary build/bin/cis565_path_tracer_headless
python3 scripts/plot_experiments.py results/experiments-<UTC timestamp>
python3 scripts/verify_experiments.py results/experiments-<UTC timestamp>
python3 scripts/verify_preview.py
```

The validation script checks raw finite/nonnegative accumulation and exact scheduling equivalence; separate-process checkpoint continuation; seed, resolution, feature, corruption and asset-change rejection; truncated-depth lighting behavior; and agreement of independent NEE/BSDF lighting estimates. Output logs and metadata accompany every render.

## Validation system

GPU evidence uses an AWS g5.xlarge workstation with AMD EPYC 7R32 (4 vCPUs), 16 GiB provisioned host RAM (16,554,995,712 bytes visible to the OS), NVIDIA A10G (23,028 MiB reported GPU memory), NVIDIA driver 580.178.04, CUDA 12.8.93, Ubuntu 24.04.4, GCC 13.3, and CMake 3.28.3. Release builds target CUDA architecture 86. The preview uses the NVIDIA OpenGL driver on the same GPU. The measured CPU and host-memory report is retained in [system.log](../results/final-rebuild/system.log).

The local host geometry pass uses Apple clang, C++17 and `-O2`: 18 test groups, all 17 scene files, 4,639,718 assertions, zero failures. This is a CPU correctness check; it is not GPU performance evidence.

## Recorded results

Renderer build-input SHA-256: `f30dfa95d6395ddf1f655ef85441b90e6fa7e02454e72a7a4dda4607287b3d03`. Local and remote input hashes agree. The compatibility identifier additionally records compiler/toolkit versions, Release configuration, architecture, and a compiler-flags fingerprint. Sampler version is **2**. The baseline revision field identifies the inherited starter; the content hash identifies this implementation.

| Check | Result | Retained evidence |
|---|---|---|
| Release build and CTest | Both executables build; 3/3 tests pass | [Build, configuration and test logs](../results/final-rebuild/) |
| Device equivalence and edge checks | 13/13 pass | [GPU report](../results/validation-20260930T234238Z/gpu-validation.stdout.txt) |
| Actual renders and checkpoint rejection | 17 checks pass | [Validation summary](../results/validation-20260930T234238Z/summary.json) |
| Benchmark scheduling and serialization | Exact sums across 96 processes in eight comparisons | [Exact comparison](../results/experiments-20260930T234307Z/exact-comparison.json) |
| CUDA memory validation | Zero errors | [Memcheck log](../results/final-rebuild/memcheck.log) |
| CUDA race validation | Zero hazards, errors, or warnings | [Racecheck log](../results/final-rebuild/racecheck.log) |
| Gallery continuation | 16 → 512 samples equals uninterrupted 512-sample sums byte for byte | [Resumed metadata](../results/gallery-20260930T234411Z/restart-resumed.json) |

The standalone equivalence fixture renders 17 × 13 pixels at 17 samples and depth 7. Comparisons include compaction disabled, sorting enabled, brute-force traversal, all three scheduling changes together, and a fresh-process 5 → 17 continuation. Rejection cases alter seed, resolution, NEE, checkpoint bytes, and an asset byte. The asset is restored exactly before other experiments.

Additional measured images check compaction, sorting, BVH and checkpoint toggles in all warmup/measured processes. The BVH comparisons use each of the three different mesh complexities. Raw files retain finite nonnegative float sums, sample counts and checksums.

Memory/race checks use 65 × 33 pixels, three samples and depth 8 with material sorting enabled. This non-square, non-block-multiple size exercises multiple compaction blocks alongside mesh traversal, glass, textures, bump and direct lighting.

Lighting tests use 8,192 samples at 48 × 32 and depth 4. Relative NEE/BSDF mean differences are 0.0732% (cube), 0.0938% (nonuniformly transformed sphere), and 0.2852% (mesh). Separate depth-one tests agree exactly, ensuring direct lighting does not add an untraced transport vertex.

## Images and measurements

- [Gallery settings](../results/gallery-20260930T234411Z/gallery.json) and [copied-image hash manifest](../img/render-manifest.json).
- [Every experiment invocation](../results/experiments-20260930T234307Z/runs.csv), [GPU/loop medians and ranges](../results/experiments-20260930T234307Z/timing-summary.json), and [single-iteration phase/count records](../results/experiments-20260930T234307Z/single-iteration.csv).
- Each prefix retains PNG, lossless `.sum.bin`, `.json` settings/compatibility, stdout/stderr, and benchmark CSV where requested. Checkpoints used for continuation and cadence measurements are retained.

The graphical integration check passes: 16-sample preview PNG equals headless output byte for byte; horizontal-only orbit, vertical zoom, screen-plane pan, Space, S, and Escape deliver successfully while frames advance. Each drag independently changes the camera image compared with a fresh original-camera render at the same sample count. Six unique saves are retained, Escape exits with code 0, and the post-Space image exactly equals a fresh headless render from the original camera. See the [preview summary](../results/preview-20260930T235251Z/summary.json) and [action log](../results/preview-20260930T235251Z/actions.json).

## Evidence interpretation

Each benchmark comparison holds seed, camera, depth, resolution and sample target fixed. Three warmups precede three measured processes per variant. CUDA event durations include renderer scheduling and phase overhead; render-loop wall time additionally includes checkpoint serialization when enabled. Whole-process timings include setup and file output. These quantities are reported separately.

Feature-cost comparisons describe these scenes and settings; they are not general claims about all hardware or workloads. No CPU renderer performance is fabricated. The README's CPU discussion is qualitative.

## Integrity acceptance

The authorized baseline, unchanged dependencies, own-code reuse and original source/assets are documented in [provenance.md](provenance.md), `starter-manifest.json` and `change-manifest.json`. No other student's code or copied/adapted internet implementation is permitted.

After all source, documentation, images and measurements are complete, a clean local commit is hashed as a whole deliverable. Two independent reviewers each conduct two full audits of that same snapshot. Reports are stored outside the repository; any change requires a new hash and all four reviews again. Preflight reviews do not count.
