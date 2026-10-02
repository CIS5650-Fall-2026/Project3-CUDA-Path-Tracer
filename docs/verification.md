# Verification and assignment coverage

The current renderer source hash is `a1979f061a22d1a7e815bb906ba62a2eeb1f0ae65f1c7a51ecbb19aa36bdd639`, computed by `scripts/source_hash.py` over CMake, renderer, compaction, external and CMake-helper inputs. The recorded Release build supplies this value explicitly as `PROJECT3_BUILD_ID`. Every current render identifies this build; older `results/extension-*`, `results/hedge-final-*` and `results/submission-final-*` records are retained as historical evidence and are not the basis of the current README measurements.

## Coverage and listed points

The core requirements are ideal diffuse scattering, toggleable material sorting and stochastic antialiasing. The README describes each added implementation, provides before/after views and measured costs, and discusses acceleration, a hypothetical CPU renderer and future improvements.

| Listed feature | Nominal points |
|---|---:|
| glTF mesh loading with toggleable bounding-volume culling | 4 |
| Hierarchical GPU BVH traversal | 6 |
| Fresnel reflection/refraction | 2 |
| File/procedural texture and bump mapping with mesh loading | 6 |
| Restartable path tracing | 5 |
| Thin-lens depth of field | 2 |
| Direct-light sampling | 2 |
| Per-material wavefront scheduling | 6 |
| Two complex procedural shapes and two procedural textures | 4 |
| Russian roulette | 1 |
| Scrambled Halton sampling | 3 |
| Object motion and exposure averaging | 3 |
| **Total included in the conservative portfolio** | **44** |

The first 10 feature points satisfy Part 2. The remaining 34 nominal points support a discretionary bonus capped at +25. This is an accounting of the listed implementation values, not a guaranteed award. OBJ is retained but its +2 tier is not stacked with glTF's +4 tier. No additional stream-compaction points are claimed for the authorized Project 2 reuse. Subsurface scattering, denoising, post-processing, Vulkan interop and non-perfect specular scattering are not claimed.

## Current evidence

| Check | Result | Retained evidence |
|---|---|---|
| Release CUDA and preview build; CTest | 5/5 tests pass | [Build and system records](../results/submission-checkpoint-final-rebuild/) |
| AWS host geometry, all scenes | 24 groups, 41 scenes, 10,557,812 assertions; zero failures | [Build-phase summary](../results/submission-checkpoint-final-rebuild/summary.json) |
| Mac host geometry, all scenes | 24 groups, 41 scenes, 10,556,741 assertions; zero failures | [Host report](../results/submission-checkpoint-final-host-mac/test-summary.json) |
| Host glTF, checkpoint and output-path checks | Optimized and ASan/UBSan pass; glTF 11 groups/351 assertions, checkpoint 16 groups/318 checks and output paths 8 groups/80 checks each on APFS | [Executed commands and logs](../results/submission-checkpoint-final-host-mac/) |
| Existing scene/asset preservation | All 63 preceding scene/asset inputs byte-identical to their recorded source72 host-evidence hashes | [Hash comparison](../results/submission-checkpoint-final-host-mac/existing-scenes-assets-comparison.json) |
| Instructions restored | Byte-identical to the supplied INSTRUCTION (3).md | [Restoration record](../results/submission-checkpoint-final-host-mac/instruction-restoration.json) |
| Evidence-tool self-tests and deterministic glTF regeneration check | Seven commands pass; synthetic self-tests are separately labeled | [Tool report](../results/submission-checkpoint-final-tools/summary.json) |
| Device checks | 57/57 pass | [Device output](../results/submission-checkpoint-final-validation/gpu-validation.stdout.txt) |
| Core renders, rejection and statistical regression | 57 checks pass | [Validation](../results/submission-checkpoint-final-validation/summary.json) |
| Sampling/motion/format/runtime validation | 338 checks across 108 invocations; zero failures | [New-feature validator](../results/submission-checkpoint-final-feature-validation/summary.json) |
| CUDA memory and race validation | 16 runs pass, zero errors/hazards/warnings; exact per-scene/sampler sums | [Sanitizer records](../results/submission-checkpoint-final-suite-sanitizers/summary.json) |
| Baseline timing matrix | 48 variants, 288 processes; 168 outputs in 12 exact groups verified | [Matrix](../results/submission-checkpoint-final-experiments/summary.json), [exact comparisons](../results/submission-checkpoint-final-experiments/exact-comparison.json) |
| Sampling convergence and equal-time study | 140 renders; qualified; 36.2% pooled MSE reduction, 6/6 improved means and 6/6 image-mean checks pass | [Complete quality report](../results/submission-checkpoint-final-quality/quality/summary.json) |
| Sampling and motion timing | 36 processes, three warmups/three repeats per setting | [Timing](../results/submission-checkpoint-final-timing/timing/summary.json) |
| glTF encodings, moving meshes, resume, culling and host loading | 62 renders; all checks pass | [glTF study](../results/submission-checkpoint-final-gltf/summary.json) |
| Current comparison galleries | 23 baseline +16 new-feature +8 glTF records | [Baseline settings](../results/submission-checkpoint-final-gallery/gallery.json), [new settings](../results/submission-checkpoint-final-feature-gallery/gallery/summary.json), [glTF settings](../results/submission-checkpoint-final-gltf/gallery/summary.json) |
| Interactive preview | camera controls, six unique saves, progress and reset pass; preview/headless PNG comparisons exact | [Preview check](../results/submission-checkpoint-final-preview/summary.json) |
| Actual interaction recording | 49.333-second actual MP4, complete controls; 124-spp independent comparison: 1 RGB8 channel value(s) differ, maximum 1 level; raw equality not claimed | [Recording and reset comparison](../results/submission-checkpoint-final-interaction/) |

The sanitized Mac geometry suite also passes 24 groups and 1,808,444 assertions. LeakSanitizer is disabled because it is unsupported on this Apple runtime; AddressSanitizer, UndefinedBehaviorSanitizer and float-cast-overflow checks run with error recovery disabled.

Host assertion counts include checks performed for observed hits and can vary between compiler/platform runs. They are CPU correctness checks, not measurements of a CPU path tracer.

## Checkpoint boundary regressions

The headless preflight compares physical output identities before rendering. Our original helper resolves symbolic links, handles existing hardlinks and temporarily reserves absent path components as exclusive empty directories. Comparing those real filesystem objects also detects native case folding and Unicode normalization. It removes only its own empty reservations and fails closed on inspection, creation, comparison or cleanup errors. Ownership and path buffers are allocated before any creation; native empty-directory removal supplies allocation-safe cleanup during exception unwinding.

The camera parser expressly supports `FOVY = 0`. Checkpoint validation now accepts its finite zero projection dimensions while continuing to reject negative or nonfinite values. A test constructs the accepted scene, saves a checkpoint and verifies exact state on reload. Fresh actual GPU CLI runs also save a zero-FOV render, continue it from 3 to 8 samples, and compare it with an uninterrupted 8-sample render. Symlink-parent, dangling-final-symlink and existing-hardlink collisions are rejected without modifying the protected files. [Actual command records and artifacts](../results/submission-checkpoint-final-checkpoint-boundaries/) retain the evidence.

The Mac path tests observe case and Unicode collation on APFS. The Linux build reports its own filesystem behavior; two conditional native-equivalence assertions make its check count 78 when both forms are distinct, versus 80 on APFS. The test population remains eight groups on both platforms. Independent preliminary reviewers additionally exercised collation, complete filesystem snapshots, permission failures and injected allocation failures. These focused checks do not count as the four final whole-bundle audits.

## Hardware and reproduction

GPU evidence uses an AWS g5.xlarge workstation with AMD EPYC 7R32 (4 vCPUs), 16 GiB provisioned host RAM, NVIDIA A10G (23,028 MiB reported memory), Ubuntu 24.04.4, CUDA 12.8.93, GCC 13.3 and CMake 3.28.3. Release targets architecture 86. [System logs](../results/submission-checkpoint-final-rebuild/system.log) retain the actual reports. The graphical preview uses NVIDIA OpenGL on the same GPU. Local host checks use Apple M5, 24 GiB RAM, macOS 27.0.1 and Apple clang 21.0.0; [host specifications](../results/submission-checkpoint-final-host-mac/system.json) are separate.

Run each suite phase with a new results prefix:

```sh
python3 scripts/run_hedge_suite.py --phase build --build build --prefix reproduction
python3 scripts/run_hedge_suite.py --phase baseline-validation --build build --prefix reproduction
python3 scripts/run_hedge_suite.py --phase new-validation --build build --prefix reproduction
python3 scripts/run_hedge_suite.py --phase sanitizers --build build --prefix reproduction
python3 scripts/run_hedge_suite.py --phase baseline-measurements --build build --prefix reproduction
python3 scripts/run_hedge_suite.py --phase new-measurements --build build --prefix reproduction
python3 scripts/run_hedge_suite.py --phase gallery --build build --prefix reproduction
python3 scripts/measure_gltf.py --binary build/bin/cis565_path_tracer_headless --build build --output results/reproduction-gltf --phase all
```

The current measurement driver is the exact driver executed for this refresh; each session records its hash. The CLI returns 0 for a completed qualified result, 2 for a completed negative/inconclusive result, and 1 for an execution failure. Fresh synthetic completion-branch checks and tool self-tests exercise those distinctions without fabricating render evidence.

Phases record commands, UTC times, elapsed time, logs, binary hashes and source identity before/after. Output guards prevent overwriting an existing study. A completed negative or inconclusive sampling result has exit code 2 and remains evidence; execution failure has exit code 1. The design does not discard unfavorable seeds or trials.

## Exactness, motion and restart

Both samplers are indexed by original pixel and absolute sample number. Sorting, compaction, traversal and wavefront scheduling must preserve the complete raw accumulation at a fixed physical configuration. Fresh-process 5-to-17 continuations inherit omitted sampler/motion/camera/configuration. Explicit mismatches, altered assets, invalid fields and corrupted records are rejected. Checkpoint format 3, sampler identifiers 2/3 and raw accumulation format 1 are separate version contracts; checkpoint versions 1/2 are rejected.

A path samples one exposure time, preserved through every bounce and visibility ray. Moving sphere, cube, mesh and emissive geometry tests cover coherent intersections and light PDFs. Frozen shutters and motion-off runs have independent exact comparisons. Swept BVH bounds enclose the complete declared motion interval, so midpoint/full-shutter timing holds the hierarchy fixed. glTF validation checks external-buffer, data-URI and GLB equivalence, both samplers on genuinely moving loaded geometry, BVH on/off and minimal checkpoint continuation.

Memory/race checks cover the original workbench and moving validation scene, counter/Halton sampling, sorted unified/wavefront schedules and both tools. The non-square 67-by-35 resolution exercises partial blocks; raw sums agree within each scene/sampler group. Host ASan/UBSan separately covers parsing, accessor range/overflow checks, hierarchy transforms and checkpoint storage.


Final review reproduced unsafe float-to-int conversion and signed addition in checker parity at large accepted inputs. The original shared host/device fix preserves float multiplication before flooring for ordinary inputs, uses safe cell parity without signed addition, and defines overflowing float products through a finite double fallback. Optimized and ASan/UBSan/float-cast-overflow host tests and six actual CUDA validation groups cover ordinary behavior, float-rounding boundaries, signed-sum overflow, out-of-range cells, product overflow and surface consumption. All current GPU studies were rerun after this source change; older passing reports do not accept the corrected bundle.

## Statistical and timing methods

The sampling study fixes two scenes, 128-by-84 resolution, depth 10, budgets 16/64/256 and all eight seeds: 947, 104729, 130363, 155921, 196613, 262147, 314159 and 524287. Independently seeded counter references use 2718281, 3141593, 1618033 and 1414213 at 8,192 samples each. Split-reference RMSE must be at most 25% of the smaller mean candidate RMSE at 256 samples. Exactly one retry at 16,384 samples is allowed for a failing scene, using all the same reference seeds. Both attempts remain in the record.

Qualification requires adequate references, at least 10% pooled MSE reduction, improvement in at least four of six scene/budget means and all six paired image-mean criteria. The image-mean criterion is `abs(mean(halton-counter)) <= max(4*paired_SE, 1%*counter_mean)+1e-6`. A separate equal-time population uses a 4,000 ms render-loop budget and 65,536-sample ceiling, retaining actual samples, elapsed time, whole-sample overshoot and process overhead. Fixed-sample improvement does not imply improvement per unit time.

The independent roulette regression uses the same eight paired candidate seeds, 8,192 samples, 64-by-48 resolution, depth 32 and roulette start 3 on closed/glass fixtures. It retains every seed and both variances, with the same paired-SE/1% image-mean criterion. NEE/BSDF checks cover cube, transformed-sphere and mesh lights; separate depth-one comparisons agree exactly. These are finite empirical checks, not a universal proof of unbiasedness.

Baseline measurements fix 48 variants with three warmup and three measured processes each. Scheduling comparisons share camera, seed, resolution, depth and physical settings. CPU generation has one warmup/three repeats and uses temporary outputs without changing canonical assets. Load includes parsing, transforms, camera setup and initial BVH; a separate BVH rebuild is not added to load time. GPU timing does not claim GPU file parsing or geometry generation.

CUDA-event phase timing synchronizes the instrumented renderer. Overall GPU-event and render-loop times include those synchronization and host scheduling gaps; whole-process time additionally includes startup and output. Aggregate shading and its specialized shader breakdown are alternate views, and overlapping host queue time is not added to the GPU total. Graphs retain repeated-run ranges. Surface/material changes also change subsequent paths and roulette survival, so their costs are not isolated texture instruction timings.

## Images, interaction and provenance

Current README illustrations are mapped to actual render artifacts and metadata by [the image manifest](../img/render-manifest.json). [The chart manifest](../img/chart-manifest.json) maps each displayed SVG to its retained measured data and source-bound suite. The current plotting tools generate readable label positions directly from the refreshed data; their manifest records every output hash. The baseline gallery retains both original and procedural covers, six plain/pattern sculpture comparisons, diagnostics, feature pairs and exact restart. New views add counter/Halton, four exposures, mesh-presence and scheduling comparisons. glTF adds its own presence/culling pairs. Capture settings and the post-Space saved-image/headless comparison are retained with the interaction evidence; capture frame rate is not renderer throughput. Earlier pre-capture attempts retained under `results/hedge-final-interaction-attempt1/` and `results/hedge-final-interaction-attempt2/` are historical failed attempts, not current successful recordings. The corrected recorder measures absolute client coordinates with `xwininfo`, avoiding the observed parent-offset double counting in `xdotool`; the native 960×640 image is placed at (160,69) inside the 1280×720 capture. Its geometry checks do not resize the renderer. The earlier recorder is archived with the interaction tools. The preceding successful GIF and image/chart manifests are preserved under `results/hedge-final-interaction/historical-media/` with exact archival hashes; they do not validate the corrected bundle. The preceding source-07233055 successful GIF and image/chart manifests are separately preserved under `results/submission-final-interaction/historical-media/`, while its original MP4 remains with its capture evidence.

> Using third-party code without its approval, including using another student's code, is an academic integrity violation, and will, at minimum, result in you receiving an F for the semester.

Never use another student's implementation or unapproved internet implementation. Only the authorized unchanged official starter/dependencies and specifically approved unchanged tinygltf header/license are third-party exceptions. The README credits Bryce Joseph's September 21 Special OH approval and Matthew Schwartz's October 1 Ed confirmation; [the package register](approved-third-party.json) records its exact revision/hashes/license. Our Project 1/2 source reuse has path/revision provenance. Integration, renderer, tests, asset generation and evidence tooling are independently authored.

The completed local bundle requires two independent reviewers to conduct two fresh full audits each on the same frozen whole-deliverable hash. Any deliverable edit resets all four reports. Reports and the acceptance index remain outside the repository, in `../project_3_reviews/<bundle-hash>/`. Older audits are historical and do not accept this version. No publication or submission is authorized before the student's inspection.
