<!--
Use this template to create pull requests for your projects as part of your submission.

Title your Pull Request as "Project 3: Your Name" 

Diligently go over the project instructions https://github.com/CIS5650-Fall-2025/Project3-CUDA-Path-Tracer/blob/main/INSTRUCTION.md as part of your submission.
-->

### Repo Link

[Faris Rafie Syahzani — CUDA Path Tracer](https://github.com/farisyahz/Project3-CUDA-Path-Tracer)

Suggested PR title: **Project 3: Faris Rafie Syahzani**

Implementation and analysis follow the repository's current `INSTRUCTION.md`; the original template comments below are retained.

### Core Features Completed

<!--
List of core features completed https://github.com/CIS5650-Fall-2025/Project3-CUDA-Path-Tracer/blob/main/INSTRUCTION.md#part-1---core-features
-->
- [x] Shading kernel with BSDF evaluation (diffuse, perfect specular surfaces)
- [x] Stream compacted path termination (toggleable Thrust removal)
- [x] Sorting by material type (toggleable zipped paths/intersections)
- [x] Stochastic sampled antialiasing (toggleable subpixel jitter)

### Extended Features Completed

<!--
List the extended features implemented from https://github.com/CIS5650-Fall-2025/Project3-CUDA-Path-Tracer/blob/main/INSTRUCTION.md#part-2---make-your-pathtracer-unique
-->

- **Visual Improvements**
    - Dielectric refraction with Schlick Fresnel and total internal reflection.
    - Thin-lens depth of field with disk aperture sampling.
    - Procedural gyroid and three-lobed woven ring; checker and marble textures.
    - Direct area-light sampling of transformed cube/sphere emitters with visibility rays.
- **Mesh Improvements**
    - Original OBJ loader: positions, normals, negative indices, convex polygon triangulation.
    - Original CPU-built mesh BVH and stackless iterative GPU traversal, toggleable against full triangle scanning for both continuation and shadow rays.
    - Attributed Stanford Bunny assets at 16,301 and 69,451 triangles.
- **Performance Improvements**
    - Russian roulette with survivor reweighting, evaluated in open/closed rooms.
    - Mesh BVH measured at 16.58× and 52.45× elapsed-time speedups over linear GPU scanning on the two mesh scenes.
    - Sorting and compaction costs reported honestly, including slower variants.
- **Other Improvements**
    - Headless rendering, PNG/HDR outputs, checkpoints, deterministic seed offsets, timing and per-bounce logs.
    - GUI feature switches reset accumulated samples.

### Other Features and Details

<!--
Include any other features and details that you implemented. Include information about libraries added/modified/removed, command line changes, scene file changes, or anything else that is signficiant to your project for grading.
-->

- README includes an original non-Cornell gallery render, feature comparisons, five-run timing ranges, survivor curves, independent Nsight Systems GPU kernel stacks, convergence/error analysis, hypothetical CPU discussion, and future optimizations.
- Release build completed; actual headless renders/analysis completed on RTX 4050 Laptop GPU. No automated test suite was added. Interactive CUDA/OpenGL mode could not be validated on this hybrid-GPU laptop.
- Relative to starter: CMake C++17/CUDA17, native architecture, separable compilation, Release `-lineinfo`/`-src-in-ptx`, Windows CUDA host `/Zc:preprocessor`, and `src/mesh.h` header entry. Debug/RelWithDebInfo retain device `-G`; benchmarks use Release.
- No third-party loader/traversal code or new libraries were added. Existing framework dependencies retained. Model source, terms, conversion, and checksums documented.
- Raw CSV data, Python helpers, downloaded archive, and generated profiler captures are ignored; finished figures/renders and `analysis/data/summary.json` are included.
- OBJ materials are assigned per mesh; UV/MTL/glTF and scene-level BVH are not implemented. Compaction uses Thrust, not a custom shared-memory scan.

### README Completion Checklist

<!--
Checklist as a rmeinder for your readme. Revisit this once you have completed your README updates.
-->

- [x] Cover image in readme does not use Cornell Box
- [x] Descriptions, screenshots, debug images, side-by-side comparisons of features implemented
- [x] Analysis
- [x] Scenes and meshes included or linked
- [x] Third-party library changes or compilation changes documented
- [x] Bloopers (optional): Bunnyzilla, Stanford Pancake, nightclub lighting, and misplaced-focus renders, plus nonmonotonic enclosed-room lighting convergence

### Late Days Used

<!--
Add number of Late Days used - for both code submission and README submission
-->

Code: **[fill before submitting]**

README/scenes: **[fill before submitting]**

### Project feedback

<!--
Add any project feedback to help make the project better.
-->

Suggested feedback to review before submitting: a short hybrid-GPU/headless setup note and an example distinguishing Nsight GPU kernel durations from CUDA-event elapsed intervals would make profiling easier to get started with.
