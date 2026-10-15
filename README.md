# CUDA Path Tracer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

- Anirudh Akula
- Tested on: Windows 11, NVIDIA T1000 4096MB (CETS Virtual PC)

<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/cbff80bb-2f2e-436b-8f2b-5421c572a762" />
<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/58efed41-c463-4213-8859-fc6648a93f2b" />

## Features

- CUDA path tracer with iterative bounce loop
- Camera ray generation
- Primitive inbuilt geometry (spheres and cubes)
- glTF / glB mesh loading via tinygltf (optional NODE filter, optional normalize-to-unit-box)
- Triangle intersection with Möller–Trumbore
- Mesh acceleration modes (toggleable): brute-force triangles, AABB culling, octree
- Octree built on the CPU, traversed on the GPU with nearest-first child ordering
- Materials:
  - Diffuse, Specular (mirrors + roughness)
  - Glossy (diffuse + reflective coat)
  - Emitting Cosine-weighted hemisphere sampling for diffuse BRDFs
- Stream compaction of terminated paths (thrust::partition + finalGather)
- Optional material sorting (thrust::sort_by_key) to group shading by material
- Stochastic antialiasing
- Interactive OpenGL preview with ImGui controls

## Quick Start

**Requirements:** CMake ≥ 3.24, CUDA Toolkit, C++17 (MSVC on Windows), OpenGL

### Build (Windows)

```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

The binary is written to `build/bin/Release/cis565_path_tracer.exe`.

### Run

From `build/bin/Release`:

```powershell
.\cis565_path_tracer.exe ..\..\..\scenes\cornell.json
```

Other scenes:

```powershell
.\cis565_path_tracer.exe ..\..\..\scenes\dragon_showcase.json
.\cis565_path_tracer.exe ..\..\..\scenes\skull_spheres.json
.\cis565_path_tracer.exe ..\..\..\scenes\chessboard_cubes.json
```

### Controls

- **Esc** — save image and exit
- **S** — save image (filename printed in the console)
- **Space** — re-center the camera on the original look-at

## Theory

### Path Tracing Loop

Each sample for a pixel is a **path**: a ray that leaves the camera and bounces through the scene until it hits a light, misses everything, or runs out of bounces.

Per iteration, the GPU:

1. **Generates** one camera ray per pixel (`generateRayFromCamera`)
2. **Intersects** that ray against the scene (`computeIntersections`)
3. **Shades** the hit and updates path. Afterwards either terminates or scatters a new ray (`shadeMaterial` / `scatterRay`)
4. **Compacts** away finished paths and repeats until no paths remain or `DEPTH` is reached
5. **Accumulates** the path’s contribution into the image buffer

Throughput starts at `(1,1,1)` and is multiplied by the BRDF at each bounce, so later bounces contribute less light.

### Camera Ray Generation

For pixel `(x, y)`, a ray originates at the camera position and aims through that pixel. The direction is built from the camera basis (`view`, `right`, `up`) and the pixel’s offset from the image center, scaled by `pixelLength`.

With antialiasing enabled, `(j_x, j_y)` are uniform random offsets in `([0,1))`, so each iteration samples a slightly different point inside the pixel.

Each path also stores `remainingBounces = DEPTH` and a `pixelIndex` so its final color can be written back to the correct pixel.

### Intersections

`computeIntersections` walks every geometry in the scene and keeps the closest hit (`t_min`):

- **Spheres / cubes** — intersection tests with inverse transforms
- **Meshes** — each candidate triangle is tested with **Möller–Trumbore**

### Möller–Trumbore Triangle Test

Moller–Trumbore finds the ray to triangle intersection by solving for barycentric coordinates and distance without pre computing and storing extra data for the plane equation of the triangle.

**Why this over other mesh tests?**
Moller–Trumbore:

- Avoids a separate plane equation and projection step
- Uses only vector ops that map well to GPUs
- Rejects many misses early via barycentric bounds
- Needs only the three vertices (no precomputed plane or edge equations)

## Optimizations

### Stream Compaction
After each bounce, many paths are already finished (hit a light, missed the scene, or hit depth 0). Without compaction, every later bounce still launches a thread for those dead paths.
With `COMPACTION` on, `thrust::partition` splits the path buffer into **alive** and **terminated**. Terminated paths are gathered into the image (`finalGather`), then `num_paths` shrinks to only the alive count. Later bounce kernels therefore run fewer threads, which cuts wasted intersection and shade work as the path wavefront dies off.

### Material Sorting
Shading branches on material type (diffuse vs specular vs glossy vs emit). If neighboring threads in a warp hit different materials, those branches diverge and serialize.
With `MATERIAL_SORT` on, after intersection we extract each path’s material id, then `thrust::sort_by_key` reorders both the intersection and path arrays so paths with the same material sit together. The shade kernel then runs with less warp divergence.

### Mesh AABB Culling (`MESH_ACCEL_BBOX`)
Before testing any triangle of a mesh, the ray is tested against that mesh’s axis-aligned bounding box. If the ray misses the box (or hits it beyond the closest hit so far), the entire mesh is skipped.
This is cheap and helps a lot when many rays never come near a mesh (e.g. wall bounces in a Cornell box). If the ray **does** hit the box, every triangle in the mesh is still tested — so culling alone does not help once you are inside a dense mesh.

### Octree (`MESH_ACCEL_OCTREE`)
Each mesh builds a CPU-side octree (max depth 8, leaf target ≤ 16 triangles). Triangles that straddle child boxes are referenced in every leaf they touch (exact triangle–box tests during build). The tree is flattened to `OctNode[]` + a leaf triangle-index buffer and uploaded to the GPU.
On the GPU, traversal is iterative with a fixed stack:
1. Pop a node; skip it if its AABB misses the ray (or is farther than the closest hit).
2. **Leaf:** run Möller–Trumbore only on that leaf’s triangles.
3. **Interior:** push children in **nearest-first** order (direction-based `nearMask`) so closer hits are found early and farther nodes can be pruned.
Compared to bbox-only mode, most of the mesh’s triangles are never tested for a given ray.




### Stats:

#### Benchmark for regular optimization measurement:

<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/9284b49c-2e99-4d8b-ac84-b1b767dd4ea1" />

#### Benchmark for mesh optimization measurement:

<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/234d4adf-4e29-4a50-9d1e-e21381068047" />


#### Non-Mesh optimization methods:

<img width="1482" height="888" alt="image" src="https://github.com/user-attachments/assets/4a4377d5-01dc-4421-a116-e2d68b3f3a17" />

<img width="1576" height="1070" alt="image" src="https://github.com/user-attachments/assets/01049566-780f-4007-8dfa-a28f65977aa7" />


#### Mesh optimization methods:

<img width="1768" height="1058" alt="image" src="https://github.com/user-attachments/assets/fce963b8-9b2a-4fde-a883-fde23bf80aef" />

<img width="1572" height="1070" alt="image" src="https://github.com/user-attachments/assets/d131b7c1-500c-404b-9c55-e916a3515249" />



### Results

**Material sorting vs compaction only.** On the chessboard scene enabling material sorting was **marginally slower** than compaction alone. Shading in this tracer is cheap (a small branch in `scatterRay`), so the divergence savings are small. The sort cost likley exceeds the time saved. Sorting would matter more with heavier shaders or many more material IDs. Here compaction is the optimization that shows up clearly.

**Octree vs AABB culling.** AABB culling only lets you avoid paying the O(triangles) cost when the ray doesn't interest the bounding box of teh mesh. The octree runs triangle tests only inside partitioned leaves. On mesh-heavy scenes that means far fewer Moller Trumbore tests per ray, which is why octree mode shows **large speedups** over bbox-only.


## Visuals

### Materials

All non emitting surfaces go through the same scatter step in `scatterRay`. On a hit, the path either takes a **reflective lobe** or a **diffuse lobe**, then multiplies throughput and continues.

**Diffuse bounce**: Rays fan out randomly, so neighboring pixels see different parts of the room. Sharp reflections disappear and the surface looks matte (blue sphere).

**Reflective bounce**: With `ROUGHNESS = 0` this is a perfect mirror. Every path that hits the same spot continues in the same direction (mirror sphere).

**Reflectivity** (`hasReflective` / `REFLECTIVITY`) = probability of choosing the reflective lobe vs the diffuse lobe on that bounce.

- `0` → always diffuse (pure Diffuse material)
- `1` → always reflective (Specular materials)
- in between = Glossy: some paths mirror, some diffuse. Over many samples this looks like a tinted base with a clear coat. Red ball (`0.25`) is mostly matte with some glint; teal ball (`0.5`) shows a stronger mirror coat.

**Roughness** — only applied when the reflective lobe is chosen. The mirror direction is blended toward a random hemisphere sample.

`dir = normalize(mix(mirrorDir, hemisphereSample, roughness))`

- `0` → pure mirror
- higher values → reflection cone spreads out making soft metal (gold ball at `0.3`, teal ball at `0.2`)

Emitting surfaces skip scatter.

`scenes/cornell_5spheres.json` puts five spheres of various materials in one Cornell box.

| Sphere      | Material | Bounce setup                                                |
| ----------- | -------- | ----------------------------------------------------------- |
| Mirror ball | Specular | Reflectivity 1, roughness 0 always perfect reflect          |
| Gold ball   | Specular | Reflectivity 1, roughness 0.3 always reflect, but blurred   |
| Blue ball   | Diffuse  | Reflectivity 0 always hemisphere sample                     |
| Red ball    | Glossy   | Reflectivity 0.25 mostly diffuse, some sharp coat           |
| Teal ball   | Glossy   | Reflectivity 0.5, roughness 0.2 half coat, soft reflections |

<div align="center">
<img width="720" alt="Cornell box with five material spheres" src="https://github.com/user-attachments/assets/7784f677-4042-4f8c-89f0-96fa762defbc" />
<br>
<em>Five materials under shared lighting </em>
</div>

### Antialiasing

Without AA, every camera ray aims at the pixel center, so silhouette edges and high contrast boundaries stay jagged. With `ANTIAIASING` enabled, each iteration jitters the ray within the pixel (`j_x, j_y ∈ [0,1)`). Progressive averaging over iterations softens those edges.

<div align="center">

<table>
  <tr>
    <td align="center" width="50%">
      <strong>No antialiasing</strong>
    </td>
    <td align="center" width="50%">
      <strong>With antialiasing</strong>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img width="400" alt="No AA full frame" src="https://github.com/user-attachments/assets/d754900e-9e3b-4d2c-8c94-364933a1d9b2" />
    </td>
    <td align="center" width="50%">
      <img width="400" alt="AA full frame" src="https://github.com/user-attachments/assets/894f3d0f-8c74-47d4-ace4-568d6b747c40" />
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img height="280" alt="No AA crop" src="https://github.com/user-attachments/assets/46104194-dc17-450b-bad2-bf42850ad075" />
      <br>
      <em>Crop: hard stairsteps along the edge</em>
    </td>
    <td align="center" width="50%">
      <img height="280" alt="AA crop" src="https://github.com/user-attachments/assets/be472e69-37bf-48b3-b26d-462cb48d430c" />
      <br>
      <em>Crop: softer boundary after jittered samples</em>
    </td>
  </tr>
</table>

</div>

## Credits

### Libraries

- **[tinygltf](https://github.com/syoyo/tinygltf)** Used for glTF 2.0 mesh loading (`.gltf` / `.glb`).  
  Licensed under the [MIT License](https://opensource.org/licenses/MIT).

### 3D Models

Models sourced from the [Khronos glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets) repository:

| Model      | Asset                                                                                                      |
| ---------- | ---------------------------------------------------------------------------------------------------------- |
| **Duck**   | [Duck](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Duck)                           |
| **Dragon** | [DragonAttenuation](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/DragonAttenuation) |
| **Skull**  | [ScatteringSkull](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/ScatteringSkull)     |

Used AI agents to generate JSON scene files (Model: Claude Opus 5.5)
