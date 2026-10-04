# CUDA Path Tracer
**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Rayyan Shaik - [LinkedIn](https://www.linkedin.com/in/rayyan-shaik)
* Tested on:  Windows 11, AMD Ryzen 7 5700H @ 3.3GHz 32GB, Mobile RTX 3070 (Personal Computer)

![neon_space.png](/img/neon_space.png)

## Overview
This CUDA path tracer renders 3D scenes with realistic lighting, reflections, glass, and depth of field. It supports OBJ and glTF models and includes several toggleable optimizations for comparing rendering performance.Gi

## Features

### Arbitrary `.GLTF` / `.GLB` & `.OBJ` Mesh Loading

Using [tinygltf](https://github.com/syoyo/tinygltf/) and [tinyobj](https://github.com/syoyo/tinyobjloader) `.gltf`, `.glb`, and `.obj` files can be parsed on the CPU. I implemnted the conversion from the parsed geometry into world-space triangle, which are then uploaded to GPU memory for ray-triangle intersection testing.

The glTF loader traverses the selected scene's node hierarchy, combining parent and child transforms with the object's scene-defined transform. It supports indexed and non-indexed triangle geometry. For OBJ files, tinyobjloader triangulates polygon faces, and my laoder extracts and transforms their vertices.

Each mesh gets a (world-space) axis-aligned bounding box (AABB). With bounding-volume culling enabled, rays that intersect the box are tested against its triangles. this optimization this toggleable.

In the current implementation, importing is geometry-only: materials are assigned through the scene JSON rather than from model data itself. Surface normals are calculated via triangle geometry.

| Glass Bunny Scene | Tabletop Scene | Cornell Fish Scene |
|---|---|---|
| ![glass_bunny.png](/img/glass_bunny.png) | ![tabletop_2000_samp.png](/img/tabletop_2000_samp.png) | ![cornell_fish_diffuse_500_samp.png](/img/cornell_fish_diffuse_500_samp.png) |

#### Mesh Bounding Volume Culling Performance
![meshbvculling](/img/graphs/meshbvculling.png)

Mesh bounding volume culling proved to be a significant optimization on complex scenes, as shown by the performance speed up in the tabletop scene with it on versus off. The benefit is larger in complex scenes because rays that miss a mesh’s bounding box skip all of its triangle tests. For a simple cube, there are few triangles to test, so culling saves little work.

File parsing and tirangle conversion already runs on the CPU. We could improve this by implementing a BVH, reducing triangle tests across the whole mesh.

### Physically Based Depth of Field
I implemented depth of field using the model described in [PBRTv4](https://pbr-book.org/4ed/Cameras_and_Film/Projective_Camera_Models#TheThinLensModelandDepthofField).

For each camera ray, the point at which its pinole direction intersects the focal plane is calculated. Concentric disk sampling is then used to select a uniformly distirbuted point on the circular lens aperture. The ray's origin is moved to this point, and its direction is adjusted to pass through the original focal-plane intersection.

As the samples accumulate, objects near the focal plane remain sharp, while objects closer to or further from the camera appear blurred.

The scene JSON allows for the configuration of two camera settings:
- `LENS_RADIUS`: controls aperture size
- `FOCAL_DISTANCE`: Distance to the focal plane from camera "eye"

| Cornell Scene | Cornell Scene + DOF | Tabletop Scene | Tabletop Scene + DOF |
| --- | --- | --- | --- |
| ![](/img/cornell_2_spheres_dof_2000_samp.png) | ![](/img/cornell_2_spheres_2000_samp.png) | ![](/img/tabletop_2000_samp.png) | ![tabletop-dof](/img/tabletop_dof.png) |

#### DoF Performance
![dofperformance](/img/graphs/dofperf.png)

Depth of field adds a small performance cost because each camera ray requires aperture sampling, a lens-origin offset, and a recalculated direction toward the focal plane. These operations occur only during camera-ray generation, so their overhead is relatively small compared with tracing the full path.

A CPU implementatino would use the same aperture sampling logic, while the GPU generates many camera rays at the same time.

### Reflections & Refraction
I implemented mirror reflection using `glm::reflect`, and glass refraction using `glm::refract` for [Snell's Law](https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission).

Glass surfaces use [Schlick's approximation](https://en.wikipedia.org/wiki/Schlick%27s_approximation) to calculate the probability of reflection versus transmission. Each interaction randomly selects one direction, with the total internal reflection always producing a reflected ray. The surface orientation determines if the ray is entering or exiting, enabling the correct refractive-index ratio to be used.

Glass IOR (index of refraction) is configurable through the scene JSON. Transmitted paths also include the correction for "accounting for non-symmetric scattering across refractive boundaries" as described in [PBRTv4](https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF).

| Diffuse Fish | Mirror Fish | Glass Fish|
| --- | --- | --- |
| ![diffusefish](/img/cornell_fish_diffuse_500_samp.png) | ![metalfish](/img/cornell_fish_metal_500_samp.png) | ![glassfish](/img/cornell_fish_glass_500_samp.png) |

#### Reflections & Refractions Performance
![reflrefrperformance](/img/graphs/diffusemirrorglass.png)

Reflection and refraction change ray directions and therefore how long paths remain in the scene. Although glass requires additional Fresnel and refraction calculations, its lower measured render time likely reflects more paths escaping (open scene) or reaching lights sooner. These results measure the entire path, not just the cost of evaluating each material.

Both GPU and GPU implemenations would save work by terminating low-contribution paths. But on the GPU, compaction helps reduce the segment processing range when a path is terminated.

### Russian Roulette Path Termination
I implemented [Russian roulette](https://pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Path_Tracing) to probibalistically terminate low-contribution paths after their initial bounces (Specifically when `bounces > 3`). The termination probability is based on the brightness/luminance of the path's accumulated color, with dimmer paths more likely to terminate. Surviving paths also have their color divided by their survival probability, to preserve th expected result.

| Cornell Closed | Cornell Closed w/ Russian Roulette |
| --- | --- |
| ![ccnorr](/img/cornell_closed_2_spheres_2000_samp.png) | ![ccrr](/img/cornell_closed_2_spheres_rr_2000_samp.png) |

#### Russian Roulette Performance
![rrperf](/img/graphs/rrrperf.png)

Russian roulette improves performance by terminating low-contribution paths early, reducing later-bounce intersection and shading work. This increases variance per sample, producing noticeably more grain (while producing the expected result) - this is seen in the shadows in the images above.

## Base Features
### Stochastic Antialiasing

![aa](/img/aa.png)
Stochastic antialiasing randomly jitters camera rays within each pixel. Averaging these samples captures partial coverage along object boundaries, reducing the apperance of jagged edges as the image converges

### Stream Compaction
After each bounce, stream compaction groups active paths at the front of the buffer using `thrust::partition`. Subsequent intersection and shading kernels process only this smaller range, avoiding launches over terminated paths. Completed paths remain in the buffer for final image accumulation.

A CPU could partition paths without the GPU kernel overhead, but the GPU can more effectivley process large buffers in paralle. Maybe skipping compaction on early bounces when few paths are terminated could help reduce overhead.

![sc1](/img/graphs/sc1.png)

The timing graph shows that partitioning overhead outweighs the savings in both simple Cornell scenes. In Tabletop, more expensive intersection work, with multiple meshes, makes processing a smaller path range worthwhile, so compaction improves performance.

![sc2](/img/graphs/sc2.png)

The active-path graph shows that rays escape quickly in open Cornell, while most paths survive until the depth limit in closed Cornell.

### Material Sorting
Before shading each bounce, `thrust::sort_by_key` groups intersections by material ID while reordering their corresponding paths together. This aimst o place similar shading operations in neighboring GPU threads, reducing branch divergence.

![matsort](/img/graphs/matsort.png)

Material sorting adds a sorting pass at every bounce. In these scenes, the shading operations are relatively cheap, so reduced branch divergence saves less time than sorting costs.

## References & Credits
### Third-Party Code
- [tinygltf](https://github.com/syoyo/tinygltf/)
  - `external/tiny_gltf/tiny_gltf.h`
  - `external/tiny_gltf/json.hpp`
- [tinyobj](https://github.com/syoyo/tinyobjloader)
  - `external/tinyobjloader/tiny_obj_loader.h`

### Model Credits
- **Furniture models:** [Kenney Furniture Kit](https://kenney.nl/assets/furniture-kit), created by Kenney.
- **Spaceport models:** [Kenney Space Kit](https://kenney.nl/assets/space-kit), created by Kenney.
- **Stanford Bunny:** Stanford University Computer Graphics Laboratory. Original dataset available from the [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/).
- **Barramundi Fish:** Created by Microsoft, distributed through [Khronos glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/BarramundiFish)
- **Scattering Skull:** Created by Vladimir Petkovic, Adobe. Distributed through [Khronos glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/ScatteringSkull)
- **Box (`box.glb`):** 2017 Cesium, distributed through [Khronos glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Box)

### References
- https://pbr-book.org/4ed/Cameras_and_Film/Projective_Camera_Models#TheThinLensModelandDepthofField
- https://pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions#SampleUniformDiskConcentric
- https://pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions#fragment-Mapmonouto-112andhandledegeneracyattheorigin-0
- https://pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions#fragment-Applyconcentricmappingtopoint-0
- https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/Russian_Roulette_and_Splitting
- https://pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Path_Tracing#PossiblyterminatethepathwithRussianroulette
- https://en.wikipedia.org/wiki/Relative_luminance
- https://en.wikipedia.org/wiki/Schlick's_approximation
- https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission#fragment-Computecostheta_romantusingSnellslaw-0
- https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission

## Additional
### CMAKE Changes
CMake was updated to include the custom mesh-loader source files and add the TinyGLTF and tinyobjloader header directories.