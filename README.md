CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Grace Tan
* Tested on: Windows 11 Home 25H2 (Build 26200), AMD Ryzen 9 8945HX @ 2.50GHz, 16GB RAM, NVIDIA GeForce RTX 5060 Laptop GPU 8GB

# CUDA Monte Carlo Path Tracer

Monte Carlo Pathtracing is a rendering technique that produces photorealistic images by simulating real-life light transport modeled by the rendering equation. Rays fired from the camera accumulate color and intensity based on interactions with different materials in the scene, ultimately producing an average of their contributions for each pixel. 

Unlike simple rasterization, a different rendering technique that picks the color of geometry closest to the camera at each pixel, pathtracing allows us to render scenes with increased physical accuracy and complexity. 

<a name="room-scene"></a>
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/envmap.2026-10-05_15-45-41z.5000samp.png" width="500"><br>
    </td>
    <td align="center">
      <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/envmap.2026-10-04_11-55-43z.5000samp.png" width="500"><br>
    </td>
  </tr>
</table>
Note: handling emission in gltf loading not implemented until after the coding deadline

## Table of Contents

- [Features](#features)
  - [Anti-Aliasing](#anti-aliasing)
  - [Physically Based Materials](#physically-based-materials)
  - [Triangle Intersection and BVH](#triangle-intersection-and-bounding-volume-hierarchy)
  - [Mesh and Texture/Material Loading](#mesh-and-texturematerial-loading)
  - [Environment Mapping](#environment-mapping)
  - [Depth of Field](#depth-of-field)
  - [Intel Open Image Denoise](#intel-open-image-denoise)
- [Performance Analysis](#performance-analysis)
- [3rd Party Resources](#3rd-party-resources)

## Features

### Anti-Aliasing

Aliasing is the jagged, staircase effect that appears when smooth borders are rendered on a pixel display. Anti-aliasing addresses it by sampling a pixel's neighborhood and averaging the colors, which softens edges.

Because a path tracer already fires multiple rays through each pixel, anti-aliasing comes essentially for free: each ray is simply jittered within the pixel.

### Physically Based Materials

There is one shading function for opaque materials (metals, plastics) and one for transmissive materials (glass).

#### Opaque Materials: Cook-Torrance BRDF

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/envmap.2026-10-05_16-48-47z.5000samp.png" width="333"><br>
      <sub>Diffuse - Metallic: 0.0, Roughness: 1.0</sub>
    </td>
    <td align="center">
      <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/envmap.2026-10-05_16-59-59z.5000samp.png" width="333"><br>
      <sub>Semi-Gloss - Metallic: 1.0, Roughness: 0.5</sub>
    </td>
    <td align="center">
      <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/envmap.2026-10-05_16-50-53z.5000samp.png" width="333"><br>
      <sub>Glossy - Metallic: 1.0, Roughness: 0.0</sub>
    </td>
  </tr>
</table>

Diffuse and specular reflection are modeled with a Cook-Torrance BRDF.

The specular lobe is based on the microfacet theory, which assumes that a surface is composed of many small, flat facets that determine the appearance of a material. To model the distribution of microfacet normals, I used the GGX distribution function. The fresnel term models increased reflectivity at grazing angles and the smith geometry term corrects for camera-aligned facets that are masked and/or shadowed by neighboring geometry. 

[Joe Schutte’s probability distribution simplified version](https://schuttejoe.github.io/post/ggximportancesamplingpart1/) was incredibly helpful for implementing the specular lobe. The diffuse lobe was implemented simply using uniform cosine-weighted hemisphere sampling.

I used this shading model
$$
f(\omega_o, \omega_i) = k_d \frac{R}{\pi} + k_s \frac{D\,G}{4(\omega_o \cdot n)(\omega_i \cdot n)}
$$
, published by Epic as a practical PBR model for real-time rendering, for the full metallic-roughness BRDF. `k_s` is simply the fresnel term and `k_d = 1 - k_s`.

I stochastically sampled the diffuse and specular lobes by comparing a uniformly sampled probability `p` and the luminance of the multi-channel fresnel term. To compute this, I interpolated between the metallic fresnel term, which takes the material’s albedo as the base reflectance, and `0.04`, the base reflectance for non-metallic materials.

#### Transmissive Materials

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/transmissive_pbr.2026-10-05_17-12-15z.5000samp.png" width="250"><br>
      <sub>Glass in Environment Map</sub>
    </td>
    <td align="center">
      <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/cornell.2026-10-05_17-19-18z.5000samp.png" width="250"><br>
      <sub>Glass in Cornell Box</sub>
    </td>
    <td align="center">
      <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/transmissive_pbr.2026-10-05_17-13-11z.5000samp.png" width="250"><br>
      <sub>Tinted Glass in Environment Map</sub>
    </td>
    <td align="center">
      <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/cornell.2026-10-05_17-22-54z.5000samp.png" width="250"><br>
      <sub>Tinted Glass in Cornell Box</sub>
    </td>
  </tr>
</table>

For transmissive materials, I referenced [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html#dielectrics/refraction) to learn about Snell’s law, which describes the direction of refracted rays. Refraction is mixed with perfect reflection based on the fresnel term, which is again approximated with Schlick but instead given a base reflectance based on the material’s IOR. Additionally, when a ray travels from a higher IOR to a lower IOR at a large angle, it experiences total internal reflection and cannot be refracted due to an imbalance in Snell’s law. In this case, the ray is also reflected. 

Finally, I used Beer’s law of absorption, which says that light intensity decays exponentially with increasing distance traveled through an absorbing medium, to calculate material attenuation. When absorption differs per channel, the ray becomes tinted and gives the transmissive material color. 

### Triangle Intersection and Bounding Volume Hierarchy

To support arbitrary meshes, the tracer has triangle intersection and a BVH.

**Triangle intersection** uses the [Möller-Trumbore algorithm](https://scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html). Equating the ray `o + t·d` with the barycentric form of a point on a triangle lets us solve for `t`, `u`, and `v`: the distance along the ray and the barycentric coordinates of the hit. `u` and `v` are later used to interpolate UVs for texture sampling.

<p align="center">
  <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/img/bvh_diagram.png" width="400"><br>
</p>

**Bounding Volume Hierarchy (BVH)** is a tree that recursively partitions the scene into increasingly smaller axis-aligned bounding boxes. Since scenes can contain hundreds of thousands of triangles, testing every primitive for every ray would be far too slow; thus, we use a spatial acceleration structure to speed up intersection testing. Following [Jacco Biker's blog](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/), I implemented a simple BVH that significantly improved performance for more complex scenes (see [Performance Analysis](#performance-analysis)).

### Mesh and Texture/Material Loading

Meshes are loaded from `.gltf` / `.glb` files using [tinygltf](https://github.com/syoyo/tinygltf). The scene graph is traversed to extract triangle data: positions, normals, and UVs. Understanding the structure of a gltf file was crucial in implementing loading - this [reference](https://www.khronos.org/files/gltf20-reference-guide.pdf) was extremely helpful for me.

Material data such as image textures and roughness is also optionally extracted and stored. Alternatively, a material can be specified directly in the scene's JSON file.

### Environment Mapping

<p align="center">
  <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/envmap.2026-09-30_06-16-36z.5000samp.png" width="400"><br>
</p>

Environment maps in `.exr` format are loaded with [tinyexr](https://github.com/syoyo/tinyexr). When one is present, rays that miss all geometry sample the map as an infinite light source instead of contributing black.

### Depth of Field

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/dof.2026-10-05_18-11-25z.5000samp.png" width="333"><br>
      <sub>Focal Length: 8, Aperture Radius: 0.2</sub>
    </td>
    <td align="center">
      <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/dof.2026-10-05_18-09-22z.5000samp.png" width="333"><br>
      <sub>Focal Length: 8, Aperture Radius: 0.3</sub>
    </td>
    <td align="center">
      <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/dof.2026-10-05_18-13-21z.5000samp.png" width="333"><br>
      <sub>Focal Length: 8, Aperture Radius: 0.5</sub>
    </td>
  </tr>
</table>

A physically based lens camera produces [depth of field](https://blog.demofox.org/2018/07/04/pathtraced-depth-of-field-bokeh/). The focal plane sits at a fixed distance from the camera, and a random point is uniformly sampled on a circular aperture of a given radius. That point determines the ray origin and direction.

This is an improvement over firing a ray straight from the camera through each pixel, which models a pinhole camera that can only produce perfectly sharp images.

### Intel Open Image Denoise

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/envmap.2026-10-05_18-22-26z.50samp.png" width="500"><br>
      <sub>50 iterations, not denoised</sub>
    </td>
    <td align="center">
      <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/envmap.2026-10-05_18-25-17z.50samp.png" width="500"><br>
      <sub>50 iterations, denoised</sub>
    </td>
  </tr>
</table>

An AI-based denoiser from [Intel Open Image Denoise](https://www.openimagedenoise.org/) greatly reduces the number of iterations needed for the image to converge to a clean result.

The denoised version of the current accumulation is output every 10 frames, so results can be viewed continuously without a significant performance cost.

## Performance Analysis

To evaluate performance, I compared average FPS for different scenes using the same set of optimizations. The Cornell Scene features a Stanford Dragon inside a Cornell box and the Environment Map Scene features the same dragon model with an environment map. 

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/cornell.2026-10-06_01-52-25z.5000samp.png" width="500"><br>
      <sub>~100k tris</sub>
    </td>
    <td align="center">
      <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/envmap.2026-10-05_16-48-47z.5000samp.png" width="500"><br>
      <sub>~100k tris</sub>
    </td>
  </tr>
</table>

### Stream Compaction

<p align="center">
  <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/img/Stream%20Compaction%20Performance%20for%20Different%20Scenes.png" width="800"><br>
</p>

Stream compaction can be used to partition the paths such that terminated paths are separated from paths are still alive. Instead of launching one thread per path every bounce, we can reduce kernel launch size and only work on paths that are still alive. 

From the graphs, we can observe a significant performance improvement from using stream compaction in the open Environment Map Scene versus the closed Cornell Scene. Since paths terminate more quickly in an open scene (randomly generated directions easily miss scene geometry), it is clear that open scenes benefit more from stream compaction, which specifically handles those dead paths and removes them from the work queue.

### Material Sorting 

<p align="center">
  <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/img/Material%20Sorting%20Performance%20for%20Different%20Scenes.png" width="800"><br>
</p>

To evaluate material sorting, we also measure average FPS for the [Room Scene](#room-scene), which contains a greater number of materials than the Cornell Scene and the Environment Map Scene. However, in all three scenes, material sorting decreases performance. Material sorting uses Thrust's `sort_by_key` over every live path on every bounce, which can increase memory traffic significantly due to many read/write passes over data. In this case, the overhead of sorting outweighs any gain from reduced divergence. 

We can observe that material sorting hurts the Room Scene the least. When shading kernels become more complex and the number of materials increases significantly, material sorting could potentially increase performance.

### Bounding Volume Hierarchy

<p align="center">
  <img src="https://github.com/grace866/Project3-CUDA-Path-Tracer-CIS565/blob/post-deadline/img/BVH%20Performance%20for%20Different%20Scenes.png" width="800"><br>
</p>

The speedup from using BVH is enormous! It is clear that naively iterating through every triangle in the scene is unreasonable and would increase render time considerably. When handling arbitrary meshes built from triangle primitives, it is absolutely necessary to improve intersection testing with a spatial acceleration structure. 

## 3rd Party Resources 

### Assets 
| Model | Source |
|-------|--------|
| Concrete cat statue | [Poly Haven](https://polyhaven.com/a/concrete_cat_statue) |
| Room | [CGTrader](https://www.cgtrader.com/3d-models/interior/bedroom/the-room-f749b9ec-815a-4e98-84d5-16ebbb2c9bd3) |
| Crate | [CGTrader](https://www.cgtrader.com/3d-models/interior/bedroom/the-room-f749b9ec-815a-4e98-84d5-16ebbb2c9bd3) |
| Chair | [CGTrader](https://www.cgtrader.com/free-3d-models/furniture/chair/office-chair-e29dcbbd-0abe-4ca5-866a-c8f75a2fc11f) |
| Neon signs | [CGTrader](https://www.cgtrader.com/free-3d-models/exterior/street-exterior/free-scifi-neon-text) |
| Cabinet | [CGTrader](https://www.cgtrader.com/free-3d-models/furniture/kitchen-cabinet/john-lewis-slatted-2-door-cabinet) |
| Tall Potted Plant | [CGTrader](https://www.cgtrader.com/free-3d-models/plant/pot-plant/potted-plant-birds-of-paradise) |
| Monstera Plant | [CGTrader](https://www.cgtrader.com/free-3d-models/interior/living-room/plant-monstera-vase) |
| Small Potted Plant | [CGTrader](https://www.cgtrader.com/free-3d-models/plant/pot-plant/bush-pot-plant) |
| PC | [CGTrader](https://www.cgtrader.com/free-3d-models/electronics/computer/pc-in-apple-imac-style) |
| Rug | [CGTrader](https://www.cgtrader.com/free-3d-models/interior/living-room/vivense-elite-boucle-cream-carpet) |
| Shelves | [CGTrader](https://www.cgtrader.com/free-3d-models/interior/bedroom/hexagonal-shelves) |
| Mirror | [CGTrader](https://www.cgtrader.com/free-3d-models/interior/hall/round-mirror-free) |
| Bed | [BlendSwap](https://blendswap.com/blend/17079) |
| Dragon | [KhronosGroup/glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets) |

- Dusk Sky from [Poly Haven](https://polyhaven.com/a/qwantani_dusk_2_puresky) and Night Sky from [Poly Haven](https://polyhaven.com/a/qwantani_night_puresky)

### Libraries 
- [TinyEXR](https://github.com/syoyo/tinyexr)
- [TinyGLTF](https://github.com/syoyo/tinygltf)
- [Intel Open Image Denoise](https://github.com/RenderKit/oidn) 
