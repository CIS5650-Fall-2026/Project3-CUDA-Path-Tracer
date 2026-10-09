CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

<img width="1920" height="1200" alt="greenpath" src="https://github.com/user-attachments/assets/7ba13637-5a06-495f-a444-2a4e0188da04" />

* Lilian Xiao
  * [LinkedIn](https://www.linkedin.com/in/lilian-xiao/), [personal website](https://lilianxiao.carrd.co/), [art website](https://lilianxvis.carrd.co/)
* Tested on: Windows 11, Intel(R) Core(TM) Ultra 9 185H (2.50 GHz), 16.0 GB RAM, NVIDIA GeForce RTX 4070 Laptop GPU (8 GB)
Intel(R) Arc(TM) Graphics (128 MB) (Personal Laptop)

## Important Modifications

I made a few changes to the cmakelists, including adding extra files (such as procedural.cu and stb_image.h), as well as adding the below code block to fix some cuda issues.

`if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
        -Xcompiler=/Zc:preprocessor")
endif()`

## Credits

All 3D models and textures used as assets are made by me, with the exception of the HDRI maps, Asaro head (created by [Fabiano Araujo](https://sketchfab.com/3d-models/asaro-head-9d26548182f8465a8e97371a9170561e)), and the Mario model and associated texture (belonging to Nintendo.)  Hornet and the Knight are characters from Hollow Knight, a game series by Team Cherry.

I use [tinyobj](https://github.com/tinyobjloader/tinyobjloader) and [stb_image](https://github.com/nothings/stb) for obj and image loading support.

## Overview

Path tracing provides a more physically accurate simulation of light by considering its reflection, refraction, transmission, absorption, and scattering properties.  In basic ray tracing, a ray is emitted from the camera/eye for each pixel of the image.  The intersections of these rays with objects in the scene, depending on the surface behaviors of the objects, can result in secondary rays.  For instance, when the view ray hits a specular object, a secondary ray is emitted.  Ray tracing supports direct illumination and approximates the behavior of light in terms of reflection, refraction, and shadowing.

On the other hand, path tracing can simulate global illumination; in nature, light can come from anywhere, and this results in both direct and indirect lighting.  While ray tracing is a deterministic process, Monte Carlo path tracing is stochastic; we trace multiple rays per pixel because light in nature is random.  Path tracing for more iterations allows the render to converge to a less noisy and more usable image, since each iteration is responsible for tracing a random light path, and the average over these iterations grows closer to the Monte Carlo estimate.

The following two renders of the same Cornell box scene are taken at 7 samples and 204 samples respectively.

<img width="800" height="800" alt="cornell 2026-10-08_17-13-56z 7samp" src="https://github.com/user-attachments/assets/b88680e4-6442-4d08-a844-df262f448bfa" />
<img width="800" height="800" alt="cornell 2026-10-08_17-13-56z 204samp" src="https://github.com/user-attachments/assets/82e55449-a7d9-4f10-b70f-37a104569bc1" />

## Core Features

A standard Monte Carlo pathtracer must trace a single path to completion before tracing the next path.  This project implements a CUDA pathtracer, which leverages the GPU to advance paths simultaneously at each bounce.  It uses kernels that run in parallel and buffers important data that these kernels index into.  For instance, there are kernels for generating rays from the camera, computing intersections, and applying material shading.  Scene data (geometry, triangles, textures, materials, BVH nodes, etc.), image accumulation, and path/intersection information are buffered into flat arrays, as kernels do not recurse via pointers.

### Stream Compaction

Thrust partitioning puts live paths in the front of the buffer.  Material sorting organizes intersections such that they are contiguous in memory by material type.  Within a kernel, we can expect different materials and BSDFs evaluations to finish at different times, but paired with idle threads, this results in a lot of branching.  Material sorting allows for memory adjacency, while stream compaction removes paths that have already been finished.  Overall, this results in fewer threads launched and intersection hits for the same material type being in the same group.

Notably, sorting by material is somewhat expensive.  It's not ideal to use for limited materials (for example, only 1-2 diffuse materials), but the tradeoff becomes worth it for scenes with a lot of materials (for example, having diffuse, reflective, transmissive, subsurface scattering, and procedural materials).

### Stochastic Sampled Anti-Aliasing

This was adapted from Paul Bourke's notes [here](https://paulbourke.net/miscellaneous/raytracing/).  Without anti-aliasing, the "edges" of an object appear to be sharp and jagged like a staircase.  This is because the pixels there must be the object's color or the background/not the object.  One might notice how the jaggies become more apparent for images with smaller resolution.

In each iteration, a random offset is picked using thrust rng and applied to the pixel coordinates.  Based on the rng seed, the offset will be different per iteration.  Averaging over iterations, this ultimately results in smoother edges.

<img width="224" height="224" alt="image" src="https://github.com/user-attachments/assets/d2bd508c-b7fd-417e-9a01-071bcdfb1655" />
<img width="224" height="224" alt="image" src="https://github.com/user-attachments/assets/5907ba53-ae47-47a4-9c55-fd042474c052" />

## Additional Features

Multiple additional features were implemented in this pathtracer, from materials to optimization techniques.

### Fresnel-Schlick Effects for Refraction

Reflective and transmissive materials using [Schlick's Approximation](https://en.wikipedia.org/wiki/Schlick's_approximation).

<img width="800" height="800" alt="cornell 2026-10-07_05-08-23z 412samp" src="https://github.com/user-attachments/assets/d441bc0f-b9b4-4481-bf61-897a3ea97009" />

<img width="800" height="800" alt="cornell 2026-10-07_05-09-38z 330samp" src="https://github.com/user-attachments/assets/2e33411a-8f01-4b87-b33f-d5b57c755a80" />

<img width="800" height="800" alt="cornell 2026-10-07_05-12-38z 362samp" src="https://github.com/user-attachments/assets/bbb4f789-b362-4abf-b9ca-b0ec2dc287b0" />

In the fresnel effect, light either passes through the material or is reflected.  This reflection is viewing-angle dependent, as reflection is greater at glancing angles.  Schlick's approximation evaluates the fraction of light reflected (F) and the fraction transmitted (1 - F). 

<img width="1200" height="1200" alt="cornell 2026-10-08_13-29-28z 201samp" src="https://github.com/user-attachments/assets/4b52573f-610f-420f-a865-7cad78653def" />

<img width="1200" height="1200" alt="cornell 2026-10-08_13-26-12z 88samp" src="https://github.com/user-attachments/assets/acfb655c-8a52-45ae-9cb8-5066712ed687" />

### Physically-Based Depth Of Field

<img width="1200" height="1200" alt="cornell 2026-10-07_10-55-15z 262samp" src="https://github.com/user-attachments/assets/53c81a42-34c3-4291-9bec-d0bba580a78c" />

Originally, the camera is a lensless pinhole that is perfectly focused on everything captured in the scene.  However, adding a lens radius and focal distance allows for one to control both the pinhole size and the distance in front of the viewing position where the "sharp" plane is.

Rays per pixel are guaranteed to pass through the same focal point, which is why originally, everything in the scene is focused.  To blur, jitter is applied, so rays will actually hit some randomized point on the lens.  Overall, the average will cover a range of the object, resulting in this blurring effect for anything that isn't at the focal distance. 

Top image: lens radius of 0.1 and focal distance of 5.  Bottom image: lens radius of 0.5 and focal distance of 5.

<img width="1200" height="1200" alt="silksong 2026-10-08_18-17-12z 38samp" src="https://github.com/user-attachments/assets/9b3ff6c3-840f-4098-920b-a6879919efe3" />
<img width="1200" height="1200" alt="silksong 2026-10-08_18-18-50z 50samp" src="https://github.com/user-attachments/assets/b98e38e1-1f4f-4198-8f45-f721a7ff145f" />

### Procedural Textures

Perlin and Voronoi noise are available as procedural texturing materials.  The images below are: fbm perlin, fbm perlin input voronoi, and voronoi materials, respectively.

<img width="800" height="800" alt="cornell 2026-10-07_01-33-44z 190samp" src="https://github.com/user-attachments/assets/caa320cc-d633-4274-aac6-95cbb54645b6" />
<img width="800" height="800" alt="cornell 2026-10-07_02-03-28z 201samp" src="https://github.com/user-attachments/assets/f64239f7-6793-4c8f-b0de-ddbd8d8e1d43" />
<img width="800" height="800" alt="cornell 2026-10-08_18-28-45z 47samp" src="https://github.com/user-attachments/assets/f4d57da5-edc5-41ce-a696-3d0c48c550de" />

In Perlin noise, the entire space is subdivided into 3D unit cubes.  Then, a hashing function is used to randomly assign some direction to each corner of the cubes.  An input point has some distance along the gradient directions of the cube's 8 corners, and these values are blended using Ken Perlin's smooth fade function.

<img width="442" height="442" alt="image" src="https://github.com/user-attachments/assets/351dcfea-90e3-42f8-81c3-b076e5e876ca" />

<img width="441" height="400" alt="image" src="https://github.com/user-attachments/assets/6080cd61-fcc8-4c9f-8650-f5b9775895f5" />

(Images sourced from [here](https://rtouti.github.io/graphics/perlin-noise-algorithm))

Voronoi also divides space into unit cubes, but has a slighly different behavior.  Each cube has some randomly positioned point; each point is checked against the other 27 cubes to find the distance to the nearest other point, which creates a cell-effect of boundaries created between these "centroids".

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/fd553a1f-eac2-4cc0-be5b-bf9a3ec9a3d9" />

(Image sourced from [here](https://en.wikipedia.org/wiki/Weighted_Voronoi_diagram))

FBM stands for Fractal Brownian Motion and is a method of overlaying several octaves/layers of Perlin noise, depending on frequency and strength parameters for each succeeding layer.  As octaves increase, more fine detail can be achieved.

### Direct Lighting using Multiple Importance Sampling

First image: naive pathtracing.  Second image: with multiple importance sampling.

<img width="793" height="835" alt="image" src="https://github.com/user-attachments/assets/aa509407-54b6-441e-84ab-e86a9ff05e2a" />

<img width="787" height="835" alt="image" src="https://github.com/user-attachments/assets/57950c22-68f9-4646-9a76-c6c7686141a3" />

In naive pathtracing, light is only accumulated if a path hits an emissive surface.  Therefore, with just BRDF sampling, smaller emitters will converge rather poorly.  On the other hand, direct lighting explicitly traces to light sources as a guarantee, but as a result, larger emitters will converge poorly.  Multiple Importance Sampling (MIS) allows for these two sampling methods to have a weighted contribution, ultimately yielding images that are less noisy.  There is slightly higher cost per iteration due to testing another visibility ray per hit, but over time, converges much faster, especially in a closed scene.

<img width="605" height="374" alt="image" src="https://github.com/user-attachments/assets/21d4ba52-d156-493c-9eea-0dbdacc62c05" />

Here, we observe that performance appears to be better for open scenes.  This is expected, as paths tend to terminate sooner in an open scene, additionally with the presence of stream compaction, which removes inactive paths.  On the contrary, in a closed scene, all paths survive and thus do muchh more work (where the work in one iteration is approximately the summation of all active paths / all bounces).  Direct lighting involves firing secondary shadow rays, which have relatively the same cost as the initial intersection test, which also adds cost to a live path.

### Subsurface Scattering with Russian Roulette Path Termination

In these renders, Hornet's mask is an offwhite subsurface scattering material, and the dark 3D model of her head beneath the mask is slightly visible.  As the density parameter increases, the material becomes less permeable and closer to a regular opaque surface.

<img width="800" height="800" alt="cornell 2026-10-07_09-45-03z 146samp" src="https://github.com/user-attachments/assets/6b9e4b9a-303e-41ad-a309-e4126aeb6073" />

<img width="800" height="800" alt="cornell 2026-10-07_09-43-23z 174samp" src="https://github.com/user-attachments/assets/19268dc4-ceec-4be1-b6b4-0b8ea559854b" />

<img width="800" height="800" alt="cornell 2026-10-07_09-48-41z 181samp" src="https://github.com/user-attachments/assets/058922a3-8e7f-4b09-a7d2-fa9da8486db5" />

In nature, the effects of Subsurface Scattering (SSS) can easily be observed when a permeable or thin material is exposed to the sun and achieves a slightly translucent, reddish/saturated color.  This can be seen when a light shines against your fingers or your earlobe.  SSS treats an object's inside as a uniform gradient with scattered particles.  A path will enter the surface, perform a random walk, and leave the mesh.  Generally speaking, light enters a surface and scatters, then leaves at a random/different point.

Because random walks can easily have a lot of scattering steps, we need a way to control dim and less usable paths without completely removing them or their light contribution.  Russian Roulette path termination is a method where the path will survive only with a probability equal to its brightest color channel.  For example, a path with only around 5% strength will only have that probability of persisting, and if it does, it will be counted at full strength contribution.  This ultimately introduces marginally more noise, but it is reasonable for growing iterations.

### BVH Tree Acceleration Structure

<img width="609" height="263" alt="image" src="https://github.com/user-attachments/assets/69e1875b-4ec2-4c70-81d8-5edaa9a37f68" />

(Image sourced from [here](https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/))

BVH stands for "Bounding Volume Hierarchy" and is an tree optimization structure.  Suppose we have a triangulated mesh; then we find a bounding box that encloses the triangles.  Suppose a ray misses this box; then nothing inside the box can be hit either, and we can safely disregard the general group.

A BVH consists of the root node (which is a box enclosing the entire mesh), the interior nodes (which each have two children enclosed by boxes), and leaves (in this implementation, 4 enclosed triangles).  One can see how the mesh is subdivided into halves.

First, the bounding boxes are computed for the current group.  Then, we pick the longest axis to equally halve the triangles by their position, and repeat.  This results in a balanced tree of approximately log_2(n) depth, where n is the total number of triangles.  During building, this tree is reordered such that triangles belonging to the same leaf are oriented next to each other in the buffer.

The GPU is responsible for traversal and tests a ray against the root bounding box, continually tracking the closest hit.  This prevents every single ray from being tested against every single triangle, which can lead to humongous cost for very large and complex meshes.  With BVH, cost is proportional to the tree depth, not the total number of triangles.

<img width="604" height="372" alt="image" src="https://github.com/user-attachments/assets/208e5b7d-71fa-4f72-9a87-a79745bd070b" />

Performance is not affected by increasing the BVH stack size.  This is because stack size controls capacity and does not affect the work being done; that is, a BVH traversal will use entries based on the depth of the tree, and additional reserved space will not change this.

### Arbitrary Mesh Loading and Bounding Volume Intersection Culling

OBJ meshes are loaded using [tinyobj](https://github.com/tinyobjloader/tinyobjloader).  The mesh triangles are buffered and a BVH is created.  The idea is that when some ray hits a mesh, the triangle is tested with single-triangle intersection (Moller-Trumbore), and if there is a hit, barycentric blending is done for smooth shading.  Bounding volume intersection culling works basically the same as the aforementioned BVH acceleration method.  Suppose a mesh in the scene only occupies a small visible area.  Without culling, a ray will be tested against every mesh triangle, so for a complex mesh, this is rather undesirable.  Culling does a box test against the bounding box, and if the ray misses, then the rest of the mesh is negligible.

Testing using a scene where a complex mesh is only partially visible, without BVH acceleration:

Testing using a scene where a complex mesh is only partially visible, with BVH acceleration also enabled:

<img width="606" height="374" alt="image" src="https://github.com/user-attachments/assets/6b3451ec-f646-415e-ad9f-b08ff10a80b2" />

<img width="605" height="372" alt="image" src="https://github.com/user-attachments/assets/bacc9740-58a3-4add-9296-e319d7f49931" />

Culling in conjunction with BVH acceleration structures helps somewhat for meshes that are only partially in view.  For a mesh with n triangles, without culling, there will be n ray tests.  For an object that is completely missed, the number of ray tests will be drastically diminished (to only one if the initial hit misses.)

### Texture Mapping and Bump Mapping

Texture images are loaded with the help of [stb_image](https://github.com/nothings/stb), and added to a texel buffer.  Note that textures are RGBA to handle transparent images.  Bump mapping is a method of heightmapping, which allows for 2D surfaces to appear 3D.  Based on the texture's u and v directions, the surface rises in a certain direction, and is overall scaled by some bump strength.

First image: bump mapping applied over the textures on Hornet's cloak and needle to give a 3D effect.  Second image: a noisy, staticky bump map applied over Mario's textures.

<img width="800" height="800" alt="cornell 2026-10-07_09-16-05z 199samp" src="https://github.com/user-attachments/assets/269b09fb-34ef-4b6a-8a4b-ff939e26da45" />

<img width="800" height="800" alt="cornell 2026-10-07_08-21-41z 172samp" src="https://github.com/user-attachments/assets/df2a368c-448a-45f6-8f70-8a1438d4ef71" />

<img width="604" height="373" alt="image" src="https://github.com/user-attachments/assets/9e086615-e394-46f5-86a7-4de5224ed0f4" />

Relative to other features implemented, texture and bump mapping are relatively low cost.  Bump mapping makes approximately three times the memory reads that texture mapping does, since it has to do three texel reads whenever there is a hit.  Procedural textures have no memory cost but still use hashing as well as layering (for FBM).  One might expect a simple texture to be more cost-efficient than FBM Perlin, and much less costly than a significantly heavier FBM-Perlin attenuated Voronoi material.  Overall, it's likely dependent on the number of different materials and how heavy/detailed the procedural materials are.  The performance results appear to support this.
