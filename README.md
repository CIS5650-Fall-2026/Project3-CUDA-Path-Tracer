CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Ryan Morris
  * [LinkedIn](https://www.linkedin.com/in/r19)
* Tested on: Windows 11, Intel i7-12700H @ 2.3GHz 64GB, GeForce RTX 3070 Ti Laptop GPU 8GB

<p align="center">
<img src="img/final.png" width="500" hspace="20">
<img src="img/final_blur.png" width="500">
</p>


## Discussion of Features

### Shading Kernels and Initial Optimizations
- **Shading**: This project's shading is implemented in a new shading kernel `shadeRealMaterial` located in `pathtrace.cu`. This shader, for a given intersection, retrieves the material, and updates the color based on the material's emittance as a termination condition (e.g., reaching a light source), or calculates the ray's next bounce with `scatterRay`
- `scatterRay`, defined in `interactions.cu`, dictates the logic by material
	- For diffuse material, the next direction is sampled from the cosine-weighted direction with the provided `calculateRandomDirectionInHemisphere`
	- For reflective material, the probability of the direction being determined by diffusion (above) and a direct `glm::reflect` (mirror effect, angle in matching angle out) is based on the weighting of the specular vs. diffuse color defined in the scene file for that particular type of object
	- For refractive material, see in-depth discussion in a later section

- **Optimization: Sorting paths by material** (`SORT_BY_MATERIAL`): When this optimization flag is set to 1 in the preprocessor instruction, before shading, `thrust::sort_by_key` sorts the intersections and path segments device arrays by `materialId`. This way, threads in the same warp shade the same material where possible.
	- *Analysis:* Test ran on initial settings, 800x800px `scenes/cornell.json` and `scenes/pedestals.json` (the cover image). The former has no gltf mesh renderings, and the latter contains a couple of meshes in addition to basic cube and sphere shapes. The results indicate that this optimization leads to worse performance for these scenes, as the benefit of separating the relatively inexpensive materials (sphere and cube) with cheap evaluations from the meshes introduces more sorting overhead than efficient use of warps. One thing that is clear though is that the performance penalty is less for the pedestal scene, meaning that the inclusion of the triangle geometry mesh is starting to point to the usefulness of the sorting in implementation effeciency

	<img src="img/sort_toggle.png" width ="500">

- **Optimization: Stream compaction** Instead of having one single kernel gather all colors to write-back to the image at the end, terminated paths (misses, light hits, or no bounces left) are added to the image and removed with `thrust::remove_if`, reducing the number of warps that need to be launched for later iterations, as not all of the initial rays will make it to the end
	- *Analysis*: On the first iteration, here are the number of remaining paths at each bounce (note sorting is off). This was ran on `scenes/cornell.json` . The number of paths quickly decreases on the first few bounces in the initial open cornell box. A modification of the open Cornell box, called `scenes/cornell_closed.json` adds a front wall to the box and places the camera inside the box. Now, because the termination condition is much narrower, most of the rays stay active as they will only stop if they hit the much smaller light rectangle on the box's ceiling

<table>
<tr>
<td>

| Round | Open Box | Closed Box |
|------:|---------:|-----------:|
| 0 | 640,000 | 640,000 |
| 1 | 522,721 | 605,095 |
| 2 | 362,633 | 593,057 |
| 3 | 284,423 | 583,561 |
| 4 | 231,639 | 573,720 |
| 5 | 191,487 | 564,415 |
| 6 | 158,923 | 555,449 |
| 7 | 132,012 | 546,602 |

</td>
<td>
<img src="img/chart2.png" width="500">
</td>
</tr>
</table>



- **Stochastic sampled antialiasing**: In `generateRayFromCamera` the initial camera rays intersection points are jittered between -0.5 and 0.5 pixels which ensures that pixels landing on the boundaries are blurred between the two materials, creating an antialiasing affect. Below are pictures rendered from the cover scene `scenes/pedestals.json` before and after the anti-aliasing code modification. Most noticable is the difference in the cube geometries, as the lines jag over at a certain point. Additionally, on the more complex glTF renders like the Stanford dragon, you can see the edges look way more jagged

<table align="center">
  <tr>
    <td align="center"><img src="img/antialiasing_off.png" width="420"><br><em>Antialiasing off</em></td>
    <td align="center"><img src="img/antialiasing_on.png" width="420"><br><em>Antialiasing on</em></td>
  </tr>
</table>
<p align="center">
<img src="img/aa_pedestal.png" width="500" hspace="20">
</p>
<p align="center">

<img src="img/aa_dragon.png" width="500">
</p>



### Refraction

- To handle the rendering of refractive surfaces like glass or water, I extended the `scatterRay` helper for the `shadeRealMaterial` kernel. For this feature, sampling must randomly choose between reflection and transmission. [PBR 9.5](https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF) illustrates the Dielectric BDSF, which is a combination of the BRDF for specular reflection and BTDF for specular transmission according to the Fresnel formula for dielectrics. The Fresnel formula was referenced in [PBR 9.3.5](https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission#TheFresnelEquations). This equation is implemented in `FrDielectric` helper in `interactions.cu`. It takes the angle of the intersection to the surface normal `cosThetaI` and `eta` which is equivalent to the `IOR`, assuming the `IOR` of the outside "air" is 1.0. The result of this equation gives the percentage of rays`R` which are reflected using `glm::reflect` across the surface normal, where `1-R` are transmitted. The rays allowed to pass through are handled using `glm::refract (wi, n, eta)` which implements Snell's law, where `wi` is the incoming ray, `n` is the surface normal, and `eta` is IOR_from / IOR_to, meaning it is 1/IOR of the object that we pass into the scene. 
- GPU vs CPU: While this was straightforward to add to our `scatterRay` helper of the shading kernel to be implemented in parallel on the GPU, if we were to attempt a CPU-based implementation, we would have to iterate each pixels ray consecutively. However, compared to the base implementation, the branched code for either reflecting or transmitting is friendlier on a CPU because the CPU is able to execute one branch of the code only. In our case, we are now forcing the GPU to complete the Fresnel equation calculation on all pixels, even those without a refractive surface intersection. You can see this in the performance chart below, where enabling the feature in code changed performance. 
- Performance: There was a 2.3% reduction in framerate when switching the simple ball in `scenes\cornell_one.json` from Diffuse to Refractive. As mentioned above, this is likely due to warp divergence and the expense of calculating the Fresnel equation.

| Sphere material | FPS  | Change |
|-----------------|-----:|-------:|
| Diffuse         | 52.2 |      - |
| Refractive      | 51.0 |  -2.3% |

<p align="center">
<img src="img/diffuse.png" width="350" hspace="20">
<img src="img/refractive.png" width="350">
</p>


### Depth of Field
- In order to simulate a camera with a lens and the focal distance being sharp with the rest blurry, I modified the scene `.json` file template to contain a focal distance and lens size field in the Camera object. Lens size of 0 functions the same as it currently functions, with everything being equally sharp. `generateRayFromCamera` was edited, using [PBR 5.2.3](https://pbr-book.org/4ed/Cameras_and_Film/Projective_Camera_Models#TheThinLensModelandDepthofField) as a reference. The camera ray's initial origin and direction is set via a random `theta` and `r` within the lens disk. The rays direction is set based on the direction between this random point and the focal point, which is calculated as the pinhole ray from the center of the camera to the focal plane, which is the plane `focalDistance` in front of the camera.
- **Performance**: There was a measurable drop in having this feature on vs. having it off (lens radius of 0). FPS dropped about 1 frame. This is due to the extra computation on the GPU at the start of each iteration that can be avoided if we just shoot the ray directly (calculating the focal distance to the plane, etc.). This cost should NOT scale with the lens radius, so it makes sense that the performance with lens radius increasing does not drop off as much as the initial drop in performance from adding the feature. However, increasing the lens radius to the approximate size of the scene (10.0) results in another noticable drop in performance. A potential hypothesis for this is that because the rays at any given camera pixel can be shot from nearly anywhere in the scene, warp divergence is increased because there will be rays on the same warp hitting many different objects. Because of this, I ran a final test with sorting by object type `SORT_BY_MATERIAL` back on with lens radius 10.0. The frame rate was still significantly lower at 25.1 FPS.
- **CPU vs GPU cost difference**: Increasing the lens radius does not have any impact on CPU implementation, as rays are traced one at a time. As described above, it may introduce some warp divergence in the GPU implementation which could require other mitigations. For an implementation on either CPU or GPU, the cost of this change is quite little as it only impacts the initial rays at the beginning of each iteration and adds only a few instructions which are less computationally expensive than the shader and intersection finder. 

<table>
<tr>
<td>

| Lens radius | FPS (more better) |
|------------:|------------------:|
| 0.0         | 41.4 |
| 0.4         | 40.5 |
| 1.2         | 40.5 |
| 2.0         | 40.1 |
| 10.0        | 38.4 |

</td>
<td>
<img src="img/dof_chart.png" width="500">
</td>
</tr>
</table>

<table align="center">
  <tr>
    <td align="center"><img src="img/no_blur.png" width="350"><br><em>Lens radius 0.0</em></td>
    <td align="center"><img src="img/final_blur.png" width="350"><br><em>Lens radius 0.4</em></td>
  </tr>
  <tr>
    <td align="center"><img src="img/1.2_blur.png" width="350"><br><em>Lens radius 1.2</em></td>
    <td align="center"><img src="img/2.0_blur.png" width="350"><br><em>Lens radius 2.0</em></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><img src="img/10.0_blur.png" width="350"><br><em>Lens radius 10.0</em></td>
  </tr>
</table>


### glTF Mesh
- In the next phase of this project, I extended `loadFromJSON` in `scene.cpp` to support loading glTF meshes. This only loads the geometry, the material is still set with the color and material type as specified in the JSON file. The third-party `tiny_gltf_v3.c` library is included in this repo to support the parsing of gltf mesh. On my code's side, I call `loadGLTF` and pass in reference to the `triangles` array which is built on the CPU and later memcpy'd to the GPU. The function appends triangles to this array, starting at index `triangles.size()` and adding additional triangles, keeping the starting index of the first triangle in the mesh and the triangle count as objects built into that geometry's metadata on the CPU side. Only in `pathtraceInit` does the memory get `cudaMalloc`'d and `cudaMemcpy`'d into the `dev_triangles` device array.
- Now that the geometry is represented on the device as an array of triangles, I implemented a new test in `intersections.cu`, `meshIntersectionTest()` and its helpers `triangleIntersect` and `enterBox`. `triangleIntersect` is an implementation of `glm::intersectRayTriangle` which takes the ray origin and the triangle's three vertices and returns a bool of whether the ray hits this particular triangle, along with setting barycentric coordinates and the ray length `t` as input reference parameters. 
- As an initial performance improvement, a bounding box test was added and stored in each geoemtry, if the ray doesn't fall within the bounding box, -1 is returned without having to iterate through all triangles. This is toggled with the preprocessor directive `MESH_BBOX_CULLING`. Discussion of its performance will be included in the next section, along with the BVH. 


### BVH Data Structure

- As adding the `glTF` support slowed down the path tracer significantly, I decided to add a bounding volume hierarchy (BVH) data structure which is loaded first on the CPU and then copied to device memory alongside the `dev_triangles` array discussed before. The BVH is a binary tree, where each node stores contiguous triangles belonging to it, its bounding box with two glm::vec3s, and then the index in the node array of the two children nodes. The nodes themselves when built on the CPU are stored in a flat array, because it is built in a depth-first manner, the left child is always the following index, but the right child may have a much further index in the array. Thus, it would not be strictly necessary to store the left child.

```cpp
struct BVHNode {
    int t_start_idx = 0; // if leaf, first triangle
    int t_count= 0; // if leaf, num triangles
    glm::vec3 bottom_corner = glm::vec3(FLT_MAX); // bounding box mins
    glm::vec3 top_corner = glm::vec3(-FLT_MAX); // bounding box maxes
    int left_child = -1; // left child index
    int right_child = -1; // right child index
};
```
- `buildBVH()` in `scene.cpp` contains the code for actually building the BVH. It tracks the minimum and maximum centroid of all triangles in that specific geometry. The axis with the biggest difference is used for the binary split, and the triangles are sorted by centroids in the selected dimension. The midpoint value on that dimension is used as the split point, and the triangles are partitioned by that value. If all triangles would fall on one side (e.g. they all have the same value) then the triangles are split in half. Any time there are less than 4 triangles left, that is considered a leaf and a termination condition. More testing needs to be done on the ideal size of a leaf and/or maximum depth of the tree.
- Performance: While the bounding box culling discussed above does improve performance substantially with a single glTF geometry, it is not sufficient for scenes such as `img/pedestals.json` which contain many glTF geometries. As you can see in the chart below, both had sub 0.5 FPS performance on my machine. However, the 4 sized leaf BVH has 41.5 FPS (similar to other runs in experiments above).
- Building the BVH is much easier to do iteratively, and since it only needs to be done once when the scene is loaded, the CPU rendering is sufficient for our purposes. The data structure would also help significantly for a CPU implementation of a path tracer, as CPU would still require checking all triangles for every ray in absence of this or some similar spatial data structure to restrict the search. 

<table>
<tr>
<td>

| Method               | FPS  | ms per frame |
|----------------------|-----:|-------------:|
| No optimizations     | 0.1  | 9141 |
| Bounding box culling | 0.3  | 3801 |
| BVH                  | 41.5 | 24.4 |

</td>
<td>
<img src="img/bvh_chart.png" width="500">
</td>
</tr>
</table>

Below, we can see the results from Nsight Systems profiling, showing that prior to introduction of BVH,the computeIntersections kernel took up virtually all of the time of the program, and you can see it being reduced to 91.5% of the GPU time. Note that this is a stacked 100% graph, as you wouldn't be able to even see the RHS bar on the chart otherwise.

<table>
<tr>
<td>

| Kernel                | BVH off | BVH on |
|-----------------------|--------:|-------:|
| computeIntersections  | 99.95%  | 91.5%  |
| shadeRealMaterial     | 0.02%   | 4.6%   |
| Stream compaction     | 0.01%   | 1.6%   |
| generateRayFromCamera | 0.01%   | 1.0%   |
| newGather             | 0.01%   | 1.0%   |
| sendImageToPBO        | 0.00%   | 0.2%   |
| **ms per iteration**  | **4212** | **21.2** |

</td>
<td>
<img src="img/nsight_kernel.png" width="500">
</td>
</tr>
</table>


And as a funny comment, this is the result upon launching without BVH turned on:


<p align="center">

<img src="img/without_bvh.png" width="500">
</p>


## References
- [tinygltf](https://github.com/syoyo/tinygltf) (v3 C API: `tiny_gltf_v3.h`,
  `tiny_gltf_v3.c`, `tinygltf_json_c.h`) for parsing glTF/GLB files.
  Copyright (c) 2017 Syoyo Fujita, Aurélien Chatelain and others
- [Duck](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Duck)
  glTF sample model from the Khronos glTF-Sample-Assets repository.© 2006, Sony.
- [Stanford Dragon](https://graphics.stanford.edu/data/3Dscanrep/)
  model from the Stanford 3D Scanning Repository.
  © Stanford University Computer Graphics Laboratory. Non-commercial research use.
- [Utah Teapot](https://en.wikipedia.org/wiki/Utah_teapot)
  model created by Martin Newell at the University of Utah.
  © 1975, Martin Newell / University of Utah.
- [Cloud Low Poly - Large](https://sketchfab.com/3d-models/cloud-low-poly-large-19f7ecd70a4f4c5f9b1a589d2cc308c1)
  model by [PolyDavid](https://sketchfab.com/PolyDavid), from Sketchfab.
  Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

