**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Alice Liu
  * [LinkedIn](https://www.linkedin.com/in/aliceliuu/), [personal website](https://aliceliu.xyz/)
* Tested on: Windows 11, AMD Ryzen 5 7640HS @ 4.30GHz 16GB, RTX 4060 Laptop GPU 8GB (Personal Laptop)

## CUDA Path Tracer
<p align="center">
  <img src="img/custom_1080px_7000i_8d_dof.png" width="600"><br>
  1080 x 1080 pixels, 7000 iterations, 8 depth, 31.5k triangles
</p>

## Overview


## Refraction


## Physically-Based Depth-of-Field
| No depth of field | With depth of field |
|---|---|
| ![](img/custom_1080px_7000i_8d_dof.png) | ![](img/custom_1080px_7000i_8d_no_dof.png) |


## GLTF Mesh Loading


## Bounding Volume Hierarchies


## Performance Analysis
### Stream Compaction Analysis
Stream compaction helps most after a few bounces. Print and plot the effects of stream compaction within a single iteration (i.e. the number of unterminated rays after each bounce) and evaluate the benefits you get from stream compaction.

Compare scenes which are open (like the given cornell box) and closed (i.e. no light can escape the scene). Again, compare the performance effects of stream compaction! Remember, stream compaction only affects rays which terminate, so what might you expect?

### Material Sorting Analysis

### Bounding Volume Hierarchies Analysis



## Bloopers
<p align="center">
  <img src="img/bug_rng_no_depth.png" width="550"><br>
  Blooper from the very beginning where I forgot to mark iterations as complete after <br> reaching the max trace depth. Resulted in these lines/sections in the render. 
</p>


<p align="center">
  <img src="img/0.0001_refraction_blooper.png" width="550"><br>
  Refraction implementation blooper where my original ray origin epsilon (0.0001) offset was <br> not large enough,  causing the white ring. Increasing epsilon to 0.001 fixed this!
</p>

## Attribution
### Libraries: 
- [tinygltf](https://github.com/syoyo/tinygltf)

### References: 
- [Diffuse Materials](https://pbr-book.org/4ed/Reflection_Models/Diffuse_Reflection)
- [Generating Specular Rays](https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling)
- [Specular and Refractive Materials](https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission)
- [Schlick's Approximation](https://en.wikipedia.org/wiki/Schlick%27s_approximation)
- [Stochastic Anti-Aliasing](https://paulbourke.net/miscellaneous/raytracing/)
- [Thin Lens Camera/Depth of Field](https://pbr-book.org/4ed/Cameras_and_Film/Projective_Camera_Models#TheThinLensModelandDepthofField)
- [Bounding Volume Hierarchies](https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies#)

### Models:
- [Duck](https://github.khronos.org/glTF-Assets/model/Duck)