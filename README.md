# Fractal Path Tracer

![alt text](img/mandelbox_orange_gui.png)
![alt text](img/mandelbox_blue_metallic.png)
![alt text](img/mandelbulb_purple_metallic.png)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

- Zachary Leong
    - [LinkedIn](https://linkedin.com/in/zleong), [personal website](https://zacharyleong.com)
- Tested on: Windows 11, Ultra 9 185H @ 2.30GHz 32GB, RTX 4060 Laptop (personal)

## features

- [Disney BSDF](#disney-bsdf) (Diffuse, Metal, Clearcoat, Glass, Sheen)
    - GGX
    - anistropy
    - emission
- [environment lighting](#environment-lighting)
- [anti-aliasing & depth of field](#the-camera) (thin lens approx.)
- [fractal distance estimator ray marching](#fractals)
- [imgui + scene loading](#scenes) / scene saving
- [agx tonemapping](#color-management)

### optimizations

- [mis + nee](#next-event-estimation-and-multiple-importance-sampling) (and separate shadow ray kernel)
- [russian roulette](#russian-roulette)
- [material sorting + path compaction](#material-sorting-and-compaction)
- [intrinsics](#intrinsics) (hot paths during raymarching)
- [CUDA streams](#cuda-streams) (reducing synchronization with Thrust)

## disney bsdf

![alt text](img/disney_showcase.png)
![alt text](img/disney_demo.png)


Material labels: subsurface, metallic, specular, specular tint, roughness, ansitropic, sheen, sheen tint, clearcoat, clearcoat roughness, transmission, transmission roughness

Value labels: 0.0, 0.2, 0.4, 0.8, 1.0

I implemented the Disney BSDF by following HW1 from this [USCD course](https://cseweb.ucsd.edu/~tzli/cse272/wi2026/). I used `std::variants` when initially testing each BRDF initially, and then combined them into an Uber shader.

I will write more on this in the future.

## next event estimation and multiple importance sampling

![alt text](img/cornell_pic.png)

A technique introduced by Eric Veach which significantly improves convergence rates by essentially sampling additional paths that contribute light per iteration. See the [performance analysis](#mis--nee) section for results.

### russian roulette

Unbaised termiantion of paths based on their contribution, while boosting the contribution of surviving paths.

### environment lighting

Used cuda texture samplers to sample high dynamic range environment maps (uniformly). 

## the camera

anti aliasing: performing sub-pixel samples to soften edges (otherwise we'd be sampling the same position at each iteration, leading to visible pixel aliasing).

### depth of field

Used thin lens approximation, where samples are taken from a circular disk.

## fractals
- mandelbulb
- mandelbox

Fractls are first intersected with a bounding sdf sphere before evaluating.

![alt text](img/mandelbox_light_glass.png)
*mandelbox*

### distance estimators

Fractal formulas do not give exact signed distances! There exists heuristics that can be applied to estimate a conservative distance to fractals (hence, distance estimators), which can be used for raymarching.

### raymarching

Performed classic sphere marching until we reach a certain threshold or max iteration count.

## scenes

Added a fly camera (UE5 style), ImGui for render, controls, camera, sdfs, material settings, and save/load functionality for scenes. Scenes are automatically saved on exit so you don't lose your settings!

![alt text](img/mandelbox_metallic.png)

### color management

I used Agx tonemapping, wiht gamma correction.

## performance analysis

Material sorting is off by default, since it was slower in my [tests](#material-sorting). Time is total kernel time from Nsight Systems per iteration.

### MIS + NEE

![alt text](profiling/graph_mis.png)

|      |                        off                        |                        on                        |
| :--: | :-----------------------------------------------: | :----------------------------------------------: |
| open | ![alt text](profiling/cornell_mis_off/render.png) | ![alt text](profiling/cornell_mis_on/render.png) |

The graphs show that MIS costs about 1 ms more per iteration, but the image converges significantly faster (the right image has visibly less noise).

### compaction

![alt text](profiling/graph_compaction.png)

|        |                             off                              |                             on                              |
| :----: | :----------------------------------------------------------: | :---------------------------------------------------------: |
|  open  |    ![alt text](profiling/cornell_compact_off/render.png)     |    ![alt text](profiling/cornell_compact_on/render.png)     |
| closed | ![alt text](profiling/cornell_closed_compact_off/render.png) | ![alt text](profiling/cornell_closed_compact_on/render.png) |

The open box with compaction is about 30% faster but the closed box is about 5% slower. This is likely because rays in the open box escape and terminate when they hit the HDRI, while rays in the closed box only termiante via russian roulette, so the overhead of performing the compaction (non-stable) outweighs the benefit. 


### material sorting
![alt text](profiling/graph_sort.png)

|                            off                             |                            on                             |
| :--------------------------------------------------------: | :-------------------------------------------------------: |
|     ![alt text](profiling/cornell_sort_off/render.png)     |     ![alt text](profiling/cornell_sort_on/render.png)     |
| ![alt text](profiling/disney_showcase_sort_off/render.png) | ![alt text](profiling/disney_showcase_sort_on/render.png) |

Material sorting is significantly slower for each scene (including the disney showcase, which has 12 materials), likely because the overhead of performing radix sort (on `uint8_t` mat ids) and scattering the pathsegments and intersection data (zipped thrust tuple) outweights the potential benefit from reducing warp divergence, especially since our BSDFs are still relatively cheap to compute.

### cuda streams

I used cuda streams for all the kernels I launch (include thrust) so that sorting and compaction can happen asynchronously, but ordered within the stream.

### intrinsics

I didn't do a formal measurement of this: using cuda intrinsics in the distance estimator hot paths (for trig functions and exponents) helped significantly increase the framerate (about 10x, empirically). The CUDA Best Practices article recommended to not enable project wide intrinsics, but to only uses them in specific places.

### other notes
We update the `num_paths` variable each iteration after compacting to reduce the number of blocks we launch, however this performs a synchronization between the device and the host. I plan to profile and determine if this is a bottleneck in the future.

## more renders

|                                            |                                               |
| :----------------------------------------: | :-------------------------------------------: |
|    ![alt text](img/mandelbox_setup.png)    | ![alt text](img/mandelbox_blue_and_black.png) |
|    ![alt text](img/mandelbox_scifi.png)    |     ![alt text](img/green_mandelbox.png)      |
| ![alt text](img/mandelbulb_unfocused.png)  |    ![alt text](img/cinematic_spheres.png)     |
|    ![alt text](img/gold_mandelbulb.png)    |    ![alt text](img/purple_mandelbulb.png)     |
| ![alt text](img/iridescent_mandelbulb.png) | ![alt text](img/metallic_blue_mandelbulb.png) |
| ![alt text](img/mandelbulb_reflection.png) |     ![alt text](img/glass_mandelbulb.png)     |

See `/saves` and `/renders` for more cool renders.

## timelapse

![alt text](gifs/renders_320_github.gif)

## bloopers

|                                             |                                            |
| :-----------------------------------------: | :----------------------------------------: |
|     ![alt text](img/fake_fish_eye.png)      | ![alt text](img/refraction_distortion.png) |
|    ![alt text](img/black_and_white.png)     |   ![alt text](img/black_and_white2.png)    |
| ![alt text](img/mandelbulb_lobotomized.png) |  ![alt text](img/too_few_march_steps.png)  |



## build instructions

requirements: windows, nvidia gpu, CUDA Toolkit 13.0, Visual Studio 2022 (MSVC), CMake and Ninja.

Run these from a VS 2022 cmd from the repo root dir:
```
cmake -S . -B build/ninja -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/ninja
build\ninja\bin\cis565_path_tracer.exe scenes\cornell.json
```

### feature toggles
Found in `src/config.h`
- SORT_PATHS
- COMPACT_TERMINATED
- LI_MIS
- RUSSIAN_ROULETTE


## controls

- Esc to exit
- S to save an image in `/saves`
- H to recenter camera at original scene point
- Ctrl + S to writes the scene settings to `/scene-saves`
- Scroll to move zoom forward
- Right mouse:
    - WASD for movement along local axes
    - Space and Shift for up and down along the y-axis respectively
    - Scroll to adjust camera acceleration


## future features

- [ ] render more fractals!
- [ ] .exr output for hdr images
- [ ] per intersection shading (procedural materials)
- [ ] further optimizations
    - [ ] CUB
    - [ ] env map mis
    - [ ] relaxed sphere tracing
    - [ ] low-discrepancy sampling (sobol + blue noise)
- [ ] light trees
- [ ] homogenous volumetric rendering
- [ ] photon mapping

## references

- my CIS 4610 path tracer
- pbrt v4
- agx tonemapping: https://github.com/bWFuanVzYWth/AgX/blob/main/agx.glsl
- fractal formulas: https://jbaker.graphics/writings/DEC.html
- mandelbox: https://www.shadertoy.com/view/7tfSzB
- disney bsdf math: https://cseweb.ucsd.edu/~tzli/cse272/wi2026/
- artist friendly metallic fresnel: https://jcgt.org/published/0003/04/03/
