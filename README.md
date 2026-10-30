OptiX/CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Luke (Hyuk Che) Kwon
  * [LinkedIn](https://www.linkedin.com/in/hyukchekwon/), [Personal Website](https://lukekwon98.github.io/)
* Tested on: Windows 11, AMD Ryzen 5 5600X 6-Core Processor @ ~3.7GHz 16GB, Nvidia GeForce RTX 3060 (Compute Capability 8.6)

## Overview

This project is a GPU path tracer built with CUDA and NVIDIA OptiX, supporting glTF meshes, physically based materials, texture and normal mapping, HDR environment lighting with multiple importance sampling, and depth of field.

The project explores how rendering architecture affects performance, starting with separate CUDA kernels for camera generation, intersection, and shading, then introducing OptiX intersections, moving the bounce loop into raygen, and finally generating camera rays within the same program. The earlier execution paths remain available for comparison.

<img src="https://github.com/user-attachments/assets/6b081d53-71d7-45c5-96e4-296da682ee11" alt="5" width="900" />

### Render Gallery

| Render 1 | Render 2 |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/9b3ce840-2d83-4ab5-948b-c7c038f368a6" alt="1" width="360" /> | <img src="https://github.com/user-attachments/assets/7f69a0ba-f6ef-45e6-8ffa-d7ec2b07e257" alt="NotFinal" width="360" /> |

| Render 3 | Render 4 |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/9e88f119-b532-464f-91b4-1011f2da8805" alt="3" width="360" /> | <img src="https://github.com/user-attachments/assets/ad3da490-d793-4595-acb9-d72624c7ecaf" alt="4" width="360" /> |

| Render 5 | Render 6 |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/6b081d53-71d7-45c5-96e4-296da682ee11" alt="5" width="360" /> | <img src="https://github.com/user-attachments/assets/52349970-5202-43f4-8761-e4c857d4fcfb" alt="6" width="360" /> |

| Render 7 | Render 8 |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/66a21fa1-66bb-477f-b683-75c02681bcb7" alt="2" width="360" /> | <img src="https://github.com/user-attachments/assets/4c291d1b-6c14-4230-a87c-b7dd20ddd155" alt="cornell 50000samp" width="360" /> |

## OptiX Integration

The renderer uses OptiX to build a geometry acceleration structure (GAS) over the scene’s triangles and trace rays against it. The integration consists of the following components:

| Component | Role in this renderer |
|---|---|
| Device context | Associates OptiX with the CUDA context used by the renderer. |
| PTX module | Loads the GPU programs compiled from `optixPrograms.cu`. |
| Program groups | Select the raygen, closest-hit, and miss entry points. |
| Pipeline | Links those program groups and configures their stack requirements. |
| Shader binding table (SBT) | Identifies the programs used for ray generation, hits, and misses. |
| Geometry acceleration structure (GAS) | Accelerates intersection queries over the uploaded triangle mesh. |
| Launch parameters | Provide GPU programs with the scene buffers, camera, materials, textures, and rendering settings. |

### Setup Sequence

OptiX setup happens once during initialization. The resulting pipeline, shader binding table, and scene data are then reused for rendering.

1. **Create the device context.** Initialize CUDA and OptiX, then create an OptiX device context associated with the renderer’s CUDA context.

2. **Load the PTX and create a module.** CMake compiles `optixPrograms.cu` into PTX during the build. At runtime, OptiX loads that PTX into a module containing the raygen, closest-hit, and miss programs.

3. **Create program groups.** Each program group references the module and selects the entry point for its role: ray generation, closest hit, or miss.

4. **Create the pipeline.** Link the program groups into a pipeline and configure the stack sizes required to execute them.

5. **Build the shader binding table (SBT).** Pack headers from the program groups into SBT records and upload them to GPU memory. The SBT tells a launch which raygen, miss, and hit-group records to use.

6. **Upload geometry and build the GAS.** Upload the triangle mesh and build its acceleration structure. OptiX returns a traversable handle that identifies the GAS for subsequent ray queries. This geometry setup is independent of module and pipeline creation.

7. **Populate launch parameters.** Store the GAS handle, GPU buffer pointers, camera data, and rendering settings in the launch-parameter structure, then upload it to the GPU.

8. **Launch rendering.** Call `optixLaunch` with the pipeline, SBT, launch parameters, CUDA stream, and launch dimensions. The raygen program calls `optixTrace` with the GAS handle. Traversal then invokes the appropriate closest-hit or miss program.

```mermaid
flowchart TD
    A["CUDA and OptiX device context"] --> B["PTX module"]
    B --> C["Program groups"]
    C --> D["Pipeline"]
    C --> E["SBT records"]
    A --> F["Upload triangles and build GAS"]
    F --> G["Launch parameters: GAS handle and scene data"]
    D --> H["optixLaunch"]
    E --> H
    G --> H
    H --> I["Raygen: optixTrace"]
    I --> J["GAS traversal and intersection"]
    J --> K["Closest-hit or miss program"]
```

The pipeline supplies the executable programs, the SBT selects their records, and the launch parameters supply the data those programs operate on.

### Geometry and intersection data

The glTF loader applies scene-node transforms to mesh data before uploading it. Meshes are combined into a single triangle GAS, with per-triangle material IDs linking geometry to shading data.

The closest-hit program uses triangle barycentric coordinates to interpolate vertex normals, texture coordinates, and tangents. It also computes the geometric normal, which is retained separately for refraction and ray-origin offsets. The miss program records that no surface was found.

### Two OptiX execution paths

The first path uses OptiX only for intersection queries. The CPU controls the bounce loop, and a separate CUDA kernel shades each batch of intersections.

The second path moves the bounce loop and shading into `__raygen__pathtrace`. Each invocation maintains a path locally, repeatedly traces and shades it until termination, and writes its completed contribution for the CUDA `finalGather` kernel. Camera-ray generation can also run inside this raygen program, removing the separate camera-generation kernel.

Both paths share the same closest-hit and miss programs. Image accumulation and conversion to the OpenGL pixel buffer remain CUDA operations.

## Rendering Architecture and Performance

### CUDA Baseline

The CUDA baseline uses a CPU-controlled bounce loop with separate GPU kernels for camera-ray generation, intersection testing, and shading. Each iteration generates one jittered camera ray per pixel and accumulates its contribution into the image.

The intersection kernel tests each active ray against every sphere and box in the JSON scene. The shading kernel updates the path’s throughput and direction, or terminates it when it misses, reaches an emitter, or exhausts its bounce budget.

```mermaid
flowchart TD
    A["Generate camera rays — CUDA"] --> B{"Bounce limit reached?"}
    B -->|No| C["Intersect scene — CUDA"]
    C --> D["Shade and update paths — CUDA"]
    D --> B
    B -->|Yes| E["Accumulate contributions — CUDA"]
    E --> F["Tone map and write OpenGL PBO — CUDA"]
```

Without stream compaction, each bounce launches kernels over the full path buffer. Terminated paths return early, but their slots remain part of subsequent launches.

### Stream Compaction

Stream compaction groups surviving paths at the front of the path buffer after shading. The next bounce can then launch work over only the surviving paths.

```mermaid
flowchart TD
    A["Generate camera rays — CUDA"] --> B{"Active paths remain?"}
    B -->|Yes| C["Intersect active paths — CUDA"]
    C --> D["Shade and update paths — CUDA"]
    D --> E["Partition active and terminated paths"]
    E --> F["Read active count to CPU"]
    F --> B
    B -->|No| G["Accumulate contributions — CUDA"]
    G --> H["Tone map and write OpenGL PBO — CUDA"]
```

Partitioning preserves terminated paths and their completed contributions. Each path retains its original pixel index, allowing `finalGather` to accumulate the result into the correct pixel after reordering.

The implementation supports four configurations:

| `USE_PARTITION` | Configuration | Subsequent launches |
|---|---|---|
| `0` | No partitioning | Full path count. Terminated paths return early. |
| `1` | Thrust partition | Active path count returned by `thrust::partition`. |
| `2` | CUB partition with reusable temporary storage | Active path count copied back to the CPU. |

The flowchart above describes modes `1` and `2`. Mode `3` retains a fixed number of bounce iterations and full-size launches, avoiding the active-count readback while still grouping active paths.

Compaction introduces partitioning overhead, and modes `1` and `2` also require the CPU to obtain the active count before scheduling the next bounce. Its benefit therefore depends on how quickly paths terminate and how much intersection and shading work remains.

### Performance Evaluation

The comparison uses identical camera settings, materials, resolution, and maximum bounce depth within each test scene.

#### CUDA Profiling: Brute Force, BVH, and Compaction

| Configuration | Capture |
|---|---|
| CUDA brute force | <img src="https://github.com/user-attachments/assets/2bf9c658-5221-44aa-bf48-49ec5645594d" alt="CUDA brute force" width="900" /> |
| CUDA brute force + CUB | <img src="https://github.com/user-attachments/assets/09412aa1-b10d-4e89-9701-df34e60cf843" alt="CUDA brute force + CUB" width="900" /> |
| CUDA BVH | <img src="https://github.com/user-attachments/assets/1c726837-9c6b-4b50-8b94-2cfd9e211447" alt="CUDA BVH" width="900" /> |
| CUDA BVH + CUB | <img src="https://github.com/user-attachments/assets/9f1b7dc5-f3ac-40ad-98d4-d719ed430ec2" alt="CUDA BVH + CUB" width="900" /> |

The brute-force capture is dominated by long `computeTriangleIntersections` kernels. Its selected-frame average is **65.87 ms (15.18 FPS)**. Adding CUB reduces this to **51.36 ms (19.47 FPS)**, about **22% less frame time**. Intersection work still dominates, but reducing the active workload is valuable while every surviving ray must test the mesh's triangles.

Introducing a CUDA BVH changes the scale of the workload. The no-compaction capture averages **4.35 ms (229.72 FPS)**, approximately **15.1× faster** than the brute-force capture. The timeline now exposes the repeated intersection/shading sequence instead of being dominated by one very long intersection kernel.

Adding CUB to the BVH version averages **5.48 ms (182.33 FPS)** in the selected frames, about **26% more frame time** than BVH alone. The GPU timeline contains additional work, memory operations, and gaps, while the CPU API row repeatedly enters `cudaMemcpy`. Once intersection is much cheaper, partitioning and the dependencies between bounces can outweigh the work saved by removing terminated paths. The controlled Suzanne benchmark shows the same direction, although a smaller difference: **226 FPS without CUB versus 210 FPS with CUB**

### OptiX Intersection Queries

This version preserves the CPU-controlled bounce loop but replaces the CUDA intersection kernel with an OptiX launch. Each raygen invocation reads an existing path and calls `optixTrace`. The closest-hit or miss program writes an intersection result, which the separate CUDA shading kernel then consumes.

```mermaid
flowchart TD
    A["Generate camera rays — CUDA"] --> B{"Continue bouncing?"}
    B -->|Yes| C["Launch intersection queries — OptiX"]
    subgraph OPTIX["OptiX intersection launch"]
        C --> D["Trace against triangle GAS"]
        D --> E["Closest-hit or miss writes intersection"]
    end
    E --> F["Shade and update paths — CUDA"]
    F --> G["Optional path partitioning"]
    G --> B
    B -->|No| H["Accumulate contributions — CUDA"]
    H --> I["Tone map and write OpenGL PBO — CUDA"]
```

With active-count compaction enabled, the loop continues while active paths remain. Without it, the loop runs to the configured bounce limit, and terminated paths return early.

This architecture introduces accelerated triangle intersection while preserving the original separation between tracing and shading. However, every bounce still requires an OptiX launch, a CUDA shading launch, and optional partitioning. Intersection results and updated path state pass between stages through GPU buffers.

#### OptiX Profiling: No Compaction, Thrust, and CUB

| Configuration | Capture |
|---|---|
| OptiX intersections — no compaction | <img src="https://github.com/user-attachments/assets/cbc2e081-6160-4053-a626-c988f5fca8b2" alt="OptiX intersections — no compaction" width="900" /> |
| OptiX intersections + Thrust | <img src="https://github.com/user-attachments/assets/8dc1836b-47a3-4030-8904-282cb9fbd4cc" alt="OptiX intersections + Thrust" width="900" /> |
| OptiX intersections + CUB | <img src="https://github.com/user-attachments/assets/b1cfd5d3-a423-4512-9e8a-640bef1df6e1" alt="OptiX intersections + CUB" width="900" /> |

Without compaction, the capture shows repeated pairs of OptiX intersection launches and CUDA shading kernels. Later pairs are shorter, but each bounce still requires host scheduling and separate stages. Small memory events and repeated CPU `cudaMemcpy` calls are visible even in the no-compaction capture, so not every transfer in an OptiX timeline can be attributed to path compaction.

The Thrust capture shows substantial time in kernels whose names begin with thrust, followed by further Thrust work. Long `cudaStreamSynchronize` calls and a visible `cudaFree` accompany this sequence. This is evidence that the selected compaction implementation introduces substantial processing and synchronization around an otherwise short intersection/shading stage. The benchmark results establish the overall penalty for Suzanne, **320 FPS without compaction drops to 132 FPS with Thrust**. Thrust is slower in all four reported scenes.

The CUB capture shows a prominent initial partition stage followed by much smaller later-bounce tasks and memory events. This is consistent with the benefit of processing fewer surviving paths. In the controlled results, CUB raises Suzanne from **320 to 342 FPS** and FlightHelmet from **295 to 300 FPS**, but lowers Sponza from **99 to 77 FPS**. 

#### Extra: Fixed-Count CUB Compaction

The fixed-count CUB mode retains full-size launches and a fixed number of bounce iterations, avoiding the active-count readback while still grouping active paths.

| Implementation | Box | Suzanne | FlightHelmet | Sponza |
|---|---:|---:|---:|---:|
| OptiX ISect | 350 | 320 | 295 | 99 |
| OptiX ISect + CUB | 390 | 342 | 300 | 77 |
| OptiX ISect + Fixed CUB | 156 | 150 | 147 | 78 |

*FPS*

Fixed-count CUB is slower than no compaction in every tested scene. Although it avoids reading the active count back to the CPU, it still pays for partitioning and launches subsequent kernels over the full path count. Grouping surviving paths alone does not recover these costs in these tests.

Compared with active-count CUB, fixed-count CUB performs substantially worse for Box, Suzanne, and FlightHelmet. Sponza is nearly unchanged at 78 versus 77 FPS.

#### Enclosed Scene

| Open | Enclosed |
|---|---|
| <img src="https://github.com/user-attachments/assets/77171d03-9835-409a-a178-f7dc240016ff" alt="TestObject2 Suzanne3936" width="360" /> | <img src="https://github.com/user-attachments/assets/e02a7a57-9da3-40c8-9d59-c5864f8c7c77" alt="TestObject5 OccludedBackground" width="360" /> |

| Configuration | Suzanne Open | Suzanne Enclosed |
|---|:---:|:---:|
| **OptiX ISect** | <img src="https://github.com/user-attachments/assets/cbc2e081-6160-4053-a626-c988f5fca8b2" alt="Open Suzanne — no compaction" width="600" /> | <img src="https://github.com/user-attachments/assets/3df02eaa-99e3-4ad4-b61d-d1aa8b61155e" alt="Enclosed Suzanne — no compaction" width="600" /> |
| **OptiX ISect + CUB** | <img src="https://github.com/user-attachments/assets/b1cfd5d3-a423-4512-9e8a-640bef1df6e1" alt="Open Suzanne — CUB compaction" width="600" /> | <img src="https://github.com/user-attachments/assets/df2bfd90-5aa7-4871-b595-8dcac35381a1" alt="Enclosed Suzanne — CUB compaction" width="600" /> |

In the enclosed no-compaction capture, substantial intersection and shading work remains across the bounce sequence. With CUB, substantial work also persists, with partition stages and memory operations repeated between bounces. Unlike the rapidly shrinking tail in the other CUB capture, this sequence suggests that more paths continue bouncing inside the enclosure.

The selected-frame average increases from **8.37 ms (119.52 FPS)** without compaction to **12.11 ms (82.56 FPS)** with CUB: about **45% more frame time**, or **31% lower FPS**. This supports the expected limitation of compaction in a closed scene: paying to partition paths is less useful when many paths survive.

| Implementation | Suzanne Open | Suzanne Enclosed |
|---|---:|---:|
| OptiX ISect | 320 | 128 |
| OptiX ISect + CUB | 342 | 91 |

CUB improves throughput by **6.9%** in the open scene but reduces it by **28.9%** in the enclosed scene. Escaping paths in the open scene allow compaction to reduce subsequent work. In the enclosure, more paths continue bouncing, leaving less work to eliminate while partitioning and active-count readback still incur overhead. This aligns with the profiling captures above.


### Moving the Bounce Loop into OptiX Raygen

The next version moves the entire per-path bounce loop into `__raygen__pathtrace`. Each invocation reads a camera path, traces it, evaluates its material, and repeats until the path terminates.

```mermaid
flowchart TD
    A["Generate camera rays — CUDA"] --> B["Launch path tracer — OptiX"]
    subgraph OPTIX["One OptiX launch"]
        B --> C["Read camera path"]
        C --> D{"Path still active?"}
        D -->|Yes| E["Trace against triangle GAS"]
        E --> F["Closest-hit or miss writes intersection"]
        F --> G["Shade and update local path"]
        G --> D
        D -->|No| H["Write completed path contribution"]
    end
    H --> I["Accumulate contributions — CUDA"]
    I --> J["Tone map and write OpenGL PBO — CUDA"]
```

The CPU no longer schedules individual bounces. Path state is maintained locally within each invocation, and the completed contribution is written to the output buffer after termination. Intersection results still use the shared intersection buffer.

This removes the separate shading launches and per-bounce partitioning passes. It also changes how work is distributed, since invocations can execute different numbers of bounces and take different material branches. The performance comparison examines the balance between reduced launch and buffer traffic overhead and this variation in execution.

#### If we pass more code into "OptiX kernels", where is that code actually run? Can RT cores handle non-RT supported code?

According to [NVIDIA documentation](https://forums.developer.nvidia.com/t/take-full-advantage-of-cuda-core-and-rt-core/241682), application-defined OptiX programs execute on the GPU’s SMs, while RT cores accelerate acceleration-structure traversal and ray–triangle intersection. Camera-ray generation, random sampling, BSDF evaluation, and path-throughput updates therefore continue to execute on ordinary programmable SM hardware when moved into raygen. Closest-hit and miss programs also execute on SMs. CUDA kernels and OptiX programs also access the same CUDA-allocated GPU buffers through device pointers. 

The Nsight Systems captures show the resulting change in execution structure. Repeated OptiX intersection launches and CUDA shading kernels become one main OptiX interval, followed by CUDA accumulation and display conversion. When camera generation also moves into raygen, its separate CUDA kernel disappears. These traces support the reduction in separately launched stages. They do not expose the division of work between SMs and RT cores within the OptiX interval.

#### Profiling the Raygen Bounce Loop

| Configuration | Capture |
|---|---|
| OptiX bounce loop | <img src="https://github.com/user-attachments/assets/bd769251-14db-4994-9b55-1531d0656da9" alt="OptiX bounce loop" width="900" /> |

The repeated intersection/shading launch pairs are replaced by one main OptiX GPU interval. A separate `generateRayFromCamera` kernel still precedes it, and `finalGather` and display conversion follow it. The capture therefore directly shows the reduction in separately scheduled stages. It reports **1.73 ms (577.38 fps)** over the selected frames.

The controlled Suzanne benchmark increases from **320 FPS** with OptiX intersection queries alone to **680 FPS** with the bounce loop in raygen, a **2.13× speedup**. This is consistent with avoiding per-bounce host scheduling and separate shading launches.


### Generating Camera Rays Inside Raygen

The final architecture also generates camera rays inside `__raygen__pathtrace`. Each invocation derives its pixel coordinates from its launch index, generates a jittered camera ray, optionally applies thin-lens depth of field, and enters the bounce loop.

This replaces the separate camera-generation kernel and the initial read from the path buffer. The rest of the raygen-loop architecture remains unchanged. Completed contributions are written once, then accumulated and converted for display by CUDA kernels.

The `USE_RAYGEN_CAMERA` toggle preserves the separate-camera version for comparison.

#### Profiling Camera Generation Inside Raygen

| Configuration | Capture |
|---|---|
| OptiX bounce loop + camera | <img src="https://github.com/user-attachments/assets/b62510f3-566d-4cca-8ef2-f16081fde46d" alt="OptiX bounce loop + camera" width="900" /> |
| OptiX bounce loop + camera — no forced inline | <img src="https://github.com/user-attachments/assets/eb7fa38e-43b1-410c-97c3-5bc08483f515" alt="OptiX bounce loop + camera — no forced inline" width="900" /> |

With camera generation included, the separate camera kernel disappears. The visible main GPU sequence becomes **OptiX path tracing → final gather → display conversion**. The selected-frame average is **1.13 ms (884.20 FPS)**, compared with **1.73 ms** in the separate-camera capture. The controlled Suzanne benchmark similarly improves from **680 to 960 FPS**, a **41% increase**.

Gaps remain around the short GPU sequence. Once tracing is this fast, host execution, presentation, and scheduling are plausible contributors to total frame time, but these results do not isolate their individual costs.

#### Extra: Forced Inlining

| Implementation | Box | Suzanne | FlightHelmet | Sponza |
|---|---:|---:|---:|---:|
| OptiX Loop + Cam | 1070 | 960 | 723 | 172 |
| OptiX Loop + Cam — No Forced Inline | 1050 | 960 | 720 | 170 |

*FPS*

Removing forced inlining has little effect on measured throughput, FPS remains virtually unchanged for all models. Without repeated measurements and variability estimates, these small differences do not establish a consistent performance benefit.

Both profiling captures retain the same main GPU sequence: OptiX path tracing, final gather, and display conversion. Removing `__forceinline__` allows the compiler to make its own inlining decisions, as it does not guarantee that these functions remain uninlined.

### Shared Benchmark Configuration

Within each scene, all implementations use identical triangle geometry, image resolution,
emissive-light geometry and intensity, and maximum bounce depth.

Surfaces use Lambertian diffuse shading with cosine-weighted hemisphere
sampling. Textures, normal mapping, metallic–roughness shading, refraction, depth of
field, environment lighting, and explicit direct-light sampling/MIS are
disabled for these comparisons.

Camera framing may differ between benchmark scenes, but remains fixed
across implementations within each scene. The enclosed Suzanne scene uses
a fully closed diffuse box surrounding both Suzanne and the light source.

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 |
| Build configuration | Release |
| Resolution | 800 x 800 |
| Maximum bounce depth | 8 |
| Samples / measurement interval | 10s, after 5 seconds of warmup |

### Performance Results

Average FPS under the shared benchmark configuration above. Column headings list mesh triangle counts.

**CUDA BVH comparison:** I did not write the CUDA BVH and Möller-Trumbore intersection implementations. They were generated using Chat GPT under the permission of professor Schwartz, and used only as a performance baseline for comparison with my OptiX implementation. Therefore they were not included in my commits, and are not listed as supported features of this project.

<img width="600" height="371" alt="performance_chart" src="https://github.com/user-attachments/assets/a97d47a1-f7a3-4480-8cba-078b9821c94b" />

| **Implementation** | **Box: 12** | **Suzanne: 3,936** | **FlightHelmet: 94,722** | **Sponza: 262,267** |
| :---: | ---: | ---: | ---: | ---: |
| **CUDA Brute** | 420 | 15 | 0.7 | 0 |
| **CUDA Brute + CUB** | 460 | 19 | 0.8 | 0.1 |
| **CUDA BVH** | 410 | 226 | 110 | 11 |
| **CUDA BVH + CUB** | 460 | 210 | 124 | 11 |
| **OptiX ISect** | 350 | 320 | 295 | 99 |
| **OptiX ISect + Thrust** | 175 | 132 | 125 | 31 |
| **OptiX ISect + CUB** | 390 | 342 | 300 | 77 |
| **OptiX Loop** | 755 | 680 | 576 | 162 |
| **OptiX Loop + Cam** | 1070 | 960 | 723 | 172 |

| Box — 12 triangles | Suzanne — 3,936 triangles |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/fc3befeb-e141-429b-98f5-e413fbe92649" alt="TestObject1 Box12" width="360" /> | <img src="https://github.com/user-attachments/assets/77171d03-9835-409a-a178-f7dc240016ff" alt="TestObject2 Suzanne3936" width="360" /> |

| FlightHelmet — 94,722 triangles | Sponza — 262,267 triangles |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/c3f66374-f247-4b0b-8601-b30db7042e9a" alt="TestObject3 FlightHelmet94722" width="360" /> | <img src="https://github.com/user-attachments/assets/34f7b529-89e4-4e50-808f-4b02bb25e5a2" alt="TestObject4 Sponza262267" width="360" /> |

The fastest measured configuration is **OptiX Loop + Cam** in all four scenes. Relative to CUDA BVH without compaction, it is **2.61× faster for Box**, **4.25× for Suzanne**, **6.57× for FlightHelmet**, and **15.64× for Sponza**. These ratios use the controlled FPS table, rather than selected profiler frames. The profiling captures above explain the architectural progression of accelerating intersections first, then reducing the repeated scheduling and staging work around it.

## Visual Features

The full feature set is implemented in the OptiX raygen path. The host-controlled rendering paths remain available for architecture comparisons.

### Stochastic Antialiasing

Each iteration samples a random position within every pixel and generates a camera ray through that position. Accumulating these samples progressively smooths silhouette edges and other subpixel details.

Random seeds depend on the iteration and pixel index, and both the CUDA camera kernel and the OptiX camera implementation use this approach.

| Without jitter - Mesh | With jitter - Mesh |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/c4ff0c9f-5d0e-430b-84ab-a97fb8eb20b2" alt="JitterX" width="360" /> | <img src="https://github.com/user-attachments/assets/8ea4ffec-e8c1-4e3d-9395-3d98440544d2" alt="JitterO" width="360" /> |

| Without jitter - Implicit geometry | With jitter — Implicit geometry |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/05b47519-6486-4388-b0fe-94559746b913" alt="JitterX Cornell" width="360" /> | <img src="https://github.com/user-attachments/assets/b816cc70-a4e2-4987-bec5-603dba39aed3" alt="JitterO Cornell" width="360" /> |

### Reflection, Refraction, and Fresnel

The renderer supports smooth dielectric transmission and reflection. At each glass intersection, the geometric normal determines whether the ray is entering or leaving the surface, selecting the corresponding incident and transmitted indices of refraction.

The full dielectric Fresnel equations determine the probability of reflection. Otherwise, the ray refracts according to Snell’s law. Total internal reflection produces a reflected ray, and transmitted paths include the squared relative-ior factor for radiance transport.

| Reflection | Reflection 2 | Reflection 2 |
| :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/5aeab3fe-8a2f-4fd4-b424-8e7e98f33d5c" alt="Reflective" width="360" /> | <img src="https://github.com/user-attachments/assets/d560495f-a8d7-4db4-86e4-30f6ff0ced62" alt="Reflective1" width="360" /> | <img src="https://github.com/user-attachments/assets/65ed8a8e-f0e7-4b46-8354-657deec5012c" alt="Reflective2" width="360" /> |

| Refraction | Refraction 1 |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/eff32a26-d407-43f0-9589-99c5fed35f13" alt="Refractive1" width="360" /> | <img src="https://github.com/user-attachments/assets/c67598d9-b398-46ae-924d-a5ca8ce4b23e" alt="Refractive2" width="360" /> |

### Microfacet and Metallic–Roughness Materials

#### Metallic–Roughness Grids

The top left is **metallic = 1, roughness = 0**, and the bottom right is **metallic = 0, roughness = 1**.

| White | Gold | Blue |
| :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/b8770a1d-44ca-4cf3-8073-84980035a6f7" alt="Metallic Roughness" width="360" /> | <img src="https://github.com/user-attachments/assets/93ab6e98-6aac-4109-8491-e92f9f487d70" alt="Metallic Roughness Gold" width="360" /> | <img src="https://github.com/user-attachments/assets/c62bac1c-4c05-4cc9-bca6-f9104cac4fbc" alt="Metallic Roughness Blue" width="360" /> |

#### Roughness Tests

| Roughness: 0 | Roughness: 0 |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/a45110fd-82f4-4a7f-83b0-68b563afeec6" alt="Roughness0" width="360" /> | <img src="https://github.com/user-attachments/assets/4b088eed-bc23-4c6e-a3d3-9fcbe821778d" alt="Roughness0F" width="360" /> |

| Roughness: 0.2 | Roughness: 0.2 |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/d84a16ef-6f20-4d32-b7ae-543cf14ec54d" alt="RoughnessP2" width="360" /> | <img src="https://github.com/user-attachments/assets/b1ff3d82-a87a-4da0-a87c-9b1f5e6de6dc" alt="RoughnessP2F" width="360" /> |

| Roughness: 0.5 | Roughness:0.5 |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/b6c4d600-f27a-4581-a4f5-693d5f85480b" alt="RoughnessH" width="360" /> | <img src="https://github.com/user-attachments/assets/8e820d09-2ef3-4ecb-a8a8-ea9228ff5dcf" alt="RoughnessHF" width="360" /> |

| Roughness: 1 | Roughness: 1 |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/de16eb16-8c9e-4c5c-9518-f79e0bdda7f4" alt="Roughness1" width="360" /> | <img src="https://github.com/user-attachments/assets/289351dd-12e5-47a9-9888-5d6efa31b95e" alt="Roughness1F" width="360" /> |

Rough reflective surfaces use the GGX / Trowbridge–Reitz microfacet distribution, a masking-shadowing term, and a Fresnel term. Sampled directions update path throughput using the BSDF value, surface cosine, and sampling probability density.

The metallic–roughness material combines diffuse and specular reflection:

- At metallic zero, the material has a colored diffuse component and a neutral specular component.
- At metallic one, the diffuse component disappears and the base color controls specular reflection.
- Intermediate values blend these behaviors.

Sampling chooses between diffuse and specular directions. The throughput calculation uses the combined mixture PDF, accounting for both ways of generating a direction.

For glTF materials, perceptual roughness is converted to the microfacet parameter using alpha = (roughnessFactor * roughnessTexture)^2.

### glTF Mesh Loading

The renderer loads indexed triangle meshes from `.gltf` files using TinyGLTF. Scene traversal combines parent and child transforms, applying the resulting world transform to each mesh instance.

The loader reads vertex positions, normals, texture coordinates, tangents, and material properties. It handles accessor offsets and strides, validates buffer bounds, and supports unsigned 8-bit, 16-bit, and 32-bit triangle indices.

Loaded mesh data is uploaded to GPU buffers and combined into a single OptiX geometry acceleration structure. Per-triangle material IDs connect each intersection to its material.

The closest-hit program interpolates surface attributes using triangle barycentric coordinates. When vertex normals are unavailable, it uses the geometric triangle normal.

The benchmark mesh renders are shown in the performance results above.

### Texture, Metal Roughness, Normal Mapping

| Base Color | Metallic–Roughness | Normal |
| :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/e27e00d8-9869-4a39-9d99-0227dd082011" alt="Base-color texture applied" width="360" /> | <img src="https://github.com/user-attachments/assets/4da282d2-7856-4a8b-b0ac-6037eaf65571" alt="Diagnostic metallic–roughness map applied" width="360" /> | <img src="https://github.com/user-attachments/assets/36eb52eb-9690-4619-814b-f2e5253790e4" alt="Diagnostic normal map applied" width="360" /> |

Material images are decoded into unsigned 8-bit RGBA data and uploaded to CUDA arrays. CUDA texture objects provide linear filtering and the configured wrapping behavior. Base-color textures receive sRGB decoding, while metallic–roughness and normal maps are sampled as linear numerical data. All three use interpolated `TEXCOORD_0` coordinates. Texture transforms and additional UV sets are outside the current implementation’s scope.

**Base-color mapping** multiplies the sampled texture color by the material’s base-color factor, allowing surface color to vary across a mesh.

**Metallic–roughness mapping** reads roughness from the texture’s green channel and metallic from its blue channel. These values multiply the corresponding material factors. The combined roughness is squared to obtain the GGX microfacet parameter, while metallic remains linear. This creates spatially varying surface finishes without adding geometry.

**Normal mapping** perturbs the shading normal using a tangent-space direction decoded from the texture. The decoded X and Y components are scaled by the material’s normal-map strength before normalization.

The tangent frame uses interpolated glTF tangents when available. Otherwise, it is derived from triangle edges and UV differences. The tangent is orthogonalized against the interpolated normal, and its handedness determines the bitangent orientation. The mapped normal is accepted only when it faces both the appropriate geometric hemisphere and the viewing direction. Degenerate tangent frames retain the original shading normal. Normal mapping applies to metallic–roughness materials, while glass continues to use the geometric normal.

Together, these maps independently control surface color, material response, and shading detail:

<img src="https://github.com/user-attachments/assets/43dfa633-abe1-4518-8aec-3ab391c037ed" alt="Base-color, metallic–roughness, and normal maps combined" width="600" />

*All three maps applied together.*

### HDR Environment Lighting and Multiple Importance Sampling

| Environment MIS Disabled | Environment MIS Enabled |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/4c4cbd31-4290-49ab-8a36-76444e5dcd8e" alt="Environment lighting without MIS" width="360" /> | <img src="https://github.com/user-attachments/assets/fa2c5f14-f4a9-4186-ab7c-dc1be2c8a0a5" alt="Environment lighting with MIS" width="360" /> |

An rectangular HDR image supplies incoming radiance from the surrounding environment. Ray directions are converted to texture coordinates, allowing the environment to appear as the background and illuminate surfaces through material interactions. The image is loaded as floating-point data to preserve high radiance values, while lighting calculations and progressive accumulation remain in linear space.

With BSDF sampling alone, paths gather environment illumination when their sampled directions escape the scene. Small, bright regions of the environment can be difficult to sample consistently, producing high variance.

To sample the environment directly, the loader builds a cumulative distribution over its pixels. Each pixel is weighted by its luminance and spherical solid angle, with a small luminance floor to preserve sampling support in dark regions. The GPU selects a pixel using binary search, then samples a direction uniformly within that pixel’s solid angle.

For metallic–roughness materials, **multiple importance sampling (MIS)** combines two strategies:

- **Environment sampling:** Sample a direction from the environment distribution and trace a visibility ray.
- **BSDF sampling:** Sample a direction from the material distribution and evaluate the environment if the path escapes.

Both PDFs are expressed per unit solid angle. Their contributions are weighted using the power heuristic p_a^2/(p_a^2 + p_b^2}.

The renderer stores the previous BSDF PDF so that an escaping BSDF-sampled ray receives the complementary MIS weight. Direct-light contributions are accumulated separately from path throughput.

`USE_ENVIRONMENT_MIS` toggles direct environment sampling and MIS. 

### Depth of Field

| No DOF | DOF |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/9b3ce840-2d83-4ab5-948b-c7c038f368a6" alt="1" width="360" /> | <img src="https://github.com/user-attachments/assets/4c291d1b-6c14-4230-a87c-b7dd20ddd155" alt="cornell 50000samp" width="360" /> |

The OptiX camera implements a thin-lens model. A jittered pinhole ray first determines a point on the focal plane. The ray origin is then sampled uniformly over a circular aperture, and its direction is adjusted toward that focal point.

The camera exposes two JSON parameters:

| Parameter | Effect |
|---|---|
| `APERTURE_RADIUS` | Controls the aperture size. Zero produces a pinhole camera. |
| `FOCAL_DISTANCE` | Sets the focal plane’s distance along the camera’s forward direction. |

Depth of field is available when camera rays are generated inside OptiX raygen.

### Tone Mapping and Gamma Correction

RGB values in scenes are usually in a much larger range than 0 - 255, in order to account for the different intensity of light, so  we apply Reinhard tone mapping in order to map this back into the 8 bit rgb range, .


In addition, to account for the fact that humans are more sensitive to color changes in the dark, we must apply gamma correction to tailor the linear rgb values to human perception.

The result is converted to 8-bit color and written directly to the OpenGL pixel buffer. Tone mapping affects the displayed image while the accumulation buffer retains linear radiance.

| Without Reinhard and gamma correction | With Reinhard and gamma correction |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/3435923e-93de-416d-91fa-82be2120388b" alt="Without Reinhard and gamma correction" width="420" /> | <img src="https://github.com/user-attachments/assets/19f1a6e8-e5f8-477c-8004-b60cdb501649" alt="With Reinhard and gamma correction" width="420" /> |

Without the display conversion, the overall scene appears very dark, while the brightest reflections become nearly solid white. With Reinhard tone mapping and gamma correction, shadow and midtone detail is more visible and the bright reflections retain more visible variation. Reinhard compresses high radiance values, while gamma correction changes their display encoding.
## Build and Usage

### Installing the OptiX SDK

Download the NVIDIA OptiX SDK from the [official OptiX page](https://developer.nvidia.com/designworks/optix/download). This project was developed with OptiX SDK 9.1.0.

After installing or extracting the SDK, set OPTIX_ROOT in CMakeLists.txt to its installation directory. For example:

```cmake
set(OPTIX_ROOT "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0")
```

The selected directory should contain include/optix.h. Reconfigure CMake after changing the path.

The CUDA Toolkit must also be installed separately. It is used to compile the renderer and generate the PTX loaded by OptiX.


### Build Configuration

The project uses CMake with C++17 and CUDA17. The current configuration targets an RTX 3060 and references OptiX SDK 9.1.0.

Before configuring the project, update OPTIX_ROOT in CMakeLists.txt to your local OptiX SDK installation.

Build in Release mode for performance measurements. The current CMake configuration enables CUDA device debugging (`-G`) for both Debug and RelWithDebInfo.

### CMake Integration

`CMakeLists.txt` was modified beyond adding source files to integrate OptiX:

- Added the TinyGLTF and OptiX include directories.
- Added a separate `optixPrograms` target that compiles `optixPrograms.cu` to PTX.
- Configured the PTX target for compute capability 8.6 using `86-virtual`.
- Disabled separable compilation for the PTX target.
- Made the renderer depend on the PTX target.
- Defined `OPTIX_PTX_PATH` so the renderer can locate the generated PTX file.

The main executable uses native CUDA architecture selection, while the OptiX PTX target explicitly uses compute capability 8.6.

### Scene Selection

The executable accepts a JSON scene file:

```bash
cis565_path_tracer scenes/cornell.json
```

We must still give the json file as input because it supplies camera and rendering settings, including resolution, sample count, maximum bounce depth, and output filename.

The current OptiX scene selects its glTF model and HDR environment separately in `main.cpp`:

```cpp
loadGltf("../scenes/DamagedHelmet/DamagedHelmet.gltf",
         loadedMeshes, loadedImages, loadedTextures);

loadEnvironment("../img/greenwich_park_4k.hdr", environment);
```

Keep each glTF file together with its referenced buffers and textures.

### Rendering Configuration

The main architecture toggles are defined in `pathtrace.cu`:

```cpp
#define USE_OPTIX 1
#define USE_PARTITION 0
#define USE_RAYGEN_LOOP 1
#define USE_RAYGEN_CAMERA 1
```

Direct environment sampling and MIS are controlled in `optixPrograms.cu`:

```cpp
#define USE_ENVIRONMENT_MIS 1
```

These are compile-time settings and require rebuilding after changes.

## References and Credits

### Libraries and Framework

- NVIDIA OptiX — ray-tracing pipeline and acceleration structure support.
- NVIDIA CUDA, Thrust, and CUB — GPU execution and path partitioning.
- TinyGLTF — glTF parsing and material-image loading.
- stb_image / stb_image_write — image loading and output.

### Rendering References

- Physically Based Rendering
  - Diffuse Reflection.
  - Dielectric BSDF.
  - Roughness Using Microfacet Theory.
  - The Thin Lens Model and Depth of Field.
  - A Better Path Tracer.
- glTF 2.0 specification - scene transforms, accessors, material factors, and texture conventions.
- NVIDIA OptiX documentation - modules, program groups, pipelines, shader binding tables, and acceleration structures.
- My earlier GLSL path tracer and PBR shader - references for the material implementation.
- Typescript environment lighting MIS implementation from a pbr group chat.

### Scene Assets

This work is based on "Gundam Unicorn" (https://sketchfab.com/3d-models/gundam-unicorn-1410ff9dd1c94807a00b8a0936170196) by kurojishi (https://sketchfab.com/kurojishi) licensed under CC-BY-NC-4.0 (http://creativecommons.org/licenses/by-nc/4.0/)

This work is based on "Jigglypuff" (https://sketchfab.com/3d-models/jigglypuff-76ab579e21b74697a90efe566a7293c3) by DreamNoms (https://sketchfab.com/DreamNoms) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

This work is based on "Kirby Fofinho" (https://sketchfab.com/3d-models/kirby-fofinho-79bab9e6d16c43aebf04a70abd0c7dfd) by Dante Rossi - Zion (https://sketchfab.com/rossidante) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

This work is based on "KIRBY MARIO BROS" (https://sketchfab.com/3d-models/kirby-mario-bros-42081342748b4a2c9ea14ff5d67da63f) by ElOsitoAzul (https://sketchfab.com/ElOsitoAzul) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

This work is based on "Spheal Doll 3D Scan" (https://sketchfab.com/3d-models/spheal-doll-3d-scan-f084074fa24a4ca49a99f201fb978e9c) by Fallen_ice (https://sketchfab.com/Fallen_ice) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

This work is based on "Spheal Pokemon" (https://sketchfab.com/3d-models/spheal-pokemon-05815c4e8d714690bbb615588585d941) by Stopapypa (https://sketchfab.com/Stopapypa) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

This work is based on "Spiderman" (https://sketchfab.com/3d-models/spiderman-b5c87f4ac60b4f5184805edebdd981f4) by Carissa (https://sketchfab.com/cclarke468) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

This work is based on "Iron-Man Mark 85 | Rigged" (https://sketchfab.com/3d-models/iron-man-mark-85-rigged-dde1085c464d4f8da259fe6669ae4dd2) by 9A Films / Nihar Arora (https://sketchfab.com/Nihar-9Afilms) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

This work is based on "Toothless - HTTYD" (https://sketchfab.com/3d-models/toothless-httyd-3aca4dc141f141d794bce159dcf6bb62) by Malin Helene Rosenkrantz Ottesen (https://sketchfab.com/MH_Rosenkrantz) licensed under CC-BY-NC-SA-4.0 (http://creativecommons.org/licenses/by-nc-sa/4.0/)

This work is based on "Wall-E" (https://sketchfab.com/3d-models/wall-e-79a0bff7dbc34d0098cabc3c5717b48c) by snakadaktal (https://sketchfab.com/snakadaktal) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

This work is based on "Wooden Table (game ready asset)" (https://sketchfab.com/3d-models/wooden-table-game-ready-asset-7283ac1841504452b53005b8103bb064) by Pixel Life (https://sketchfab.com/pixellife) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

"Box", "Suzanne", "FlightHelmet", "DamagedHelmet", "Sponza": KhornosGroupArchives glTF-Sample-Models (https://github.com/KhronosGroupArchives/glTF-Sample-Models)

Jupiter Texture: https://www.solarsystemscope.com/textures/
