CUDA Path Tracer - Joyful Snowman
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Yanfu Ou
* Tested on: Ubuntu 24.04 LTS, AMD Ryzen 7 7840HS, Nvidia RTX 5070 Laptop GPU (8 GB VRAM), 64 GB DDR4 RAM

![Joyful snowman](images/snowman-in-brick-FINAL.png)

A CUDA path tracer built around an artistic reimagination of a snowman in a colorful house. Fresnel reflection and refraction shows the snowman's head. The mirror-like specular sphere demonstrates the snowman's upper body. Last, but not least, the crimson red illustrates the snowman's lower body. The snowman sits in a joyful colored room defined by gold, blue, and pink walls.

The renderer is a combination of diffuse and specular shading, refraction with Fresnel, depth of field, direct lighting, Halton sampling, Russian roulette, stochastic antialiasing, and shading paths sorted into contiguous memory by material type. Every feature is a `0`/`1` switch in `src/featureToggles.h`. Flip a switch and rebuild to see the difference.

## Compilation

Requires CMake, a CUDA toolkit, and OpenGL with GLFW and GLEW. Run the following from repo root:

```bash
mkdir -p build && cd build
cmake ..
make -j8
./bin/cis565_path_tracer ../scenes/snowman.json
```

The binary is `build/bin/cis565_path_tracer`. Pass any scene JSON as the argument. Feature switches live in `src/featureToggles.h`; change one and run `make -j` again.

## Features 

### 1. Depth of field

The camera is modeled as a thin lens. The ray origin is shifted to a random point on a disk of radius `0.15`, and the ray is aimed at the focal plane so that plane stays sharp. In the snowman scene, the focal distance is `10.5`, the distance along the view direction from the camera at `z = 10.5` to the plane `z = 0` through the specular sphere. The specular upper body of the snowman stays in focus, while the brick wall in the back is blurry. Setting the focal distance to `15.27` focuses near the blue back wall. That wall is centered at `z = -5`, about `15.5` from the camera, so the snowman goes soft and the bricks sharpen. Setting `DEPTH_OF_FIELD` to `0` returns to a pinhole: every pixel shares the camera origin, and the whole room is equally sharp.

| Focused on the snowman | Focused on the blue back wall |
| --- | --- |
| ![Focused on snowman](images/depth-of-field/snowman-DOF-specular-sphere.png) | ![Focused on the blue back wall](images/depth-of-field/snowman-DOF-blue-wall.png)  |
| ![Focused on snowman, zoomed](images/depth-of-field/snowman-DOF-specular-sphere-zoomed.png) | ![Focused on the blue back wall, zoomed](images/depth-of-field/snowman-DOF-blue-wall-zoomed.png)  |

_above images are generated at depth = 8 and 1k iters_

For a clearer demo, the same effect in the Cornell box, between a sphere and a cube.

| Focused on the cube | Focused on the sphere |
| --- | --- |
| ![Depth of field, cube](images/depth-of-field/cornell-DOF-cube.png) | ![Depth of field, sphere](images/depth-of-field/cornell-DOF-sphere.png) |

_above images are generated at depth = 8 and 5k iters_


#### Performance
The lens adds two random numbers per camera ray and gives neighboring pixels different origins. This tracer intersects one ray per thread, so that incoherence does not change how traversal runs. The extra math is small next to testing every object. The focal distance and the lens radius are compile-time constants. Nothing else was added to speed the feature up.

#### Hypothetical CPU comparison
A CPU ray packet wants neighbors to share an origin, so the lens hurts that style of traversal. This GPU path does not use packets. A CPU version would trace the same rays more slowly, one at a time. Reading the focal distance from the scene file would make it tunable without a rebuild. A smaller aperture would also pull neighboring origins back together, which is the part of the feature a packet tracer would actually feel.

### 2. Refraction

Glass uses Snell's law through `glm::refract` and Schlick's approximation to choose reflection or transmission. The index of refraction is the material `IOR` (1.5 in these scenes). Entering glass uses `1 / IOR`. Leaving glass uses `IOR`. The path reflects on total internal reflection, and when the Fresnel sample is below the reflection probability. A sample above that probability transmits. Setting `REFRACTION` to `0` compiles out the glass branch, and the head takes the mirror path instead.

The glass head of the snowman shows this effect: the room bends through the sphere, and the rim reflects. The mirror body next to it is the same scene with that branch removed.

| Left view | Right view |
| --- | --- |
| ![Refraction, left](images/refraction-reflection/snowman-refraction-left-view.png) | ![Refraction, right](images/refraction-reflection/snowman-refraction-right-view.png) |

_above images are generated at depth = 8 and 1k iters_

For a clearer demo, the same effect in the Cornell box with a glass sphere, a diffuse sphere, and a reflective cube.

| Front view | Side view |
| --- | --- |
| ![Refraction, front](images/refraction-reflection/cornell-refraction-specular-blue-sphere-front.png) | ![Refraction, side](images/refraction-reflection/cornell-refraction-specular-blue-sphere-side.png) |

_above images are generated at depth = 8 and 5k iters_


#### Performance
Refraction does not add primitives. It adds a branch in the shade kernel, and paths that hit glass continue through the sphere instead of stopping at the surface, so those pixels need the full trace depth. Most of the frame is diffuse bricks, so a warp only diverges on the sphere. Nothing else was added to speed the feature up.

#### Hypothetical CPU comparison
On the GPU that branch diverges only for threads that hit glass. A warp that is mostly diffuse stays on the diffuse path. A CPU version has no warp divergence, and it also has no thousands of pixels in flight. A separate glass kernel would keep the diffuse warps off the Fresnel branch. A rough dielectric, so transmission is a lobe instead of one perfect ray, would be the next look for the material.

### 3. Direct lighting

On the last bounce of a diffuse path, the tracer samples a random point on a random emissive quad and sends a shadow ray there. The contribution is the Lambertian term times the light's cosine, divided by distance squared and by the pdf of picking that light and that point. If the shadow ray hits anything else, the sample is black. Specular and glass paths do not take this branch, because a next-event ray is not the reflected or refracted direction. Setting `DIRECT_LIGHTING` to `0` skips the shadow ray.

#### Depth 1:

| No direct light | Direct light |
| --- | --- |
| ![Depth 1, no direct light, snowman](images/direct-lighting/snowman-direct-ray-no-depth1.png) | ![Depth 1, direct light, snowman](images/direct-lighting/snowman-direct-ray-depth1.png)| 
| ![Depth 1, no direct light, Cornell](images/direct-lighting/cornell-direct-ray-no-depth1.png) | ![Depth 1, direct light, Cornell](images/direct-lighting/cornell-direct-ray-depth1.png) |


At depth 1 the snowman scene without direct light is black except for the ceiling panel. The camera ray ends on the first hit, so the bricks, the floor, and the crimson body never get a second bounce that could find the light.

With direct light, that one shadow ray is enough to color the pink, gold, and blue walls and the floor. The snowman stays a dark silhouette. The glass head and the mirror body are not diffuse, so they do not take a next-event sample. The crimson sphere only picks up a little red on the top that faces the lamp. The dark patch on the floor under the snowman is the shadow ray being blocked.

The Cornell box shows the same split: only the lamp, plus the diffuse surfaces the shadow ray can see. The snowman makes the blocked shadow, and the dark glass and mirror, easier to pick out.

#### Depth 2:

| No direct light | Direct light |
| --- | --- |
| ![Depth 2, no direct light, snowman](images/direct-lighting/snowman-direct-ray-no-depth2.png) | ![Depth 2, direct light, snowman](images/direct-lighting/snowman-direct-ray-depth2.png) |
| ![Depth 2, no direct light, Cornell](images/direct-lighting/cornell-direct-ray-no-depth2.png) | ![Depth 2, direct light, Cornell](images/direct-lighting/cornell-direct-ray-depth2.png) |


At depth 2 the snowman scene without direct light is no longer black. The second bounce can reach the ceiling light, so the brick walls, the floor, and a hint of red on the crimson body show up. Most of those bounces still miss the small lamp, so the walls stay dim and speckled. The glass head and the mirror body only flash white where the reflection happens to point at the light.

With direct light, that second bounce is aimed at the lamp. The pink, gold, and blue bricks, the floor, and the crimson body fill in, and the ceiling picks up light too. The mirror body reflects the colored walls beside it instead of staying a black sphere with one white glint. The floor still has a shadow under the snowman, but the rest of the ground is evenly lit.

The Cornell box does the same thing at depth 2. The second hit can find the lamp on its own, and direct lighting aims that hit at the lamp instead.

#### Performance
Each diffuse path that reaches its last bounce traces one extra ray, and that ray walks the full object list. The depth-1 pair is the quality result: surfaces that never bounce into the lamp still get one sample. No acceleration structure was added for the shadow test.

#### Hypothetical CPU comparison
The GPU still does the extra ray as one thread per path, so the pass scales with the number of live paths. A CPU version would walk the same shadow rays serially. Multiple importance sampling, or sampling the light on every diffuse bounce instead of only the last one, would cut noise further.

### 4. Halton sampling

Random numbers come from a Halton sequence, one prime per dimension, with a per-pixel Cranley-Patterson shift so neighboring pixels do not share the same sequence. Past 32 dimensions the sample falls back to a hash. Camera jitter, the lens, diffuse bounces, Fresnel, direct-light selection, and Russian roulette consume this stream in order. Pixel jitter and the lens still consume their two dimensions when those switches are off, so a later bounce stays on the same coordinates. Direct lighting and Russian roulette do not. Their `sample1D` calls sit inside the feature switch, so turning either off skips those dimensions. Setting `HALTON_SAMPLING` to `0` uses `thrust`'s uniform generator.

#### Snowman, depth 3:

| Iterations | Halton | Uniform random |
| --- | --- | --- |
| 8 | ![Snowman Halton, 8](images/halton-sampling/snowman-halton-8iter-3depth.png) | ![Snowman random, 8](images/halton-sampling/snowman-halton-no-8iter-3depth.png) |
| 16 | ![Snowman Halton, 16](images/halton-sampling/snowman-halton-16iter-3depth.png) | ![Snowman random, 16](images/halton-sampling/snowman-halton-no-16iter-3depth.png) |
| 64 | ![Snowman Halton, 64](images/halton-sampling/snowman-halton-64iter-3depth.png) | ![Snowman random, 64](images/halton-sampling/snowman-halton-no-64iter-3depth.png) |
| 128 | ![Snowman Halton, 128](images/halton-sampling/snowman-halton-128iter-3depth.png) | ![Snowman random, 128](images/halton-sampling/snowman-halton-no-128iter-3depth.png) |
| 256 | ![Snowman Halton, 256](images/halton-sampling/snowman-halton-256iter-3depth.png) | ![Snowman random, 256](images/halton-sampling/snowman-halton-no-256iter-3depth.png) |

_above images are generated at depth 3_

On the snowman, Halton fills the pixel more evenly, so the same iteration count is less blotchy. The gap is largest at 8 to 64 samples and shrinks once both sequences have enough samples.

At 8 iterations both images are mostly noise, but they are different kinds of noise. Uniform random looks like static: pink, gold, and blue specks are scattered with no structure. The brick courses are hard to find. Halton noise is more ordered. The floor, the mortar lines, and the crimson body already read as surfaces instead of a cloud of points.

By 64 iterations the Halton frame has a fine grain and the mirror is picking up pink and gold. The uniform frame is still blotchy, especially on the gold wall and the floor.

At 256 iterations Halton has cleaned up the brick joints, the floor, and the glass head, which shows the room through the sphere. Uniform random at the same count still has a coarse grain across the walls and floor, and the glass head stays a dark sphere with a lamp highlight. The mirror settles in both because its reflection direction is fixed. The noise in that pixel comes from the diffuse surface it happens to see. The crimson body is diffuse, like the floor, so it does depend on a random bounce. It reads as a solid sphere earlier because the red is strong enough to show through the grain. The bricks and the floor are where the sequence shows up.

#### Cornell box, depth 3
| Iterations | Halton | Uniform random |
| --- | --- | --- |
| 8 | ![Halton, 8](images/halton-sampling/cornell-halton-8iter-3depth.png) | ![Random, 8](images/halton-sampling/cornell-halton-no-8iter-3depth.png) |
| 16 | ![Halton, 16](images/halton-sampling/cornell-halton-16iter-3depth.png) | ![Random, 16](images/halton-sampling/cornell-halton-no-16iter-3depth.png) |
| 64 | ![Halton, 64](images/halton-sampling/cornell-halton-64iter-3depth.png) | ![Random, 64](images/halton-sampling/cornell-halton-no-64iter-3depth.png) |
| 128 | ![Halton, 128](images/halton-sampling/cornell-halton-128iter-3depth.png) | ![Random, 128](images/halton-sampling/cornell-halton-no-128iter-3depth.png) |
| 256 | ![Halton, 256](images/halton-sampling/cornell-halton-256iter-3depth.png) | ![Random, 256](images/halton-sampling/cornell-halton-no-256iter-3depth.png) |

The glass sphere, the diffuse sphere, and the reflective box in the Cornell box follow the same pattern: Halton settles the walls and the glass earlier, and the gap shrinks by 256 iterations.

#### Performance
The radical inverse is a short integer loop per sample. It is more arithmetic than a single uniform draw, and threads in a warp can exit that loop at different sample indices. The iteration grid is the result that matters: the same sample count is cleaner, especially from 8 to 64 iterations, so the image becomes usable sooner. The radical inverse itself was not accelerated.

#### Hypothetical CPU comparison
A CPU version has the same arithmetic and none of the warp issue, but it cannot shade the whole image at once. A Sobol sequence, or a precomputed sample table in constant memory, would remove the per-sample loop.

### 5. Russian roulette

After the third bounce, a path survives with probability `q`, where `q` is its brightest throughput channel clamped to `[0.05, 1]`. Survivors are scaled by `1 / q`, so the estimate stays unbiased. Paths that fail the test are terminated. Direct-light rays are not eligible. Setting `RUSSIAN_ROULETTE` to `0` skips the test and lets paths run to `DEPTH`.

The test is unbiased, so the expected brightness matches a render that always runs to full depth. There is no before-and-after pair of images for that reason. The measured difference is time.

Set `iterations` to 64, `depth` to 8. Ran the following experiment. All three scenes are open. The Cornell box and the snowman have no front wall, so light can leave toward the camera. The sphere is the extreme case.

| Run | Scene | Room | Objects | Russian roulette | Time (s) |
| --- | --- | --- | --- | --- | --- |
| 1 | Cornell box | Open | 9 | on | 3.28 |
| 2 | Snowman | Open | 93 | on | 6.79 |
| 3 | Sphere | Open | 1 | on | 2.06 |
| 4 | Cornell box | Open | 9 | off | 3.57 |
| 5 | Snowman | Open | 93 | off | 7.26 |
| 6 | Sphere | Open | 1 | off | 2.04 |

![Roulette graph](images/charts/Russian-Roulette-On-vs-Off.png)

Russian roulette is faster in the two rooms and flat on the sphere. With roulette on, the Cornell box drops from 3.57 s to 3.28 s, about 8%, and the snowman drops from 7.26 s to 6.79 s, about 6.5%. The single sphere stays at 2.06 s and 2.04 s.

The first three bounces always run. After that, a dim path can die, and stream compaction keeps it out of later intersection launches. In these open rooms a ray can also escape through the front, or hit the light, without roulette. Roulette is the extra cut on paths that are still inside and already dim. The snowman saves more time in seconds, 0.47 s against the Cornell box's 0.29 s, because each killed ray would have tested 93 objects instead of 9. The fraction is still small, about 8% and 6.5%, because those early bounces are most of the work. Survival uses the path throughput, the product of albedos, not how strong the ceiling light is.

The sphere is the control. Most camera rays miss it and terminate on the first bounce, before roulette is allowed to run, so the two times match. A closed room, one with a front wall so light cannot escape, would carry more paths into that tail. The same test should save a larger share there than the 8% and 6.5% measured on these open rooms.

#### Hypothetical CPU comparison
This does not change the final image's expected brightness. It shortens paths whose remaining color is already small. The GPU benefits directly: `thrust::remove_if` drops the dead paths, so later intersection launches have fewer threads. A CPU version would skip those rays too, but the savings show up as less serial work rather than a smaller kernel launch. Starting roulette one bounce earlier, or using the path's luminance instead of the max channel, would terminate more aggressively.

## Core features

### Cosine-weighted diffuse BSDF

A diffuse hit multiplies the path throughput by the material color and scatters a new ray from the intersection. The direction is cosine-weighted over the hemisphere around the normal: `u1` sets the polar angle with `sqrt(u1)`, and `u2` sets the azimuth. Brighter samples land closer to the normal, which is the Lambertian distribution, so the bounce does not need an extra cosine weight in the estimator. The brick walls, the floor, the ceiling, and the crimson snowman body all use this.

This is the baseline path tracer. Turning this core feature off would result in a non-working renderer.

#### Hypothetical CPU comparison
Diffuse paths are noisy. Each bounce can go anywhere in the hemisphere, so most rays miss the small ceiling light unless direct lighting picks it for them. The hemisphere sample is a few square roots and a cross product, which is nothing next to intersecting every object. On the GPU, a warp of diffuse hits stays together and runs the same math. A CPU version does the identical sample per ray and just gets through fewer rays. Sampling the light on every diffuse bounce, instead of only the last one, would converge faster.

### Perfect specular reflection

A mirror hit reflects the incoming ray with `glm::reflect` and multiplies throughput by the material color. The new origin is nudged `0.001` along the reflected direction so the next trace does not hit the same surface. A perfect mirror consumes no random direction. Under Halton sampling it still advances two dimensions, so a later diffuse bounce stays on the coordinates it would have used if the mirror had been diffuse. The snowman's upper body is this material. With `REFRACTION` off, the glass head takes this path too and becomes a mirror.

#### Hypothetical CPU comparison
The reflection is a fixed direction, so a specular pixel is sharp after one sample. The cost shows up in path length: the mirror keeps the path alive and sends it somewhere else in the room, often into the bricks or the glass. On the GPU, mirror hits diverge from diffuse hits in the same warp. Sorting by material is an attempt to keep those branches together. The timing below shows that the sort costs more than the divergence it removes on these scenes. A CPU shader pays the same reflection math without that divergence, and it traces the continued rays one at a time. A roughness parameter, so the reflection is a lobe instead of a single ray, would be the next version of this BSDF.

### Stochastic sampled antialiasing

Each camera ray is jittered to a random point inside its pixel, so edges are averaged across samples instead of snapped to the pixel grid. Without jitter, the sphere silhouette looks like a staircase. With jitter, the same edge softens as iterations accumulate. Setting `STOCHASTIC_ANTIALIASING` to `0` sends the ray through the pixel center. The two Halton dimensions are still consumed, so later features stay on the same coordinates.

| No antialiasing | Antialiasing |
| --- | --- |
| ![No antialiasing](images/anti-alising/snowman-anti-alised-no.png) | ![Antialiasing](images/anti-alising/snowman-anti-alised.png) |
| ![No antialiasing, zoomed](images/anti-alising/snowman-anti-alising-no-zoomed.png) | ![Antialiasing, zoomed](images/anti-alising/snowman-anti-alising-zoomed.png) |

For a clearer demo, a sphere in the Cornell box.

| No antialiasing | Antialiasing |
| --- | --- |
| ![No antialiasing](images/anti-alising/cornell-no-anti-alising.png) | ![Antialiasing](images/anti-alising/cornell-anti-alised.png) |
| ![No antialiasing, zoomed](images/anti-alising/cornell-no-anti-alising-zoomed.png) | ![Antialiasing, zoomed](images/anti-alising/cornell-anti-alised-zoomed.png) |



#### Performance
The extra cost is two random numbers per camera ray and a slightly less coherent set of primary rays. Intersection and shading dominate, so the jitter itself is cheap. Nothing else was added to speed it up.

#### Hypothetical CPU comparison
On a GPU, neighboring threads already trace different pixels, and the jitter does not add much divergence. A CPU version would do the same math per ray and would not get the pixel-parallel throughput. A stratified or blue-noise pattern inside the pixel would converge with less noise than independent jitter.

### Sorting paths by material

Before shading, intersections and their paths are sorted by material id with `thrust::sort_by_key`. Threads in a warp are then more likely to take the same BSDF branch: diffuse, specular, or glass. Setting `SORT_BY_MATERIAL` to `0` shades in pixel order instead.

Set `iterations` to 256, `depth` to 8. Ran the following experiment. The Cornell box and the snowman are open rooms. The sphere is open and almost empty.

| Run | Scene | Room | Objects | SORT_BY_MATERIAL | Time (s) |
| --- | --- | --- | --- | --- | --- |
| 1 | Cornell box | Open | 9 | on | 10.12 |
| 2 | Snowman | Open | 93 | on | 24.31 |
| 3 | Sphere | Open | 1 | on | 5.26 |
| 4 | Cornell box | Open | 9 | off | 5.41 |
| 5 | Snowman | Open | 93 | off | 18.48 |
| 6 | Sphere | Open | 1 | off | 5.26 |

![sort-by-material](images/charts/sort_by_material-chart.png)

With sorting on, the Cornell box rises from 5.41 s to 10.12 s, and the snowman rises from 18.48 s to 24.31 s. The sphere stays at 5.26 s.

`thrust::sort_by_key` reorders every live path before shading. Each path carries its ray, its color, and its Halton state, and the pass runs on every bounce. The shading that sort is meant to help is a short branch among diffuse, mirror, and glass. In the Cornell box and the snowman, most pixels are diffuse walls, so a warp is already on the same BSDF. The mirror and the glass sphere are the pixels that take another branch. The sort still walks every live path to group those few.

The extra time is 4.71 s for the Cornell box and 5.83 s for the snowman. That fits a per-path sort cost that does not depend on how many objects a ray has to test. The snowman costs a bit more because more of its paths are still alive on later bounces, so the sort has more keys. Its share of the total is smaller because intersecting 93 objects is already most of the frame. The sphere has one material, and most rays miss and die on the first bounce, so the two times match.

#### Hypothetical CPU comparison
The image does not change. The measured cost is the device-wide sort, which on these scenes is larger than the divergence it removes. A CPU shader has no warps, so the same sort would only add overhead. Sorting once into separate material queues, instead of re-sorting the whole buffer every bounce, would drop that cost. The sort would earn its time if the BSDF branches were much longer, or if the frame were split more evenly between materials. A bounding volume hierarchy would matter more for the brick scene than another shading tweak, because intersection still tests every object.

### Stream compaction

After shading, any path with `remainingBounces == 0` is removed with `thrust::remove_if`. A path reaches zero when it hits a light, misses the scene, fails Russian roulette, or runs out of depth. The next intersection launch only covers the paths still alive, and the bounce loop stops when none remain.

One iteration at depth 8. Bounce 0 is every pixel, 640000 rays, before anything is traced. Each later row is how many rays compaction kept for the next intersection. The sphere rows after bounce 1 are shown as 0 because those launches never happen.

![num rays remaining chart](images/charts/Remaining-Rays-After-Each-Bounce.png)

| Bounce | Cornell box | Snowman | Sphere |
| --- | --- | --- | --- |
| 0 | 640000 | 640000 | 640000 |
| 1 | 522789 | 519074 | 0 |
| 2 | 360422 | 344938 | 0 |
| 3 | 304828 | 278902 | 0 |
| 4 | 167097 | 216931 | 0 |
| 5 | 141523 | 194710 | 0 |
| 6 | 126446 | 179353 | 0 |
| 7 | 117584 | 168930 | 0 |
| 8 | 108266 | 124101 | 0 |
| 9 | 0 | 0 | 0 |

The sphere is finished after one bounce. A ray either misses and dies, or hits the emissive sphere and dies. Compaction removes the whole buffer, and bounces 2 through 9 never launch. Without compaction those dead rays would keep a thread through depth 8. That is where compaction helps an open scene most: almost every ray terminates immediately, and the later launches would be empty work.

The Cornell box and the snowman are open too. The camera sits outside the room, and there is no front wall. On the first hit a miss is still uncommon, because the room fills the frame. The drop from bounce 0 to bounce 1 is 117211 rays in the Cornell box, about 18% of the image, and 120926 in the snowman, about 19%. Those died on the first hit, mostly on the ceiling light. Rays that hit a wall stay alive.

By bounce 3 the Cornell box is down to 304828 rays, about half the image. Some of that loss is the light, and some is rays that left through the open front. Bounce 4 is the first count that includes Russian roulette. Cornell falls to 167097, and the snowman falls to 216931. Roulette, light hits, and escapes all sit in that drop. The snowman falls less because its wall albedos are higher. Several materials have a channel at 1, so the throughput stays large and `q` stays high. Cornell's red, green, and blue are 0.85, and after a few bounces that product fails the test more often. The ceiling light's emittance does not enter `q`.

From there both rooms shrink slowly. The bounce 8 row is the next intersection launch: 108266 rays in Cornell and 124101 in the snowman. On a diffuse path that row is the direct-light shadow ray, created on the last bounce and not yet traced. Bounce 9 traces those rays, they finish, and the buffer is empty.

The later launches are where compaction earns its time in the rooms. The Cornell bounce 4 row is about a quarter of the full image, and that is the size of the following intersection. The snowman still tests all 93 objects on every surviving ray, so dropping the dead ones matters more there than in the 9-object box. I did not time a build with compaction turned off. The benefit is the smaller launch. A closed room, with a front wall, would terminate fewer rays early, because nothing escapes toward the camera. Compaction would remove less until the light and roulette start cutting the tail.

#### Hypothetical CPU comparison
Without compaction, every pixel would keep a thread through the full trace depth, including pixels that already hit the ceiling light on bounce one. `thrust::remove_if` is a device-wide stream compaction, so the GPU spends a prefix-sum-style pass to pack the survivors, then launches a smaller kernel. A CPU loop would just skip dead rays with a branch, which is cheaper per ray and slower overall because the live rays are still serial. A shared-memory compaction, or compacting before shading as well as after, would avoid shading lanes that already missed.
