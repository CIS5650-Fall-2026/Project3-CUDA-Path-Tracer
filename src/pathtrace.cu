#include "pathtrace.h"

#include "config.h"
#include "display.cuh"
#include "gui_data.h"
#include "intersections.cuh"
#include "math_utils.h"
#include "render_settings.cuh"
#include "sampling.cuh"
#include "scene.h"
#include "scene_structs.h"
#include "shading.cuh"
#include "thrust_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define check_cuda_error(msg) check_cuda_error_fn(msg, FILENAME, __LINE__)
void check_cuda_error_fn(const char* msg, const char* file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

static Scene* host_scene = NULL;
static GuiDataContainer* gui_data = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static Light* dev_lights = NULL;
static MatId* dev_isect_mat_ids = NULL;
static ShadowRay* dev_shadow_rays = NULL;
static cudaArray_t env_array = NULL;
static cudaTextureObject_t env_texture;
static LightSampler light_sampler{};

cudaStream_t pt_stream;

__constant__ RenderSettings c_settings;
void upload_settings(const RenderSettings& s) {
    cudaMemcpyToSymbolAsync(c_settings, &s, sizeof(s), 0, cudaMemcpyHostToDevice, pt_stream);
}

void upload_materials(const Scene& s) {
    cudaMemcpyAsync(dev_materials, s.materials.data(), s.materials.size() * sizeof(Material),
                    cudaMemcpyHostToDevice, pt_stream);
}

void init_data_container(GuiDataContainer* imgui_data) {
    gui_data = imgui_data;
}

static void free_env() {
    if (env_texture) {
        cudaDestroyTextureObject(env_texture);
        cudaFreeArray(env_array);
    }
    env_texture = 0;
    env_array = nullptr;
}

void pathtrace_upload_env(const Scene& s) {
    cudaStreamSynchronize(pt_stream);
    free_env();
    if (s.env.pixels.empty()) {
        return;
    }

    std::vector<float4> upload_pixels(s.env.pixels.size());

    for (size_t i = 0; i < upload_pixels.size(); ++i) {
        const glm::vec3 rgb = s.env.pixels[i];
        upload_pixels[i] = make_float4(rgb.r, rgb.g, rgb.b, 1.0f);
    }

    const cudaChannelFormatDesc format = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&env_array, &format, s.env.width, s.env.height);
    const size_t row_bytes = static_cast<size_t>(s.env.width) * sizeof(float4);
    cudaMemcpy2DToArray(env_array, 0, 0, upload_pixels.data(), row_bytes, row_bytes,
                        s.env.height, cudaMemcpyHostToDevice);

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = env_array;

    cudaTextureDesc sampler{};
    sampler.addressMode[0] = cudaAddressModeWrap;
    sampler.addressMode[1] = cudaAddressModeClamp;
    sampler.filterMode = cudaFilterModeLinear;
    sampler.readMode = cudaReadModeElementType;
    sampler.normalizedCoords = 1;

    cudaCreateTextureObject(&env_texture, &resource, &sampler, nullptr);
}

void pathtrace_init(Scene* scene) {
    const CameraData& cam = scene->state.camera;
    const int num_pixels = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, num_pixels * sizeof(glm::vec3));
    cudaMalloc(&dev_paths, num_pixels * sizeof(PathSegment));
    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));
    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMalloc(&dev_intersections, num_pixels * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_shadow_rays, num_pixels * sizeof(ShadowRay));
    cudaMalloc(&dev_isect_mat_ids, num_pixels * sizeof(MatId));

    cudaStreamCreate(&pt_stream);

    pathtrace_upload_env(*scene);

    check_cuda_error("pathtrace_init");
}

void pathtrace_reset(Scene* scene) {
    host_scene = scene;
    upload_settings(scene->state.settings);
    upload_materials(*scene);

    const CameraData& cam = host_scene->state.camera;
    const int num_pixels = cam.resolution.x * cam.resolution.y;

    cudaMemset(dev_image, 0, num_pixels * sizeof(glm::vec3));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Light),
               cudaMemcpyHostToDevice);

    light_sampler.lights = dev_lights;
    light_sampler.num_lights = static_cast<int>(scene->lights.size());
    light_sampler.env_texture = env_texture;
    bool has_env = env_texture != 0;
    if (!has_env) {
        light_sampler.p_env = 0.f;
    } else if (light_sampler.num_lights == 0) {
        light_sampler.p_env = 1.f;
    } else {
        light_sampler.p_env = 1.f / static_cast<float>(light_sampler.num_lights + 1);
    }

    cudaMemset(dev_intersections, 0, num_pixels * sizeof(ShadeableIntersection));
    cudaMemset(dev_shadow_rays, 0, num_pixels * sizeof(ShadowRay));
    cudaMemset(dev_isect_mat_ids, 0, num_pixels * sizeof(MatId));

    check_cuda_error("pathtrace_reset");
}

void pathtrace_free() {
    cudaStreamSynchronize(pt_stream);

    cudaFree(dev_image); // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_isect_mat_ids);
    cudaFree(dev_lights);
    cudaFree(dev_shadow_rays);

    free_env();

    cudaStreamDestroy(pt_stream);

    check_cuda_error("pathtrace_free");
}

__global__ void gen_ray_from_cam(CameraData cam, int iter, int trace_depth,
                                 PathSegment* path_segments) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = path_segments[index];

        RngEng rng = make_seeded_rng(iter, index, -1);
        UnifDist<float> u01(0, 1);

        glm::vec2 offset = glm::vec2(u01(rng), u01(rng));
        glm::vec2 sub_pixel_sample = glm::vec2(x, y) + offset;

        Ray& ray = segment.ray;
        ray.org = cam.position;
        ray.dir = glm::normalize(
            cam.view -
            cam.right * cam.pixel_length.x * (sub_pixel_sample.x - (float)cam.resolution.x * 0.5f) -
            cam.up * cam.pixel_length.y * (sub_pixel_sample.y - (float)cam.resolution.y * 0.5f));

        if (cam.lens_radius > 0.f) {
            float t = cam.focal_distance / glm::dot(ray.dir, cam.view);
            glm::vec3 p_focus = ray.org + ray.dir * t;
            glm::vec2 p_lens = cam.lens_radius * sample_uniform_disk(rng);
            ray.org += cam.right * p_lens.x + cam.up * p_lens.y;
            ray.dir = glm::normalize(p_focus - ray.org);
        }

        segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.pixel_index = index;
        segment.remaining_bounces = trace_depth;
        segment.prev_bsdf_pdf = 0.f;
        segment.prev_was_delta = false;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 * Lo(p, wo) = Le(p, wo) + 1/n * sum(bsdf(p, wo, wi) * Li(p, wi) * absdot(wi,
 * nor) / pdf(wi))
 */
void pathtrace(uchar4* pbo, int iter) {
    const int trace_depth = host_scene->state.settings.max_depth;
    const CameraData& cam = host_scene->state.camera;
    const int num_pixels = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 block_size_2d(8, 8);
    const dim3 blocks_per_grid_2d((cam.resolution.x + block_size_2d.x - 1) / block_size_2d.x,
                                  (cam.resolution.y + block_size_2d.y - 1) / block_size_2d.y);

    // 1D block for path tracing
    const int block_size_1d = 128;

    gen_ray_from_cam<<<blocks_per_grid_2d, block_size_2d, 0, pt_stream>>>(cam, iter, trace_depth,
                                                                          dev_paths);
    check_cuda_error("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + num_pixels;
    int num_paths = dev_path_end - dev_paths;

    while (num_paths > 0 && depth < trace_depth) {
        // tracing
        dim3 num_blocks_path_segment_tracing = math_utils::divup(num_paths, block_size_1d);
        compute_intersections<<<num_blocks_path_segment_tracing, block_size_1d, 0, pt_stream>>>(
            num_paths, dev_paths, dev_geoms, host_scene->geoms.size(), dev_intersections,
            dev_isect_mat_ids);
        check_cuda_error("trace one bounce");

#if SORT_PATHS
        sort_paths(num_paths, dev_intersections, dev_isect_mat_ids, dev_paths, pt_stream);
#endif
        shade_material<<<num_blocks_path_segment_tracing, block_size_1d, 0, pt_stream>>>(
            iter, num_paths, depth, dev_intersections, dev_isect_mat_ids, dev_paths,
            dev_shadow_rays, dev_materials, light_sampler, dev_geoms, dev_image);
        check_cuda_error("shader material");

#if LI_MIS
        int num_srays = num_paths;
        dim3 num_blocks_srays = math_utils::divup(num_srays, block_size_1d);
        trace_shadow_rays<<<num_blocks_srays, block_size_1d, 0, pt_stream>>>(
            num_srays, host_scene->geoms.size(), dev_shadow_rays, dev_geoms, dev_image);
        check_cuda_error("trace shadow rays");
#endif

#if COMPACT_TERMINATED
#if SORT_PATHS
        // use binary search to filer down paths since we already did the sort
        num_paths = filter_missed(num_paths, dev_isect_mat_ids, pt_stream);
#endif
        // compact the remaning terminated paths
        num_paths = compact_terminated(num_paths, dev_paths, pt_stream);
#endif

        if (gui_data != NULL) {
            gui_data->traced_depth = depth;
        }
        depth++;
    }

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    send_image_to_pbo<<<blocks_per_grid_2d, block_size_2d, 0, pt_stream>>>(
        pbo, cam.resolution, iter, dev_image, host_scene->state.settings.agx);
    check_cuda_error("pathtrace");
}

void copy_image_to_host() {
    const CameraData& cam = host_scene->state.camera;
    const int num_pixels = cam.resolution.x * cam.resolution.y;
    cudaMemcpy(host_scene->state.image.data(), dev_image, num_pixels * sizeof(glm::vec3),
               cudaMemcpyDeviceToHost);
}
