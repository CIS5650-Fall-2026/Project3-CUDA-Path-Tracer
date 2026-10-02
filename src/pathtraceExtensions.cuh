#pragma once

static glm::vec3 *display = nullptr, *bloomA = nullptr, *bloomB = nullptr;
static Triangle *triangles = nullptr;
static PathSegment *pathsTemp = nullptr, *completedPaths = nullptr;
static ShadeableIntersection *hitsTemp = nullptr;
static int *keys = nullptr, *order = nullptr, *lights = nullptr, *primes = nullptr, *scratch = nullptr;
bool statistics = false;
TraceStats lastStats;
struct EventPair
{
    cudaEvent_t start{}, end{};
};
std::vector<EventPair> timerEvents;
void checked(cudaError_t result)
{
    if (result != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(result));
}
template <class T> void allocate(T *&p, size_t n)
{
    if (n)
        checked(cudaMalloc(&p, n * sizeof(T)));
}
template <class T> void upload(T *p, const std::vector<T> &v)
{
    if (!v.empty())
        checked(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <class T> void release(T *&p)
{
    if (p)
        cudaFree(p);
    p = nullptr;
}
struct Timer
{
    size_t cursor = 0;
    std::vector<float *> totals;
    void begin()
    {
        if (!statistics)
            return;
        if (cursor == timerEvents.size())
        {
            EventPair pair;
            checked(cudaEventCreate(&pair.start));
            checked(cudaEventCreate(&pair.end));
            timerEvents.push_back(pair);
        }
        checked(cudaEventRecord(timerEvents[cursor].start));
    }
    void stop(float &total)
    {
        if (statistics)
        {
            checked(cudaEventRecord(timerEvents[cursor++].end));
            totals.push_back(&total);
        }
    }
    void resolve()
    {
        if (totals.empty())
            return;
        checked(cudaEventSynchronize(timerEvents[cursor - 1].end));
        for (size_t i = 0; i < cursor; ++i)
        {
            float ms;
            checked(cudaEventElapsedTime(&ms, timerEvents[i].start, timerEvents[i].end));
            *totals[i] += ms;
        }
    }
};
struct Dead
{
    __host__ __device__ bool operator()(const PathSegment &p) const
    {
        return p.remainingBounces <= 0;
    }
};

__global__ void reorderPaths(int n, const int *order, const PathSegment *input, PathSegment *output,
                             const ShadeableIntersection *inputHits, ShadeableIntersection *outputHits)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int source = order[i];
        output[i] = input[source];
        outputHits[i] = inputHits[source];
    }
}



__global__ void shadePath(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int depth, RenderOptions o, const Geom* geoms, int geomCount,
    const Triangle* triangles, const int* lights, int lightCount,
    const int* bases, PathSegment* completedPaths)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        PathSegment& p = pathSegments[idx];
        if (p.remainingBounces <= 0) return;
        int iteration = iter;
        const auto &h = intersection;
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            // Set up the RNG
            // The sampler uses persistent pixel IDs and explicit dimensions
            // so reordering and restarting cannot change a pixel's sample stream.
            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;
            const Material &m = material;
            Sampler sampler{static_cast<unsigned>(p.pixelIndex), static_cast<unsigned>(iteration - 1), o.seed, o.halton,
                            bases};
            int dim = 8 + depth * 8;
            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f)
            {
                float weight = 1;
                if (o.directLighting && !p.previousDelta && lightCount > 0)
                {
                    glm::vec3 delta = h.point - p.previousPoint;
                    float distance2 = glm::dot(delta, delta);
                    float cosine = fabsf(glm::dot(h.outwardNormal, -p.ray.direction));
                    float pdf = cosine > 1e-8f
                                    ? lightAreaPdf(geoms[h.geomId], triangles, h.point, p.ray.time, h.triangleId) *
                                          distance2 / (cosine * lightCount)
                                    : 0;
                    weight = powerWeight(p.previousPdf, pdf);
                }
                p.radiance += p.color * materialColor * (material.emittance * weight);
                p.remainingBounces = 0;
            }
            else if (p.remainingBounces <= 1)
                p.remainingBounces = 0;
            else
            {
                // Sample direct lighting, scatter by material, and apply roulette.
                bool diffuse = m.hasReflective == 0 && !(m.hasRefractive > 0 && o.refraction);
                if (o.directLighting && diffuse && lightCount > 0)
                {
                    int lightId = lights[int(sampler.at(dim) * lightCount)];
                    const Geom &light = geoms[lightId];
                    LightSample ls = sampleLight(light, triangles, p.ray.time, sampler.at(dim + 1), sampler.at(dim + 2),
                                                 sampler.at(dim + 3));
                    glm::vec3 delta = ls.point - h.point;
                    float distance2 = glm::dot(delta, delta);
                    if (distance2 > rayEpsilon * rayEpsilon)
                    {
                        glm::vec3 wi = delta / sqrtf(distance2);
                        float cosine = fmaxf(0, glm::dot(h.surfaceNormal, wi));
                        float lightCos = fabsf(glm::dot(ls.normal, -wi));
                        if (cosine > 0 && lightCos > 1e-8f)
                        {
                            float pdf = ls.areaPdf * distance2 / (lightCos * lightCount), bsdfPdf = cosine / PI;
                            Ray shadow;
                            shadow.origin = h.point + h.surfaceNormal * rayEpsilon;
                            shadow.time = p.ray.time;
                            glm::vec3 toLight = ls.point - shadow.origin;
                            float distance = glm::length(toLight);
                            shadow.direction = toLight / distance;
                            if (traceScene(shadow, geoms, geomCount, triangles, o.meshCulling, distance - 2 * rayEpsilon)
                                    .t < 0)
                            {
                                const auto &lm = materials[light.materialid];
                                p.radiance += p.color * m.color * lm.color *
                                              (lm.emittance * bsdfPdf * powerWeight(pdf, bsdfPdf) / pdf);
                            }
                        }
                    }
                }
                scatterRay(p, h.point, h.surfaceNormal, h.frontFace, m, o.refraction, sampler.at(dim + 4),
                               sampler.at(dim + 5));
                --p.remainingBounces;
                if (o.russianRoulette && depth + 1 >= o.rouletteDepth)
                {
                    glm::vec3 rr = p.color * p.etaScale;
                    float survival = glm::clamp(fmaxf(rr.x, fmaxf(rr.y, rr.z)), 0.05f, 0.95f);
                    if (sampler.at(dim + 6) >= survival)
                        p.remainingBounces = 0;
                    else
                        p.color /= survival;
                }
                if (fmaxf(p.color.x, fmaxf(p.color.y, p.color.z)) <= 0)
                    p.remainingBounces = 0;
            }
        }
        else {
            // If there was no intersection, color the ray black.
            // Preserve radiance already collected by earlier direct-light samples.
            pathSegments[idx].color = glm::vec3(0.0f);
            p.remainingBounces = 0;
        }
        // Save each terminated path before compaction so finalGather retains its radiance.
        if (p.remainingBounces <= 0)
        {
            completedPaths[p.pixelIndex] = p;
            // Live-path color is throughput; the gathered output color is radiance.
            completedPaths[p.pixelIndex].color = p.radiance;
        }
    }
}

__global__ void brightPass(int n, const glm::vec3 *image, glm::vec3 *out, int iteration, float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = glm::max(image[i] / float(iteration) - glm::vec3(threshold), glm::vec3(0));
}
__global__ void blur(int n, int width, int height, const glm::vec3 *input, glm::vec3 *output, int radius,
                     bool horizontal)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int x = i % width, y = i / width;
    glm::vec3 sum(0);
    float weights = 0, sigma = fmaxf(1, radius * 0.5f);
    for (int d = -radius; d <= radius; ++d)
    {
        int sx = horizontal ? max(0, min(width - 1, x + d)) : x, sy = horizontal ? y : max(0, min(height - 1, y + d));
        float w = expf(-float(d * d) / (2 * sigma * sigma));
        sum += w * input[sy * width + sx];
        weights += w;
    }
    output[i] = sum / weights;
}
__global__ void displayPass(int n, const glm::vec3 *image, const glm::vec3 *bloom, glm::vec3 *display,
                            int iteration, RenderOptions o)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    glm::vec3 value = glm::max(image[i] / float(iteration), glm::vec3(0));
    if (o.postprocess)
    {
        value = (value + o.bloomStrength * bloom[i]) * exp2f(o.exposure);
        value = value / (glm::vec3(1) + value);
        value = glm::pow(value, glm::vec3(1 / o.gamma));
    }
    display[i] = value;

}

void pathtraceEnableStats(bool enabled)
{
    statistics = enabled;
}
const TraceStats &pathtraceLastStats()
{
    return lastStats;
}
void pathtraceRefresh(uchar4 *pbo, int iteration)
{
    if (!hst_scene || iteration <= 0)
        return;
    auto &state = hst_scene->state;
    int n = int(state.image.size()), blocks = (n + 255) / 256;
    auto o = state.options;
    if (o.postprocess)
    {
        brightPass<<<blocks, 256>>>(n, dev_image, bloomA, iteration, o.bloomThreshold);
        blur<<<blocks, 256>>>(n, state.camera.resolution.x, state.camera.resolution.y, bloomA, bloomB, o.bloomRadius,
                              true);
        blur<<<blocks, 256>>>(n, state.camera.resolution.x, state.camera.resolution.y, bloomB, bloomA, o.bloomRadius,
                              false);
    }
    displayPass<<<blocks, 256>>>(n, dev_image, bloomA, display, iteration, o);
    if (pbo)
    {
        const dim3 blockSize2d(8, 8);
        const dim3 blocksPerGrid2d((state.camera.resolution.x + 7) / 8, (state.camera.resolution.y + 7) / 8);
        sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, state.camera.resolution, 1, display);
    }
    checked(cudaGetLastError());
    checked(cudaMemcpy(state.displayImage.data(), display, n * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
}
