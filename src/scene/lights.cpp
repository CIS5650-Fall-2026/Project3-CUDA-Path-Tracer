// The NEE light list: power shares per PBRT's power light sampler, one row per receiver mask.
// Distant lights and the environment count the flux through a disk as large as the scene's bounding sphere.
// Emissive textures weigh by their average color, as PBRT-v4 weighs an image area light.

#include "scene/scene.h"

#include "utilities.h"

#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <utility>

namespace
{
// Counterclockwise seen from outside, the same as the OptiX cube GAS.
glm::vec3 cubeCorner(int i)
{
    return glm::vec3((i & 1) ? 0.5f : -0.5f, (i & 2) ? 0.5f : -0.5f, (i & 4) ? 0.5f : -0.5f);
}

const int CUBE_TRIANGLES[12][3] = {
    {0, 4, 6}, {0, 6, 2},   // -x
    {1, 7, 5}, {1, 3, 7},   // +x
    {0, 5, 4}, {0, 1, 5},   // -y
    {2, 6, 7}, {2, 7, 3},   // +y
    {0, 2, 3}, {0, 3, 1},   // -z
    {4, 7, 6}, {4, 5, 7},   // +z
};

// Linear, as the texture unit reads it.
glm::vec3 averageColor(const Texture& texture, const TextureImage& image)
{
    float decode[256];
    for (int i = 0; i < 256; ++i)
    {
        const float v = i / 255.0f;
        decode[i] = !texture.srgb ? v : v <= 0.04045f ? v / 12.92f : powf((v + 0.055f) / 1.055f, 2.4f);
    }
    double sum[3] = { 0.0, 0.0, 0.0 };
    const size_t texels = (size_t)image.width * image.height;
    for (size_t i = 0; i < texels; ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            sum[c] += decode[image.rgba[4 * i + c]];
        }
    }
    return texels > 0 ? glm::vec3(sum[0], sum[1], sum[2]) / (float)texels : glm::vec3(1.0f);
}
}  // namespace

void Scene::buildLights()
{
    // NEE never samples emissive spheres, so one sharing a material with a cube or mesh gets its own copy.
    std::vector<bool> onTriangles(materials.size(), false);
    for (const Geom& g : geoms)
    {
        if (g.type != GeomType::SPHERE)
        {
            onTriangles[g.materialId] = true;
        }
    }
    std::unordered_map<int, int> sphereCopies;
    for (Geom& g : geoms)
    {
        if (g.type != GeomType::SPHERE || maxComponent(materials[g.materialId].emission) <= 0.0f
            || !onTriangles[g.materialId])
        {
            continue;
        }
        auto found = sphereCopies.find(g.materialId);
        if (found == sphereCopies.end())
        {
            const Material copy = materials[g.materialId];  // a copy first: push_back may reallocate
            materials.push_back(copy);
            found = sphereCopies.emplace(g.materialId, (int)materials.size() - 1).first;
        }
        g.materialId = found->second;
    }

    std::vector<glm::vec3> emission(materials.size(), glm::vec3(0.0f));
    std::unordered_map<int, glm::vec3> textureAverages;
    for (size_t m = 0; m < materials.size(); ++m)
    {
        const Material& material = materials[m];
        emission[m] = material.emission;
        if (maxComponent(material.emission) <= 0.0f || material.emissiveTexture < 0)
        {
            continue;
        }
        auto found = textureAverages.find(material.emissiveTexture);
        if (found == textureAverages.end())
        {
            const Texture& texture = textures[material.emissiveTexture];
            found = textureAverages.emplace(material.emissiveTexture,
                averageColor(texture, textureImages[texture.image])).first;
        }
        emission[m] *= found->second;
    }

    lightTriangles.clear();
    std::vector<double> power;
    auto addTriangle = [&](const Geom& g, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec2 uv0, glm::vec2 uv1,
                           glm::vec2 uv2, bool mirrored) {
        p0 = glm::vec3(g.transform * glm::vec4(p0, 1.0f));
        p1 = glm::vec3(g.transform * glm::vec4(p1, 1.0f));
        p2 = glm::vec3(g.transform * glm::vec4(p2, 1.0f));
        // Keeps cross(e1, e2) the front face under a mirroring transform.
        if (mirrored)
        {
            std::swap(p1, p2);
            std::swap(uv1, uv2);
        }
        LightTriangle t;
        t.p0 = p0;
        t.e1 = p1 - p0;
        t.e2 = p2 - p0;
        t.uv0 = uv0;
        t.uv1 = uv1;
        t.uv2 = uv2;
        t.materialId = g.materialId;
        t.group = g.lightGroup;
        const double area = 0.5 * (double)glm::length(glm::cross(t.e1, t.e2));
        if (area > 0.0)
        {
            lightTriangles.push_back(t);
            // 2 pi: an emissive triangle emits from both sides.
            power.push_back(2.0 * PI * area * luminance(emission[g.materialId]));
        }
    };
    for (const Geom& g : geoms)
    {
        if (g.type == GeomType::SPHERE || luminance(emission[g.materialId]) <= 0.0f)
        {
            continue;
        }
        const bool mirrored = glm::determinant(glm::mat3(g.transform)) < 0.0f;
        if (g.type == GeomType::CUBE)
        {
            for (const auto& tri : CUBE_TRIANGLES)
            {
                addTriangle(g, cubeCorner(tri[0]), cubeCorner(tri[1]), cubeCorner(tri[2]), glm::vec2(0.0f),
                    glm::vec2(0.0f), glm::vec2(0.0f), mirrored);
            }
        }
        else
        {
            const TriangleMesh& mesh = meshes[g.meshId];
            for (int i = 0; i < mesh.triCount; ++i)
            {
                const glm::ivec3 tri = indices[mesh.indexOffset + i];
                addTriangle(g, positions[tri.x], positions[tri.y], positions[tri.z], uvs[tri.x], uvs[tri.y],
                    uvs[tri.z], mirrored);
            }
        }
    }

    glm::vec3 lo;
    glm::vec3 hi;
    bounds(0, geoms.size(), lo, hi);
    const double radius = lo.x <= hi.x ? 0.5 * (double)glm::length(hi - lo) : 1.0;
    int pointLights = 0;
    int spotLights = 0;
    for (const PunctualLight& light : punctualLights)
    {
        if (light.type == LIGHT_POINT)
        {
            ++pointLights;
            power.push_back(4.0 * PI * luminance(light.intensity));
        }
        else if (light.type == LIGHT_SPOT)
        {
            ++spotLights;
            // The smoothstep falloff integrates to half over the band between the cones.
            const double solidAngle = 2.0 * PI * (1.0 - 0.5 * ((double)light.cosInner + light.cosOuter));
            power.push_back(solidAngle * luminance(light.intensity));
        }
        else
        {
            power.push_back(PI * radius * radius * luminance(light.intensity));
        }
    }

    // The environment goes last.
    const bool hasEnvironment = environmentLight && environment.radianceIntegral > 0.0f;
    if (hasEnvironment)
    {
        power.push_back(PI * radius * radius * environment.radianceIntegral);
    }

    double total = 0.0;
    for (double p : power)
    {
        total += p;
    }
    lightMasks.clear();
    lightCdf.clear();
    punctualPickPdf.clear();
    environmentPickPdf.clear();
    emitterAreaPdf.clear();
    numLights = 0;
    if (total <= 0.0)
    {
        lightTriangles.clear();
        punctualLights.clear();
        return;
    }

    // Masks past the limit share row 0, which only wastes picks.
    const size_t MAX_LIGHT_MASKS = 64;
    lightMasks.push_back(0xffffffffu);
    size_t unlistedMasks = 0;
    for (const Geom& g : geoms)
    {
        if (std::find(lightMasks.begin(), lightMasks.end(), g.lightMask) != lightMasks.end())
        {
            continue;
        }
        if (lightMasks.size() < MAX_LIGHT_MASKS)
        {
            lightMasks.push_back(g.lightMask);
        }
        else
        {
            ++unlistedMasks;
        }
    }
    if (unlistedMasks > 0)
    {
        fprintf(stderr, "Light linking: %zu receiver masks past the limit of %zu sample every light\n",
            unlistedMasks, MAX_LIGHT_MASKS);
    }

    const size_t numMasks = lightMasks.size();
    const size_t numPunctual = punctualLights.size();
    numLights = (int)power.size();
    lightCdf.assign(numMasks * numLights, 1.0f);
    emitterAreaPdf.assign(numMasks * materials.size(), 0.0f);
    punctualPickPdf.assign(numMasks * numPunctual, 0.0f);
    environmentPickPdf.assign(numMasks, 0.0f);
    const auto receives = [](unsigned int mask, int group) { return ((mask >> group) & 1u) != 0u; };
    for (size_t m = 0; m < numMasks; ++m)
    {
        const unsigned int mask = lightMasks[m];
        // The environment is never linked.
        const auto share = [&](size_t i) {
            if (i < lightTriangles.size())
            {
                return receives(mask, lightTriangles[i].group) ? power[i] : 0.0;
            }
            if (i < lightTriangles.size() + numPunctual)
            {
                return receives(mask, punctualLights[i - lightTriangles.size()].group) ? power[i] : 0.0;
            }
            return power[i];
        };
        double maskTotal = 0.0;
        for (size_t i = 0; i < power.size(); ++i)
        {
            maskTotal += share(i);
        }
        if (maskTotal <= 0.0)
        {
            // The row picks light 0 every time, and sampleLight rejects it as not received.
            continue;
        }

        // Double: with millions of emitters each share is near float's resolution next to 1.
        float* cdf = lightCdf.data() + m * numLights;
        double running = 0.0;
        for (size_t i = 0; i < power.size(); ++i)
        {
            running += share(i);
            cdf[i] = (float)(running / maskTotal);
        }
        cdf[numLights - 1] = 1.0f;

        for (const LightTriangle& t : lightTriangles)
        {
            if (receives(mask, t.group))
            {
                emitterAreaPdf[m * materials.size() + t.materialId] =
                    (float)(2.0 * PI * luminance(emission[t.materialId]) / maskTotal);
            }
        }
        for (size_t i = 0; i < numPunctual; ++i)
        {
            if (receives(mask, punctualLights[i].group))
            {
                punctualPickPdf[m * numPunctual + i] = (float)(power[lightTriangles.size() + i] / maskTotal);
            }
        }
        if (hasEnvironment)
        {
            environmentPickPdf[m] = (float)(power.back() / maskTotal);
        }
    }

    printf("Lights for next event estimation: %zu emissive triangles, %d point, %d spot, %zu distant%s",
        lightTriangles.size(), pointLights, spotLights, punctualLights.size() - pointLights - spotLights,
        hasEnvironment ? ", the environment" : "");
    if (numMasks > 1)
    {
        printf("; %zu receiver masks (light linking)", numMasks);
    }
    printf("\n");
    if (hasEnvironment)
    {
        printf("  the environment gets %.1f%% of the light picks\n", 100.0 * environmentPickPdf[0]);
    }
}
