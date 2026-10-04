#include "texturing.h"

#include "intersections.h"
#include "utilities.h"

namespace
{
    __device__ glm::vec3 fetchRGB(cudaTextureObject_t tex, glm::vec2 uv)
    {
        // sRGB textures get linearized when loading
        float4 c = tex2D<float4>(tex, uv.x, uv.y);
        return glm::vec3(c.x, c.y, c.z);
    }

    __device__ float fetchHeight(cudaTextureObject_t tex, glm::vec2 uv)
    {
        // Height maps are greyscale, so we can sample one value
        return tex2D<float4>(tex, uv.x, uv.y).x;
    }

    // Bump mapping for one parameterization, PBRT style. uv is already scaled by TEXTURE_SCALE, and dpdu/dpdv are
    // the object-space surface derivatives with respect to that scaled uv. Displacing the surface to
    // p' = p + strength * h(uv) * n gives dp'/du = dpdu + strength * dh/du * n (ignoring how n itself bends),
    // and the bumped normal is cross(dp'/du, dp'/dv).
    __device__ glm::vec3 bumpNormal(const Material& m, cudaTextureObject_t tex, glm::vec2 uv,
        glm::vec3 n, glm::vec3 dpdu, glm::vec3 dpdv)
    {
        // keep the frame in n's tangent plane, so cross(dpdu, dpdv) is parallel to n (n is interpolated, the
        // triangle's dpdu/dpdv lie in the flat face)
        dpdu -= n * glm::dot(n, dpdu);
        dpdv -= n * glm::dot(n, dpdv);
        // which way cross(dpdu, dpdv) faces depends on the UV winding, so remember it to orient the result
        float side = glm::dot(glm::cross(dpdu, dpdv), n) < 0.0f ? -1.0f : 1.0f;

        // central differences one texel apart
        glm::vec2 d = m.bumpTexelSize;
        float hu = (fetchHeight(tex, uv + glm::vec2(d.x, 0.0f)) - fetchHeight(tex, uv - glm::vec2(d.x, 0.0f))) / (2.0f * d.x);
        float hv = (fetchHeight(tex, uv + glm::vec2(0.0f, d.y)) - fetchHeight(tex, uv - glm::vec2(0.0f, d.y))) / (2.0f * d.y);

        glm::vec3 nb = glm::cross(dpdu + m.bumpStrength * hu * n, dpdv + m.bumpStrength * hv * n);
        float len2 = glm::dot(nb, nb);
        if (!(len2 > 1e-30f)) return n;   // degenerate frame (also catches NaN)
        return side * nb / sqrtf(len2);
    }

    // UV mapping from the hit triangle. Returns false if the triangle's UVs are degenerate.
    __device__ bool uvMapping(const Material& m, const Triangle& tri, const cudaTextureObject_t* textures,
        glm::vec3 p, glm::vec3 n, glm::vec3& albedo, glm::vec3& nOut)
    {
        // barycentrics of p (projected onto the triangle's plane, p is a hair off it)
        glm::vec3 e1 = tri.v1 - tri.v0;
        glm::vec3 e2 = tri.v2 - tri.v0;
        glm::vec3 ep = p - tri.v0;
        float d00 = glm::dot(e1, e1), d01 = glm::dot(e1, e2), d11 = glm::dot(e2, e2);
        float d20 = glm::dot(ep, e1), d21 = glm::dot(ep, e2);
        float denom = d00 * d11 - d01 * d01;
        if (!(fabsf(denom) > 1e-30f)) return false;
        float b1 = (d11 * d20 - d01 * d21) / denom;
        float b2 = (d00 * d21 - d01 * d20) / denom;
        float b0 = 1.0f - b1 - b2;
        glm::vec2 uv = (b0 * tri.uv0 + b1 * tri.uv1 + b2 * tri.uv2) * m.texScale;

        // dp/du and dp/dv: solve e1 = dpdu * duv1.x + dpdv * duv1.y (and the same for e2)
        glm::vec2 duv1 = (tri.uv1 - tri.uv0) * m.texScale;
        glm::vec2 duv2 = (tri.uv2 - tri.uv0) * m.texScale;
        float det = duv1.x * duv2.y - duv1.y * duv2.x;
        if (!(fabsf(det) > 1e-20f)) return false;
        glm::vec3 dpdu = (e1 * duv2.y - e2 * duv1.y) / det;
        glm::vec3 dpdv = (e2 * duv1.x - e1 * duv2.x) / det;

        albedo = m.colorTex >= 0 ? fetchRGB(textures[m.colorTex], uv) : m.color;
        nOut = m.bumpTex >= 0 ? bumpNormal(m, textures[m.bumpTex], uv, n, dpdu, dpdv) : n;
        return true;
    }

    // Triplanar mapping: project the texture along object X, Y and Z and blend by how much the normal faces
    // each axis. Side projections put u horizontal and v down (image rows go down), so bricks stay upright.
    __device__ void triplanarMapping(const Material& m, const cudaTextureObject_t* textures,
        glm::vec3 p, glm::vec3 n, glm::vec3& albedo, glm::vec3& nOut)
    {
        glm::vec3 w = glm::abs(n);
        w *= w;
        w *= w;                          // ^4 narrows the blend band between projections
        w /= (w.x + w.y + w.z);

        const float s = m.texScale;
        const glm::vec2 uvs[3] = {
            glm::vec2(p.z, -p.y) * s,    // X faces
            glm::vec2(p.x, p.z) * s,     // Y faces (top/bottom)
            glm::vec2(p.x, -p.y) * s };  // Z faces
        const glm::vec3 dpdus[3] = { glm::vec3(0, 0, 1) / s, glm::vec3(1, 0, 0) / s, glm::vec3(1, 0, 0) / s };
        const glm::vec3 dpdvs[3] = { glm::vec3(0, -1, 0) / s, glm::vec3(0, 0, 1) / s, glm::vec3(0, -1, 0) / s };

        glm::vec3 color(0.0f);
        glm::vec3 nSum(0.0f);
        for (int a = 0; a < 3; ++a)
        {
            if (w[a] < 1e-3f) continue;  // saves 5 texture reads for projections that barely contribute
            if (m.colorTex >= 0) color += w[a] * fetchRGB(textures[m.colorTex], uvs[a]);
            nSum += w[a] * (m.bumpTex >= 0 ? bumpNormal(m, textures[m.bumpTex], uvs[a], n, dpdus[a], dpdvs[a]) : n);
        }
        albedo = m.colorTex >= 0 ? color : m.color;
        nOut = glm::normalize(nSum);
    }
}

__device__ void textureSurface(
    const Material& m,
    const Geom& geom,
    const Triangle* triangles,
    int triId,
    glm::vec3 worldPos,
    glm::vec3 worldNormal,
    const cudaTextureObject_t* textures,
    glm::vec3& albedo,
    glm::vec3& shadingNormal)
{
    albedo = m.color;
    shadingNormal = worldNormal;
    if (m.colorTex < 0 && m.bumpTex < 0) return;

    // work in object space. World normals are invTranspose(M) * n, so transpose(M) * nWorld undoes that (up to length).
    // these could also be called objPos and objNormal
    glm::vec3 p = multiplyMV(geom.inverseTransform, glm::vec4(worldPos, 1.0f));
    glm::vec3 n = glm::normalize(multiplyMV(glm::transpose(geom.transform), glm::vec4(worldNormal, 0.0f)));

    glm::vec3 nObj = n;
    // only use UVs if we are not doing triplanar, the object is a mesh with UVs, and we don't get -1 triId
    bool useUVs = !m.triplanar && geom.type == MESH && geom.hasUVs && triId >= 0;
    // if we don't use UVs or if mapping encountered issues
    // uvMapping also updates nObj
    if (!useUVs || !uvMapping(m, triangles[triId], textures, p, n, albedo, nObj))
    {
        triplanarMapping(m, textures, p, n, albedo, nObj);
    }

    // transform back to world for shading
    shadingNormal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(nObj, 0.0f)));
}
