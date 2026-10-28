#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

// One texel by integer pixel coordinates, wrapping around the image on both axes
__host__ __device__ glm::vec3 texelAt(
    const glm::vec3* texels,
    const TextureInfo& tex,
    int x,
    int y)
{
    // % keeps the sign of the left operand, so add the size back before taking it again
    x = ((x % tex.width) + tex.width) % tex.width;
    y = ((y % tex.height) + tex.height) % tex.height;
    return texels[tex.offset + y * tex.width + x];
}

__host__ __device__ glm::vec3 sampleTexture(
    const glm::vec3* texels,
    const TextureInfo& tex,
    glm::vec2 uv)
{
    // glTF puts v = 0 at the top of the image, same as stb's row 0, so no flip
    float u = uv.x;
    float v = uv.y;

    // keep the fractional part so tiling UVs wrap and uv == 1 stays inside the image
    u = u - floor(u);
    v = v - floor(v);

#if TEXTURE_BILINEAR
    // texel centers sit at half-integer pixel coordinates, so shift by half a texel
    // to make "exactly on a center" come out as a whole number
    float fx = u * tex.width - 0.5f;
    float fy = v * tex.height - 0.5f;

    // the texel up and to the left of the sample point; floor, not a cast, because
    // fx can be negative along the left and top edges
    int x0 = (int)floor(fx);
    int y0 = (int)floor(fy);

    // how far the sample sits from that texel toward the next one, 0 to 1
    float tx = fx - x0;
    float ty = fy - y0;

    glm::vec3 topL = texelAt(texels, tex, x0, y0);
    glm::vec3 topR = texelAt(texels, tex, x0 + 1, y0);
    glm::vec3 lowL = texelAt(texels, tex, x0, y0 + 1);
    glm::vec3 lowR = texelAt(texels, tex, x0 + 1, y0 + 1);

    // blend left to right on both rows, then top to bottom
    glm::vec3 top = glm::mix(topL, topR, tx);
    glm::vec3 low = glm::mix(lowL, lowR, tx);
    return glm::mix(top, low, ty);
#else
    // nearest texel: scale to pixel coordinates, then index this image's slice
    return texelAt(texels, tex, (int)floor(u * tex.width), (int)floor(v * tex.height));
#endif
}

__host__ __device__ glm::vec3 proceduralTiles(
    const Material& m,
    glm::vec2 uv)
{
    // every integer cell of the scaled uv is one tile; the fractional part is the
    // position inside the current tile, 0 to 1 on both axes
    glm::vec2 cell = uv * m.tiles.count;
    cell = cell - glm::floor(cell);

    // distance to the nearest tile edge on each axis, 0 on an edge and 0.5 at the center
    float du = glm::min(cell.x, 1.0f - cell.x);
    float dv = glm::min(cell.y, 1.0f - cell.y);

    // a grout line straddles two tiles, so each tile owns half of its width
    glm::vec2 halfGrout = 0.5f * m.tiles.grout;
    if (du < halfGrout.x || dv < halfGrout.y) return m.tiles.groutColor;
    return m.color;
}

__host__ __device__ float proceduralTilesHeight(
    const Material& m,
    glm::vec2 uv)
{
    // same first two steps as proceduralTiles: position inside the tile, then
    // distance to the nearest edge on each axis
    glm::vec2 cell = uv * m.tiles.count;
    cell = cell - glm::floor(cell);
    float du = glm::min(cell.x, 1.0f - cell.x);
    float dv = glm::min(cell.y, 1.0f - cell.y);

    // flat at 0 inside the grout, flat at 1 on the tile, a smooth ramp half a grout
    // wide between them; a hard step would have no slope for bumpNormal to measure
    // each axis has its own grout width, and the lower of the two heights wins
    glm::vec2 halfGrout = 0.5f * m.tiles.grout;
    float hu = glm::smoothstep(halfGrout.x, 2.0f * halfGrout.x, du);
    float hv = glm::smoothstep(halfGrout.y, 2.0f * halfGrout.y, dv);
    return glm::min(hu, hv);
}

__host__ __device__ glm::vec3 bumpNormal(
    glm::vec3 normal,
    glm::vec3 tangent,
    const Material& m,
    glm::vec2 uv)
{
    // step in uv used to measure the slope; much smaller and u + step rounds back to u
    const float uvStep = 0.001f;

    // the interpolated normal and the tangent are not exactly perpendicular, so
    // remove the part of the tangent that lies along the normal
    glm::vec3 T = glm::normalize(tangent - glm::dot(tangent, normal) * normal);
    // direction of increasing v; v runs down the image, so this is T x N
    glm::vec3 B = glm::cross(T, normal);

    // slope of the height along u and along v by finite differences
    float h = proceduralTilesHeight(m, uv);
    float hu = (proceduralTilesHeight(m, glm::vec2(uv.x + uvStep, uv.y)) - h) / uvStep;
    float hv = (proceduralTilesHeight(m, glm::vec2(uv.x, uv.y + uvStep)) - h) / uvStep;

    // tilt the normal away from the uphill direction
    return glm::normalize(normal - m.surface.bump * (hu * T + hv * B));
}

__host__ __device__ void sampleBoxLight(
    const Geom& light,
    thrust::default_random_engine& rng,
    glm::vec3& point,
    glm::vec3& normal,
    float& area)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    // areas of the three pairs of faces; rotation does not change an area, so the
    // scale alone gives them
    glm::vec3 s = glm::abs(light.scale);
    float ax = s.y * s.z;   // the two faces across x
    float ay = s.x * s.z;
    float az = s.x * s.y;
    area = 2.0f * (ax + ay + az);

    // pick the axis in proportion to its faces' area, then one of its two faces
    float pick = u01(rng) * (ax + ay + az);
    int axis = pick < ax ? 0 : (pick < ax + ay ? 1 : 2);
    float side = u01(rng) < 0.5f ? -0.5f : 0.5f;

    // uniform point on that face of the unit cube, which spans -0.5 to 0.5
    glm::vec3 p(u01(rng) - 0.5f, u01(rng) - 0.5f, u01(rng) - 0.5f);
    p[axis] = side;
    glm::vec3 n(0.0f);
    n[axis] = side > 0.0f ? 1.0f : -1.0f;

    point = glm::vec3(light.transform * glm::vec4(p, 1.0f));
    normal = glm::normalize(glm::vec3(light.invTranspose * glm::vec4(n, 0.0f)));
}

// Schlick approximation of the Fresnel reflectance for a dielectric
// cosTheta is the cosine between the incoming ray and the normal facing it
__host__ __device__ float schlickFresnel(float cosTheta, float ior)
{
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosTheta, 5.0f);
}

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec3 geomNormal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    glm::vec3 newDir;
    // `normal` may be bumped; `geomNormal` is the real surface, turned to face the ray
    glm::vec3 incoming = pathSegment.ray.direction;
    if (glm::dot(incoming, geomNormal) > 0) geomNormal = -geomNormal;

    if (m.type == SPECULAR) {
        // Perfect specular: mirror the incoming ray about the surface normal
        newDir = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.color;

        if (m.exponent > 0.0f) {
            // Imperfect specular: a random direction around the mirror direction, packed
            // tighter as the exponent grows (GPU Gems 3, chapter 20, equations 7 to 9)
            thrust::uniform_real_distribution<float> u01(0, 1);
            float xi1 = u01(rng);
            float xi2 = u01(rng);

            // theta is the angle away from the mirror direction, phi goes around it
            float theta = glm::acos(glm::pow(xi1, 1 / (1 + m.exponent)));
            float phi = 2 * PI * xi2;

            // the direction in a frame whose z axis is the mirror direction
            glm::vec3 local(glm::cos(phi) * glm::sin(theta), glm::sin(phi) * glm::sin(theta), glm::cos(theta));

            // two axes perpendicular to the mirror direction; the helper axis is the
            // world axis least aligned with it, so the cross product cannot collapse
            glm::vec3 w = newDir;
            glm::vec3 helper = glm::abs(w.x) < 0.577f ? glm::vec3(1, 0, 0)
                             : glm::abs(w.y) < 0.577f ? glm::vec3(0, 1, 0)
                             : glm::vec3(0, 0, 1);
            glm::vec3 u = glm::normalize(glm::cross(w, helper));
            glm::vec3 v = glm::cross(w, u);

            // out of that frame, into world space
            newDir = local.x * u + local.y * v + local.z * w;
        }
    }
    else if (m.type == DIELECTRIC) {
        // Dielectric (glass, water): reflect or refract, chosen by the Fresnel weight
        glm::vec3 dir = pathSegment.ray.direction;
        glm::vec3 n = normal;

        // entering when the ray travels against the outward normal;
        // when leaving, flip n so it faces the ray for the formulas below
        bool entering = dot(dir, n) < 0;
        if (!entering) n = -n;

        // eta = n1 / n2 as glm::refract expects (air -> glass on entry, glass -> air on exit)
        float eta = entering ? 1 / m.ior : m.ior;

        float cosI = -glm::dot(dir, n);
        float R = schlickFresnel(cosI, m.ior);

        // Snell discriminant; glm 0.9 refract returns NaN (not zero) on total internal
        // reflection, so the test has to happen here
        float k = 1 - (eta * eta) * (1 - cosI * cosI);

        thrust::uniform_real_distribution<float> u01(0, 1);
        if (k < 0 || u01(rng) <  R) {
            newDir = glm::reflect(dir, n);
        }
        else {
            newDir = glm::refract(dir, n, eta);
        }
        // no division by the pick probability: choosing reflection with
        // probability R and weighting it by R cancels to 1
        pathSegment.color *= m.color;
    }
    else {
        // Ideal diffuse: cosine-weighted sampling makes bsdf * cos / pdf
        // collapse to just the albedo (cos and pi terms cancel)
        // normals are geometric (outward), so face the ray if hit from inside
        if (glm::dot(pathSegment.ray.direction, normal) > 0) normal = -normal;
        newDir = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= m.color;
    }

    // a bumped normal tilts the sampling hemisphere, so part of it dips under the
    // real surface; mirror those directions back above it instead of losing them
    // Glass is exempt, a refracted ray is supposed to cross the surface
    if (m.type != DIELECTRIC && glm::dot(newDir, geomNormal) < 0) {
        newDir = glm::reflect(newDir, geomNormal);
    }

    pathSegment.ray.direction = normalize(newDir);
    pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction; // Offset origin off the surface to avoid self-intersection (shadow acne)
}
