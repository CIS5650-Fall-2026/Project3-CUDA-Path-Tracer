#include "vdb_usage.h"

#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/io/IO.h>

#include "../intersections.h"

using namespace glm;

__host__ __device__ inline nanovdb::Coord toCoord(vec3 p) {
    return nanovdb::Coord(floor(p.x), floor(p.y), floor(p.z));
}
__host__ __device__ inline nanovdb::Coord getCoord(vec3 ro, vec3 rd, float t) {
    vec3 p = ro + rd * t;
    return toCoord(p);
}

#define VDBTRAVERSEEPS 1e-4f

__host__ __device__ float sampleTemperature(vec3 pos, VolumeData& vd, const Volume& volume) { // Maybe can put grid pointers in volume itself, VolumeData only for current overall specs
    nanovdb::FloatGrid* grid = (nanovdb::FloatGrid*)vd.temperature;
    auto accessor = grid->getAccessor();

    auto coord = toCoord(
        grid->worldToIndex(vec3(volume.userInvTransform * vec4(pos, 1.0f)))
    );
    //accessor.isActive()
    return accessor.getValue(coord); // make sure if it's invalid pos it just returns 0 or something right
}

__host__ __device__  float vdbIntersectionTest(VolumeData& vd, const Volume& volume, Ray ray, thrust::default_random_engine& rng, float tMax) {
    // BBX
    vec2 ts = bbxIntersectionTest(ray, volume.invTransform * volume.userInvTransform);
    if(ts.x >= ts.y || ts.y < 0.0f) return -1.0f;
    tMax = min(tMax, ts.y);

    //
    nanovdb::FloatGrid* grid = (nanovdb::FloatGrid*)vd.density;
    auto accessor = grid->getAccessor();

    //
    /*nanovdb::FloatGrid* tempGrid = (nanovdb::FloatGrid*)vd.temperature;
    nanovdb::DefaultReadAccessor<float> tempAccessor = tempGrid ? tempGrid->getAccessor() : accessor;*/

    vec3 lro = grid->worldToIndex(vec3(volume.userInvTransform * vec4(ray.origin, 1.0f)));
    vec3 lrd = grid->worldToIndexDir(vec3(volume.userInvTransform * vec4(ray.direction, 0.0f)));

    thrust::uniform_real_distribution<float> u01(0, 1);

    float t = max(0.0f, ts.x);
    int iter = 0;
    nanovdb::Coord ijk;
    glm::vec3 currPnt;

    float scatterDensityMult = powf(vd.densityDecay, static_cast<float>(vd.bounceDepth));

    float overallDensityMult = scatterDensityMult * volume.extinctionMult * vd.extinctionMult;

    while(t < tMax) {
        bool uniformMode = false;

        currPnt = lro + lrd * t;

        // Find active lower node
        const nanovdb::NanoLower<float>* lowerNode = grid->tree().template get<nanovdb::GetLower<float>>(toCoord(currPnt));
        while ((lowerNode == nullptr || lowerNode->getMax() <= 0.0f) && t < tMax) {
            nanovdb::Coord coord = toCoord(currPnt);
            // Floor to nearest 8*16
            vec3 bbxMin = vec3(coord.x() & ~127, coord.y() & ~127, coord.z() & ~127);
            vec3 bbxMax = bbxMin + vec3(128);

            t += VDBTRAVERSEEPS + alignedBbxIntersectionTest(currPnt, lrd, bbxMin, bbxMax).y;
            currPnt = lro + lrd * t;

            lowerNode = grid->tree().template get<nanovdb::GetLower<float>>(toCoord(currPnt));
        }

        // Find leaf with a child or a uniform value
        const nanovdb::NanoLeaf<float>* leafNode = grid->tree().template get<nanovdb::GetLeaf<float>>(toCoord(currPnt));
        while (!uniformMode && (leafNode == nullptr || leafNode->getMax() <= 0.0f) && t < tMax) {
            nanovdb::Coord coord = toCoord(currPnt);
            // Floor to nearest 8
            vec3 bbxMin = vec3(coord.x() & ~7, coord.y() & ~7, coord.z() & ~7);
            vec3 bbxMax = bbxMin + vec3(8);

            t += VDBTRAVERSEEPS+alignedBbxIntersectionTest(currPnt, lrd, bbxMin, bbxMax).y;
            currPnt = lro + lrd * t;

            leafNode = grid->tree().template get<nanovdb::GetLeaf<float>>(toCoord(currPnt));
            if (lowerNode->valueMask().isOn(lowerNode->CoordToOffset(coord)))
                uniformMode = true;
        }

        if (t >= tMax) return -1.0f;

        ijk = getCoord(lro, lrd, t); // just toCoord(currPnt)
        float currMax = uniformMode ? accessor.getValue(ijk) : leafNode->getMax();
        float deltaT = -log(max(1e-5f, u01(rng))) / (overallDensityMult * currMax);

        // If the woodcock step goes past the max extinction bbx, go to box and continue with no events
        vec3 bbxMin = vec3(ijk.x() & ~7, ijk.y() & ~7, ijk.z() & ~7); // TODO: turn this bbx stuff into a inline func or smth
        vec3 bbxMax = bbxMin + vec3(8);
        float deltaTcap = alignedBbxIntersectionTest(currPnt, lrd, bbxMin, bbxMax).y;
        if (deltaT > deltaTcap) {
            t += deltaTcap + VDBTRAVERSEEPS;
            currPnt = lro + lrd * t; // unnecessary but accident waiting to happen
            continue;
        }
        t += deltaT;

        if(t >= tMax) {
            return -1.0f;
        }

        ijk = getCoord(lro, lrd, t);
        float extinction = uniformMode ? currMax : accessor.getValue(ijk);

        /*const vec2 densityBurnawayParams = vec2(0.0f, 4.4f);
        if (vd.temperature)
            extinction *= fmin(1.0f, exp(densityBurnawayParams.y * (tempAccessor.getValue(ijk) - densityBurnawayParams.x)));*/

        if(u01(rng) <= extinction / currMax) //todo: Lower density according to temperature sample.. im guessing here but maybe it's better to do it in currMax (coarser tho)
            return t;

        // Iter cap
        if (iter++ > 500)
            return t;
    }

    return -1.0f;
}

