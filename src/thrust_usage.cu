#include "thrust_usage.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/remove.h>

struct compare_by_material_id
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& inter1, const ShadeableIntersection& inter2) {
        return inter1.materialId < inter2.materialId;
    }
};

void sortIntersections(ShadeableIntersection* start, ShadeableIntersection* end) {
    return thrust::sort(thrust::device, start, end, compare_by_material_id());
}

struct should_remove_path
{
    __host__ __device__
        bool operator()(PathSegment p) {
        return p.remainingBounces == 0;
    }
};

PathSegment* removeUnusedPaths(PathSegment* start, PathSegment* end) {
    return thrust::remove_if(thrust::device, start, end, should_remove_path());
}