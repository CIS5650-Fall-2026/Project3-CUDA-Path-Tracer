#include "scene_structs.h"
#include "thrust_utils.h"

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>

struct IsPathTerminated {
    __host__ __device__ bool operator()(const PathSegment& ps) const {
        return ps.remaining_bounces <= 0;
    }
};

struct IsShadowRayInvalid    {
    __host__ __device__ bool operator()(const ShadowRay& sray) const {
        return sray.pixel_index == -1;
    }
};

struct IsIsectHit {
    template <typename Tuple> __device__ bool operator()(const Tuple& t) const {
        const MatId mat_id = thrust::get<2>(t);
        return mat_id < UINT8_MAX;
    }
};

void sort_paths(int num_paths, ShadeableIntersection* isects, MatId* mat_ids, PathSegment* paths,
                cudaStream_t stream) {
    auto policy = thrust::cuda::par_nosync.on(stream);
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(isects, paths));
    thrust::sort_by_key(policy, mat_ids, mat_ids + num_paths, zip_begin);
}

int filter_missed(int num_paths, MatId* mat_ids, cudaStream_t stream) {
    auto policy = thrust::cuda::par_nosync.on(stream);
    auto new_end = thrust::lower_bound(policy, mat_ids, mat_ids + num_paths, UINT8_MAX);
    return new_end - mat_ids;
}

int compact_terminated(int num_paths, PathSegment* paths, cudaStream_t stream) {
    auto policy = thrust::cuda::par_nosync.on(stream);
    auto new_end = thrust::remove_if(policy, paths, paths + num_paths, IsPathTerminated{});
    return new_end - paths;
}
