#include "bvh.h"

__host__ __device__ void IntersectBVH(
	Ray& ray,
	BVHNode* bvhNodePool,
	const Triangle* triangles,
	const int* triPtrs,
	float& t_min,
	glm::vec3& intersectionPoint,
	glm::vec3& normal,
	bool& outside,
	int& hit_tri_index,
	float& u,
	float& v) {

	// chose DFS because queue size ~ depth of tree (2^64 is a lot of triangles) 
	int stackNodeIndices[64];
	int top = 0;
	stackNodeIndices[top++] = 0;

	float tmp_t = t_min;
	bool tmp_outside;
	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;
	float tmp_u;
	float tmp_v;

	while (top > 0) { // while stack is non empty

		// pop top node off stack 
		BVHNode& node = bvhNodePool[stackNodeIndices[--top]];

		if (IntersectAABB(ray, node.aabbMin, node.aabbMax, t_min)) { // if ray intersects bv...
			if (node.triCount != 0) { // if node is a leaf, test intersections

				for (int i = 0; i < node.triCount; i++) {
					// get pointer to triangle 
					int triIndex = triPtrs[node.leftFirst + i];
					const Triangle& tri = triangles[triIndex];

					// write info into tmp variables
					tmp_t = triangleIntersectionTest(tri, ray, tmp_intersect, tmp_normal, tmp_outside, tmp_u, tmp_v);

					// update if intersection is the closest currently found 
					if (tmp_t > 0.0f && t_min > tmp_t) {
						t_min = tmp_t;
						intersectionPoint = tmp_intersect;
						normal = tmp_normal;
						outside = tmp_outside;
						hit_tri_index = triIndex;
						u = tmp_u;
						v = tmp_v;
					}
				}

			}
			else { // push more nodes onto stack
				stackNodeIndices[top++] = node.leftFirst;
				stackNodeIndices[top++] = node.leftFirst + 1;
			}
		}
	}
}

__host__ __device__ bool IntersectAABB(
	const Ray& ray,
	const glm::vec3 bmin,
	const glm::vec3 bmax,
	float tminCurr) {
	// the t at which given ray intersections hit box's max x and min x
	float tx1 = (bmin.x - ray.origin.x) / ray.direction.x;
	float tx2 = (bmax.x - ray.origin.x) / ray.direction.x;
	// sort; which intersection occured first?
	float tmin = fminf(tx1, tx2);
	float tmax = fmaxf(tx1, tx2);
	// same for y 
	float ty1 = (bmin.y - ray.origin.y) / ray.direction.y;
	float ty2 = (bmax.y - ray.origin.y) / ray.direction.y;
	// time interval where common intersection occurs can only get smaller
	tmin = fmaxf(tmin, fminf(ty1, ty2));
	tmax = fminf(tmax, fmaxf(ty1, ty2));
	// same for z
	float tz1 = (bmin.z - ray.origin.z) / ray.direction.z;
	float tz2 = (bmax.z - ray.origin.z) / ray.direction.z;
	// keep shrinking interval 
	tmin = fmaxf(tmin, fminf(tz1, tz2));
	tmax = fminf(tmax, fmaxf(tz1, tz2));
	// if ray does not pass through area enclosed by all 3 axis-aligned slabs
	// recorded time interval will tell us 
	return tmax >= tmin && tmin < tminCurr && tmax > 0;
}
