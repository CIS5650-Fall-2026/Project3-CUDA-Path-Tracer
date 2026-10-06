#include "sceneStructs.h"
#include "scene.h"
#include "intersections.h"


class BVH {
private: 
	void UpdateNodeBounds(Scene* scene, int nodeIdx);
	void Subdivide(Scene* scene, int nodeIdx, int depth);

public: 
	int nodesUsed = 0;
	std::vector<BVHNode> bvhNodePool;
	std::vector<int> triIdx;

	void buildBVH(Scene* scene);
};

__host__ __device__ bool IntersectAABB(
	const Ray& ray, 
	const glm::vec3 bmin, 
	const glm::vec3 bmax, 
	float tminCurr);

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
	float& v);