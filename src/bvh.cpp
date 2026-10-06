#include "bvh.h"

void BVH::buildBVH(Scene* scene) {
	int numTris = scene->triangles.size();
	
	if (numTris == 0) return;

	int rootNodeIdx = 0; 
	nodesUsed = 1;

	bvhNodePool.resize(numTris * 2 - 1);
	triIdx.resize(numTris);

	for (int i = 0; i < numTris; ++i) triIdx[i] = i;

	// set up root node
	BVHNode& bvhRoot = bvhNodePool[rootNodeIdx];
	bvhRoot.leftFirst = 0;
	bvhRoot.triCount = numTris;

	// find bounds + keep splitting
	UpdateNodeBounds(scene, rootNodeIdx);
	Subdivide(scene, rootNodeIdx, 0);
}

// helpers 
glm::vec3 minVec3(glm::vec3 a, glm::vec3 b) {
	float minX = fminf(a[0], b[0]);
	float minY = fminf(a[1], b[1]);
	float minZ = fminf(a[2], b[2]);

	return glm::vec3(minX, minY, minZ);
}

glm::vec3 maxVec3(glm::vec3 a, glm::vec3 b) {
	float maxX = fmaxf(a[0], b[0]);
	float maxY = fmaxf(a[1], b[1]);
	float maxZ = fmaxf(a[2], b[2]);

	return glm::vec3(maxX, maxY, maxZ);
}

void BVH::UpdateNodeBounds(Scene* scene, int nodeIdx) {
	BVHNode& node = bvhNodePool[nodeIdx];
	node.aabbMin = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
	node.aabbMax = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	int first = node.leftFirst;

	for (int i = 0; i < node.triCount; ++i) {
		int leafTriIndex = triIdx[first + i];
		Triangle& leafTri = scene->triangles[leafTriIndex];

		// evaluate bounds using world space triangle positions 
		glm::vec3 worldPos0 = leafTri.positions[0];
		glm::vec3 worldPos1 = leafTri.positions[1];
		glm::vec3 worldPos2 = leafTri.positions[2];

		node.aabbMin = minVec3(node.aabbMin, worldPos0);
		node.aabbMin = minVec3(node.aabbMin, worldPos1);
		node.aabbMin = minVec3(node.aabbMin, worldPos2);
		node.aabbMax = maxVec3(node.aabbMax, worldPos0);
		node.aabbMax = maxVec3(node.aabbMax, worldPos1);
		node.aabbMax = maxVec3(node.aabbMax, worldPos2);
	}
}

void BVH::Subdivide(Scene* scene, int nodeIdx, int depth) {
	BVHNode& node = bvhNodePool[nodeIdx];
	if (node.triCount <= 2 || depth >= 48) return;  // cap the depth (stack overflow problem) 

	// bounds of the centroids, not of the triangles
	glm::vec3 cmin(FLT_MAX), cmax(-FLT_MAX);
	for (int k = 0; k < node.triCount; ++k) {
		const glm::vec3& c = scene->triangles[triIdx[node.leftFirst + k]].centroid;
		cmin = minVec3(cmin, c);
		cmax = maxVec3(cmax, c);
	}
	glm::vec3 extent = cmax - cmin;
	int axis = 0;
	if (extent.y > extent.x) axis = 1;
	if (extent.z > extent[axis]) axis = 2;
	if (extent[axis] <= 0.0f) return;   

	// midpoint of centroids
	float splitPos = cmin[axis] + extent[axis] * 0.5f;
	int i = node.leftFirst;
	int j = i + node.triCount - 1;
	while (i < j) {
		if (scene->triangles[triIdx[i]].centroid[axis] < splitPos) {
			i++;
		}
		else {
			std::swap(triIdx[i], triIdx[j--]);
		}
	}
	int leftCount = i - node.leftFirst;

	if (leftCount == 0 || leftCount == node.triCount) return;

	int leftChildIdx = nodesUsed++;
	int rightChildIdx = nodesUsed++;
	bvhNodePool[leftChildIdx].leftFirst = node.leftFirst;
	bvhNodePool[leftChildIdx].triCount = leftCount;
	bvhNodePool[rightChildIdx].leftFirst = i;
	bvhNodePool[rightChildIdx].triCount = node.triCount - leftCount;

	node.leftFirst = leftChildIdx;
	node.triCount = 0;
	UpdateNodeBounds(scene, leftChildIdx);
	UpdateNodeBounds(scene, rightChildIdx);

	Subdivide(scene, leftChildIdx, depth + 1);
	Subdivide(scene, rightChildIdx, depth + 1);
}