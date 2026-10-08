#include "bvh.h"

void BVHBuilder::buildBLAS(std::vector<TriangleBVHBuildData>& triangleData, int start, int end, std::vector<BLASNode>& outNodes, const std::vector<Triangle>& inTriangles, std::vector<Triangle>& outTriangles, int maxDepth)
{
	if (triangleData.empty()) {
		return;
	}

	int rootNodeIndex = outNodes.size();
	outNodes.push_back(BLASNode{}); // Placeholder for root node

	buildBLASInternal(triangleData, start, end, rootNodeIndex, outNodes, inTriangles, outTriangles, 0, maxDepth);
}

void BVHBuilder::buildTLAS(std::vector<GeometryBVHBuildData>& geometryData, int start, int end, std::vector<TLASNode>& outNodes, int maxDepth)
{
	if (geometryData.empty()) {
		return;
	}

	int rootNodeIndex = outNodes.size();
	outNodes.push_back(TLASNode{}); // Placeholder for root node

	buildTLASInternal(geometryData, start, end, rootNodeIndex, outNodes, 0, maxDepth);
}

void BVHBuilder::buildBLASInternal(
	std::vector<TriangleBVHBuildData>& triangleData, 
	int start, int end, 
	int currentNodeIndex, 
	std::vector<BLASNode>& outNodes,
	const std::vector<Triangle>& inTriangles, 
	std::vector<Triangle>& outTriangles, 
	int depth, int maxDepth)
{
	int count = end - start;

	AABB nodeBounds;
	AABB centroidBounds;
	for (int i = start; i < end; i++) {
		nodeBounds.expand(triangleData[i].bounds);
		centroidBounds.expand(triangleData[i].centroid);
	}

	// Create leaf node if count is small or max depth reached
	if (count <= 4 || depth >= maxDepth) {
		outNodes[currentNodeIndex].aabbMin = nodeBounds.min;
		outNodes[currentNodeIndex].aabbMax = nodeBounds.max;
		outNodes[currentNodeIndex].leftChild = -1; // -1 tags this explicitly as a leaf
		outNodes[currentNodeIndex].triangleOffset = static_cast<int>(outTriangles.size());
		outNodes[currentNodeIndex].triangleCount = count;

		for (int i = start; i < end; i++) {
			outTriangles.push_back(inTriangles[triangleData[i].triangleIndex]);
		}
		return;
	}
	
	// Split on longest axis
	glm::vec3 extent = centroidBounds.max - centroidBounds.min;
	int axis = 0;
	if (extent.y > extent.x && extent.y > extent.z) {
		axis = 1;
	}
	else if (extent.z > extent.x && extent.z > extent.y) {
		axis = 2;
	}

	float midpoint = 0.5f * (centroidBounds.min[axis] + centroidBounds.max[axis]);

	// Partition based on midpoint
	auto midPtr = std::partition(triangleData.begin() + start,
		triangleData.begin() + end,
		[axis, midpoint](const TriangleBVHBuildData& data) {
			return data.centroid[axis] < midpoint;
		});

	int mid = static_cast<int>(std::distance(triangleData.begin(), midPtr));

	// If partition failed, split in half
	if (mid == start || mid == end) {
		mid = start + count / 2;
	}

	int leftChildIndex = outNodes.size();
	// Make sure the left and right children are adjacent
	outNodes.push_back(BLASNode{}); // Placeholder for left child
	outNodes.push_back(BLASNode{}); // Placeholder for right child

	outNodes[currentNodeIndex].aabbMin = nodeBounds.min;
	outNodes[currentNodeIndex].aabbMax = nodeBounds.max;
	outNodes[currentNodeIndex].leftChild = leftChildIndex;
	outNodes[currentNodeIndex].triangleCount = 0; // Not a leaf node

	// Build left and right children
	buildBLASInternal(triangleData, start, mid, leftChildIndex, outNodes, inTriangles, outTriangles, depth + 1, maxDepth);
	buildBLASInternal(triangleData, mid, end, leftChildIndex + 1, outNodes, inTriangles, outTriangles, depth + 1, maxDepth);
}

void BVHBuilder::buildTLASInternal(
	std::vector<GeometryBVHBuildData>& geometryData, 
	int start, int end, 
	int currentNodeIndex, 
	std::vector<TLASNode>& outNodes, 
	int depth, int maxDepth)
{
	int count = end - start;

	AABB nodeBounds;
	AABB centroidBounds;
	for (int i = start; i < end; i++) {
		nodeBounds.expand(geometryData[i].bounds);
		centroidBounds.expand(geometryData[i].centroid);
	}

	// Create leaf node if count is small or max depth reached
	if (count <= 1) {
		outNodes[currentNodeIndex].aabbMin = nodeBounds.min;
		outNodes[currentNodeIndex].aabbMax = nodeBounds.max;
		outNodes[currentNodeIndex].leftChild = -1; // -1 tags this explicitly as a leaf
		outNodes[currentNodeIndex].geometryIndex = geometryData[start].geometryIndex;
		return;
	}

	// Split on longest axis
	glm::vec3 extent = centroidBounds.max - centroidBounds.min;
	int axis = 0;
	if (extent.y > extent.x && extent.y > extent.z) {
		axis = 1;
	}
	else if (extent.z > extent.x && extent.z > extent.y) {
		axis = 2;
	}

	float midpoint = 0.5f * (centroidBounds.min[axis] + centroidBounds.max[axis]);

	// Partition based on midpoint
	auto midPtr = std::partition(geometryData.begin() + start,
		geometryData.begin() + end,
		[axis, midpoint](const GeometryBVHBuildData& data) {
			return data.centroid[axis] < midpoint;
		});

	int mid = static_cast<int>(std::distance(geometryData.begin(), midPtr));

	// If partition failed, split in half
	if (mid == start || mid == end) {
		mid = start + count / 2;
	}

	int leftChildIndex = outNodes.size();
	// Make sure the left and right children are adjacent
	outNodes.push_back(TLASNode{}); // Placeholder for left child
	outNodes.push_back(TLASNode{}); // Placeholder for right child

	outNodes[currentNodeIndex].aabbMin = nodeBounds.min;
	outNodes[currentNodeIndex].aabbMax = nodeBounds.max;
	outNodes[currentNodeIndex].leftChild = leftChildIndex;
	outNodes[currentNodeIndex].geometryIndex = -1; // Not a leaf node

	// Build left and right children
	buildTLASInternal(geometryData, start, mid, leftChildIndex, outNodes, depth + 1, maxDepth);
	buildTLASInternal(geometryData, mid, end, leftChildIndex + 1, outNodes, depth + 1, maxDepth);
}

