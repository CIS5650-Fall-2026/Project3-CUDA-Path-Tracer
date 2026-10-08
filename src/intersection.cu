#include "intersection.h"

__device__ IntersectionData IntersectionStatics::intersectGeometry(
	const Ray& ray, 
	const Geom& geometry, 
	const BLASNode* dev_blasNodes, 
	const Triangle* dev_triangles, 
	int lightCount)
{
	// Convert world space -> object space
	Ray objectSpaceRay = ray.transform(geometry.inverseTransform);

	// Run intersection
	IntersectionData result;
	if (geometry.type == GeomType::SPHERE) {
		result = intersectSphere(objectSpaceRay);
	}
	else if (geometry.type == GeomType::CUBE) {
		result = intersectBox(objectSpaceRay);
	}
	else if (geometry.type == GeomType::MESH) {
		result = intersectMesh(objectSpaceRay, geometry, dev_blasNodes, dev_triangles);
	}

	if (result.t <= 0.0f) {
		return result;
	}

	result.materialIndex = geometry.materialid;

	// Convert normal to world space
	glm::vec3 worldNormal = glm::vec3(geometry.invTranspose * glm::vec4(result.normal, 0.0f));
	float normalScale = glm::length(worldNormal);
	result.normal = MathHelpers::safeNormalize(glm::vec3(worldNormal));

	float det = glm::abs(glm::determinant(geometry.transform));
	float jacobian = det * normalScale;
	if (jacobian <= 0.0f) {
		result.pdfIfLight = 0.0f;
	}
	else {
		result.pdfIfLight = result.pdfIfLight / jacobian;
	}

	/// Divide pdfIfLight by number of lights to match NEE sampling
	if (lightCount > 0) {
		result.pdfIfLight = result.pdfIfLight / lightCount;
	}
	else {
		result.pdfIfLight = 0.0f;
	}

	// Convert pdf if light to solid angle pdf
	float distanceSquared = result.t * result.t;
	float cosTheta = glm::abs(glm::dot(result.normal, -ray.direction));
	if (cosTheta <= 0.0001f) {
		result.pdfIfLight = 0.0f;
	}
	else {
		result.pdfIfLight = result.pdfIfLight * (distanceSquared / cosTheta);
	}

	// Convert t to world space
	glm::vec3 worldPosition = glm::vec3(geometry.transform * glm::vec4(objectSpaceRay.getPositionAtTime(result.t), 1.0f));
	result.t = glm::length(worldPosition - ray.origin);

	return result;
}

__device__ IntersectionData IntersectionStatics::intersectSphere(const Ray& ray) {
	IntersectionData result = IntersectionData{};

	float t0;
	float t1;

	glm::vec3 toRay = ray.origin;
	float a = glm::dot(ray.direction, ray.direction);
	float b = 2 * glm::dot(ray.direction, toRay);
	float c = glm::dot(toRay, toRay) - 0.5f * 0.5f;
	if (!MathHelpers::solveQuadratic(a, b, c, t0, t1)) {
		return result;
	}
	if (t0 > t1) {
		float temp = t1;
		t1 = t0;
		t0 = temp;
	}

	if (t0 < 0.0f) {
		t0 = t1;
		if (t0 < 0.0f) {
			return result;
		}
		result.bInside = true;
	}
	else {
		result.bInside = false;
	}

	result.normal = MathHelpers::safeNormalize(ray.getPositionAtTime(t0));
	result.t = t0;

	// Flip the normal if we were inside
	float normalSign = result.bInside ? -1.0f : 1.0f;
	result.normal *= normalSign;

	// Fill in pdf if this was a light
	result.pdfIfLight = 1.0 / glm::pi<float>();

	return result;
}

__device__ IntersectionData IntersectionStatics::intersectBox(const Ray& ray) {
	IntersectionData result = IntersectionData{};

	glm::vec3 boxMin(-0.5f);
	glm::vec3 boxMax(0.5f);

	// Using slab method for fast aabb
	glm::vec3 invDirection = 1.0f / ray.direction;

	glm::vec3 tMin = (boxMin - ray.origin) * invDirection;
	glm::vec3 tMax = (boxMax - ray.origin) * invDirection;

	glm::vec3 tNear = (glm::min)(tMin, tMax);
	glm::vec3 tFar = (glm::max)(tMin, tMax);

	float t0 = (glm::max)(tNear.x, (glm::max)(tNear.y, tNear.z));
	float t1 = (glm::min)(tFar.x, (glm::min)(tFar.y, tFar.z));

	// Check for miss
	if (t0 > t1 || t1 < 0.0f) {
		return result;
	}

	if (t0 < 0.0f) {
		result.t = t1;
		result.bInside = true;
	}
	else {
		result.t = t0;
		result.bInside = false;
	}

	if (result.t < 0.0f) {
		return result;
	}

	result.normal = glm::vec3(0.0f);
	if (result.t == tNear.x) {
		result.normal.x = (ray.direction.x > 0.0f) ? -1.0f : 1.0f;
	}
	else if (result.t == tNear.y) {
		result.normal.y = (ray.direction.y > 0.0f) ? -1.0f : 1.0f;
	}
	else {
		result.normal.z = (ray.direction.z > 0.0f) ? -1.0f : 1.0f;
	}

	// Flip the normal if we were inside
	float normalSign = result.bInside ? -1.0f : 1.0f;
	result.normal *= normalSign;

	// Fill in pdf if this was a light
	result.pdfIfLight = 1.0f / 6.0f;

	return result;
}

__device__ IntersectionData IntersectionStatics::intersectTriangle(const Ray& ray, const Triangle& tri) {
	IntersectionData result = IntersectionData{};

	// Fast ray triangle intersection
	const float EPSILON = 1e-8f;
	glm::vec3 edge1 = tri.v1 - tri.v0;
	glm::vec3 edge2 = tri.v2 - tri.v0;
	glm::vec3 h = glm::cross(ray.direction, edge2);
	float a = glm::dot(edge1, h);
	if (fabs(a) < EPSILON) {
		// Parallel to triangle, ignore
		return result;
	}
	float f = 1.0f / a;
	glm::vec3 s = ray.origin - tri.v0;
	float u = f * glm::dot(s, h);
	if (u < 0.0f || u > 1.0f) {
		return result;
	}
	glm::vec3 q = glm::cross(s, edge1);
	float v = f * glm::dot(ray.direction, q);
	if (v < 0.0f || u + v > 1.0f) {
		return result;
	}
	float t = f * glm::dot(edge2, q);
	if (t > EPSILON) {
		result.t = t;
		glm::vec3 cross = glm::cross(edge1, edge2);
		float crossMagnitude = glm::length(cross);

		// Fill in pdf if this was a light
		float triangleArea = 0.5f * crossMagnitude;
		result.pdfIfLight = 1.0f / triangleArea;

		result.normal = MathHelpers::safeNormalize(cross);
		result.bInside = glm::dot(result.normal, ray.direction) > 0.0f;

		float w = 1.0f - u - v;

		result.uv = w * tri.uv0 + u * tri.uv1 + v * tri.uv2;
	}

	return result;
}

__device__ IntersectionData IntersectionStatics::intersectMesh(
	const Ray& ray,
	const Geom& geometry,
	const BLASNode* blasNodes,
	const Triangle* sceneTriangles)
{
	IntersectionData result = IntersectionData{};

	int blasIterationCount = 0;

	int nodeStack[32];
	int stackPtr = 0;

	nodeStack[stackPtr++] = geometry.blasNodeOffset;

	while (stackPtr > 0) {
		blasIterationCount++;

		// Pop node off stack
		int nodeIndex = nodeStack[--stackPtr];
		const BLASNode& node = blasNodes[nodeIndex];

		float tNear;
		if (!node.intersect(ray, tNear)) {
			// Missed this node, skip it
			continue;
		}

		if (result.t > 0.0f && tNear > result.t) {
			// We already have a closer intersection, skip this node
			continue;
		}

		if (node.leftChild == -1) {
			// Hit leaf node, check triangles
			for (int t = 0; t < node.triangleCount; t++) {
				const Triangle& tri = sceneTriangles[node.triangleOffset + t];
				IntersectionData triResult = intersectTriangle(ray, tri);
				if (triResult.t > 0.0f && (result.t < 0.0f || triResult.t < result.t)) {
					result = triResult;
				}
			}
		}
		else {
			// Push children onto stack if there is room
			if (stackPtr + 2 < 32) {
				nodeStack[stackPtr++] = node.leftChild;
				nodeStack[stackPtr++] = node.leftChild + 1;
			}
		}
	}


	if (result.t > 0.0f) {
		// Flip the normal if we were inside
		float normalSign = result.bInside ? -1.0f : 1.0f;
		result.normal *= normalSign;

		// Update pdfifLight based on the number of triangles
		result.pdfIfLight = result.pdfIfLight / geometry.triangleCount;
	}

	result.blasIterationCount = blasIterationCount;

	return result;
}