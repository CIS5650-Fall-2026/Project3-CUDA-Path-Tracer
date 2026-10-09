#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float mollerTrumbore(
    const Triangle& tri,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec3& tangent,
    glm::vec2& uv,
    bool& outside
) {
    // vectors for two edges that share v0
	glm::vec3 e1 = tri.v1 - tri.v0;
	glm::vec3 e2 = tri.v2 - tri.v0;

    // find determinant
	glm::vec3 p = glm::cross(r.direction, e2);
	float determinant = glm::dot(e1, p);
	// suppose ray is parallel to triangle plane
    if (fabs(determinant) < 1e-8f) {
        return -1.0f;
	}
	float invDeterminant = 1.0f / determinant;

	// distance between v0 and ray origin, find u
    glm::vec3 tVec = r.origin - tri.v0;
	float u = glm::dot(tVec, p) * invDeterminant;
    if ((u < 0.f) || (u > 1.f)) {
        return -1.0f;
    }

    // find v
	glm::vec3 q = glm::cross(tVec, e1);
	float v = glm::dot(r.direction, q) * invDeterminant;
    if ((v < 0.f) || (u + v > 1.f)) {
        return -1.0f;
    }
	
    float t = glm::dot(e2, q) * invDeterminant;
    if (t <= 0.f) {
        return -1.f;
    }

	intersectionPoint = r.origin + t * r.direction;
    normal = glm::normalize((1.f - u - v) * tri.n0 + u * tri.n1 + v * tri.n2);

    // try to improve shading at glancing angles (noticeable for thin meshes, where the backside will get oversaturated)
    glm::vec3 geoN = glm::normalize(glm::cross(e1, e2));
    outside = glm::dot(geoN, r.direction) < 0.f;
    if (!outside) {
        geoN = -geoN;
    }
    if (glm::dot(normal, geoN) < 0.f) {
        normal = -normal;
    }

    // use barycentric interpolation for uvs
	uv = (1.f - u - v) * tri.uv0 + u * tri.uv1 + v * tri.uv2;
    tangent = tri.tangent;

    return t;
}

// "Slab" test for BVH: return entry distance, reject any boxes beyond current closest hit
// Recall: the box test is composed of three sections (region between two x planes, y planes, z planes respectively)
// ray inside box when it's in all three regions (so test to find the extent of ray inside each region and test if there's overlap)
__host__ __device__ inline bool aabbHit(
    const glm::vec3& bMin,
    const glm::vec3& bMax,
    const Ray& r,
    float tMax,
    float& tEntry
) {
    glm::vec3 invDir = 1.f / r.direction;

    // low plane and high plane
    glm::vec3 t0 = (bMin - r.origin) * invDir;
    glm::vec3 t1 = (bMax - r.origin) * invDir;

    // distance when ray reaches certain plane
    glm::vec3 tsm = glm::min(t0, t1);
    glm::vec3 tbg = glm::max(t0, t1);

    // suppose ray travels along negative axis; reaches high plane first
    // this makes direction negligible
    // tNear: most recent entry, tFar: earliest exit
    float tNear = glm::max(glm::max(tsm.x, tsm.y), tsm.z);
    float tFar = glm::min(glm::min(tbg.x, tbg.y), tbg.z);

    // 1. ray leaves a region before entering another
    // 2. entire box behind ray origin
    // 3. box begins farther away than the closest found hit
    if (tNear > tFar || tFar < 0.f || tNear > tMax) {
        return false;
    }

    tEntry = glm::max(tNear, 0.f);

    return true;
}

__host__ __device__ float triangleIntersectionTest(
    const Geom& mesh,
    const Triangle* triangles,
    const BVHNode* nodes,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec3& tangent,
    glm::vec2& uv,
    bool& outside,
    int& triIdx) {
    // everything should already be in world space (this was done during loading)
    float tBest = FLT_MAX;
    triIdx = -1;
#if BOUNDING_VOLUME_CULLING
    float tBox;
    if (!aabbHit(mesh.bboxMin, mesh.bboxMax, r, FLT_MAX, tBox)) {
        return -1.f;
    }
#endif

#if USE_BVH
    int stack[BVH_STACK_SIZE];
    float tStack[BVH_STACK_SIZE];
    int sp = 0;
    float tRoot;
    const BVHNode& root = nodes[mesh.bvhRoot];

    if (!aabbHit(root.bboxMin, root.bboxMax, r, tBest, tRoot)) {
        return -1.f;
    }

    stack[sp] = mesh.bvhRoot;
    tStack[sp] = tRoot;
    sp++;

    while (sp > 0) {
        // prevent stack overflow
        if ((sp + 2) > BVH_STACK_SIZE) {
            break;
        }

        sp--;
        // closer hit found
        if (tStack[sp] >= tBest) {
            continue;
        }
        const BVHNode& node = nodes[stack[sp]];

        if (node.left < 0) {
            for (int i = node.triStart; i < node.triStart + node.numTris; ++i) {
                glm::vec3 p;
                glm::vec3 n;
                glm::vec3 tan;
                glm::vec2 uvs;
                bool o;
                float t = mollerTrumbore(triangles[i], r, p, n, tan, uvs, o);

                if (t > 0.f && t < tBest) {
                    tBest = t;
                    intersectionPoint = p;
                    normal = n;
                    tangent = tan;
                    uv = uvs;
                    outside = o;
                    triIdx = i;
                }
            }
        }
        else { // test children
            float tL;
            float tR;
            const BVHNode& L = nodes[node.left];
            const BVHNode& R = nodes[node.right];
            bool hitL = aabbHit(L.bboxMin, L.bboxMax, r, tBest, tL);
            bool hitR = aabbHit(R.bboxMin, R.bboxMax, r, tBest, tR);

            if (hitL && hitR) {
                bool leftNear = tL <= tR;
                // do far child first
                stack[sp] = leftNear ? node.right : node.left;
                tStack[sp] = leftNear ? tR : tL;
                sp++;
                // the near child on top
                stack[sp] = leftNear ? node.left : node.right;
                tStack[sp] = leftNear ? tL : tR;
                sp++;
            }
            else if (hitL) {
                stack[sp] = node.left;
                tStack[sp] = tL;
                sp++;
            }
            else if (hitR) {
                stack[sp] = node.right;
                tStack[sp] = tR;
                sp++;
            }
        }
    }

#else

    for (int i = mesh.triStart; i < mesh.triStart + mesh.numTris; ++i) {
        glm::vec3 p;
        glm::vec3 n;
        glm::vec3 tan;
        glm::vec2 uvs;
        bool o;
		float t = mollerTrumbore(triangles[i], r, p, n, tan, uvs, o);
        if (t > 0.f && t < tBest) {
            tBest = t;
			intersectionPoint = p;
            normal = n;
            tangent = tan;
            uv = uvs;
            outside = o;
			triIdx = i;
        }
	}
#endif

	return triIdx >= 0 ? tBest : -1.f;
}
