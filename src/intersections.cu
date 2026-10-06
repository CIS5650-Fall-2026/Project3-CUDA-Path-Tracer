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

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float boundingBoxIntersectionTest(Bounds bound, const Ray& r) {
  glm::vec3 invDir = 1.0f / r.direction;

  glm::vec3 t0 = (bound.minBound - r.origin) * invDir;
  glm::vec3 t1 = (bound.maxBound - r.origin) * invDir;

  glm::vec3 tMin = glm::min(t0, t1);
  glm::vec3 tMax = glm::max(t0, t1);

  float tNear = glm::max(glm::max(tMin.x, tMin.y), tMin.z);
  float tFar = glm::min(glm::min(tMax.x, tMax.y), tMax.z);

  return tNear <= tFar && tFar > 0.0f;
}

__host__ __device__ float triangleIntersectionTest(
  Triangle triangle,
  Ray r,
  glm::vec3& intersectionPoint,
  glm::vec3& normal,
  bool& outside)
{
  // Möller–Trumbore intersection
  const float EPSILON = 0.0000001;
  glm::vec3 edge1, edge2, h, s, q;
  float a, f, u, v;

  edge1 = triangle.vertices[1] - triangle.vertices[0];
  edge2 = triangle.vertices[2] - triangle.vertices[0];

  h = glm::cross(r.direction, edge2);
  a = glm::dot(edge1, h);
  if (a > -EPSILON && a < EPSILON) {
    return -1; // ray is parallel to triangle
  }

  f = 1.0f / a;
  s = r.origin - triangle.vertices[0];
  u = f * glm::dot(s, h);
  if (u < 0.0f || u > 1.0f) {
    return -1;
  }

  q = glm::cross(s, edge1);
  v = f * glm::dot(r.direction, q);
  if (v < 0.0 || u + v > 1.0) {
    return -1;
  }

  float t = f * dot(edge2, q);
  if (t > EPSILON) {
    intersectionPoint = r.origin + r.direction * t;
    normal = glm::normalize((1.0f - u - v) * triangle.normals[0]
      + u * triangle.normals[1] + v * triangle.normals[2]);
    outside = (glm::dot(r.direction, normal) < 0.0f);
    return t;
  }
  else // line intersection but not a ray intersection.
    return -1;
}

__host__ __device__ float meshIntersectionTest(
  Geom geom,
  Ray r,
  glm::vec3& intersectionPoint,
  glm::vec3& normal,
  bool& outside,
  Triangle* dev_triangles,
  LinearBVHNode* dev_bvhNodes,
  int& hitMaterialId)
{
#if BOUNDING_BOX
  if (!boundingBoxIntersectionTest(geom.bounds, r)) {
    return -1.0f;
  }
#endif

  float t_min = FLT_MAX;
  bool hit = false;

  glm::vec3 tmp_intersect;
  glm::vec3 tmp_normal;
  bool tmp_outside;

#if BVH
  int toVisitOffset = 0, currentNodeIndex = 0;
  int nodesToVisit[MAX_BVH_DEPTH];
  nodesToVisit[toVisitOffset++] = 0;

  while (toVisitOffset > 0) {
    int currNodeIdx = nodesToVisit[--toVisitOffset];
    LinearBVHNode node = dev_bvhNodes[currNodeIdx];

    if (!boundingBoxIntersectionTest(node.bounds, r)) {
      continue;
    }

    if (node.nPrimitives > 0) {
      // intersect ray with primitives in leaf BVH node
      for (int i = 0; i < node.nPrimitives; i++) {
        Triangle tri = dev_triangles[node.primitivesOffset + i];
        float t = triangleIntersectionTest(tri, r, tmp_intersect, tmp_normal, tmp_outside);
        
        if (t > 0.0f && t < t_min) {
          t_min = t;
          intersectionPoint = tmp_intersect;
          normal = tmp_normal;
          outside = tmp_outside;
          hitMaterialId = tri.materialid;
          hit = true;
        }
      }
    }
    else {
      // put far BVH node on nodesToVisit stack, advance to near node
      if (r.direction[node.axis] < 0.0f) {
        nodesToVisit[toVisitOffset++] = currNodeIdx + 1;
        nodesToVisit[toVisitOffset++] = node.secondChildOffset;
      }
      else {
        nodesToVisit[toVisitOffset++] = node.secondChildOffset;
        nodesToVisit[toVisitOffset++] = currNodeIdx + 1;
      }
    }
  }

  return hit ? t_min : -1.0f;
#endif

  for (int i = 0; i < geom.numTriangles; i++) {
    Triangle tri = dev_triangles[i];

    float t = triangleIntersectionTest(tri, r, tmp_intersect, tmp_normal, tmp_outside);

    if (t > 0.0f && t < t_min) {
      t_min = t;
      intersectionPoint = tmp_intersect;
      normal = tmp_normal;
      outside = tmp_outside;
      hitMaterialId = tri.materialid;
      hit = true;
    }
  }

  return hit ? t_min : -1.0f;
}