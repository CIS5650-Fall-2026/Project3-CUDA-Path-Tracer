#include "config.h"
#include "intersections.cuh"
#include "render_settings.cuh"
#include "scene_structs.h"
#include "sdf/sdf_scene.cuh"

#include <cfloat>

__host__ __device__ float box_intersection_test(const Geom& box, Ray r,
                                                glm::vec3* intersection_point, glm::vec3* normal,
                                                bool* outside) {

    glm::vec3 ro = multiply_mv(box.transform.inverse, glm::vec4(r.org, 1.0f));
    glm::vec3 rd = multiply_mv(box.transform.inverse, glm::vec4(r.dir, 0.0f));

    float t_near = -FLT_MAX;
    float t_far = FLT_MAX;

    for (int axis = 0; axis < 3; ++axis) {
        if (rd[axis] == 0.0f) {
            if (ro[axis] < -0.5f || ro[axis] > 0.5f)
                return -1.0f;
            continue;
        }

        float t0 = (-0.5f - ro[axis]) / rd[axis];
        float t1 = (0.5f - ro[axis]) / rd[axis];
        if (t0 > t1) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        t_near = glm::max(t_near, t0);
        t_far = glm::min(t_far, t1);
    }

    if (t_near > t_far || t_far <= 0.0f) {
        return -1.0f;
    }

    bool out = t_near > 0.0f;
    float t = out ? t_near : t_far;
    if (outside) {
        *outside = out;
    }
    if (intersection_point) {
        *intersection_point = get_point_on_ray(r, t);
    }
    if (normal) {
        glm::vec3 p = ro + t * rd;
        glm::vec3 a = glm::abs(p);

        glm::vec3 nor(0.0f);
        if (a.x > a.y && a.x > a.z)
            nor.x = glm::sign(p.x);
        else if (a.y > a.z)
            nor.y = glm::sign(p.y);
        else
            nor.z = glm::sign(p.z);
        *normal = glm::normalize(multiply_mv(box.transform.inv_transpose, glm::vec4(nor, 0.0f)));
    }

    return t;
}

__host__ __device__ float sphere_intersection_test(const Geom& sphere, Ray r,
                                                   glm::vec3* intersection_point, glm::vec3* normal,
                                                   bool* outside) {
    glm::vec3 ro = multiply_mv(sphere.transform.inverse, glm::vec4(r.org, 1.0f));
    glm::vec3 rd = multiply_mv(sphere.transform.inverse, glm::vec4(r.dir, 0.0f));

    float a = glm::dot(rd, rd);
    float half_b = glm::dot(ro, rd);
    float c = glm::dot(ro, ro) - 1.f;

    float disc = half_b * half_b - a * c;
    if (disc < 0.f) {
        return -1.f;
    }

    float sqrt_d = glm::sqrt(disc);

    float t = (-half_b - sqrt_d) / a;

    constexpr float eps = 1e-4f;
    if (t <= eps) {
        t = (-half_b + sqrt_d) / a;
        if (t <= eps) {
            return -1.f;
        }
    }

    if (outside) {
        *outside = dot(ro, ro) >= 1.f;
    }
    if (intersection_point) {
        *intersection_point = get_point_on_ray(r, t);
    }
    if (normal) {
        glm::vec3 p = ro + rd * t;
        *normal = glm::normalize(multiply_mv(sphere.transform.inv_transpose, glm::vec4(p, 0.f)));
    }

    return t;
}

__host__ __device__ float plane_intersection_test(const Geom& plane, Ray r,
                                                  glm::vec3* intersection_point, glm::vec3* normal,
                                                  bool* back_facing) {

    glm::vec3 ro = multiply_mv(plane.transform.inverse, glm::vec4(r.org, 1.0f));
    glm::vec3 rd = multiply_mv(plane.transform.inverse, glm::vec4(r.dir, 0.0f));

    if (fabsf(rd.y) <= numeric::min_cos * glm::length(rd)) {
        return -1;
    }

    float t = -ro.y / rd.y;
    if (t <= 0.0001f) {
        return -1;
    }

    glm::vec3 p_w = ro + rd * t;

    if (fabsf(p_w.x) > 0.5f || fabsf(p_w.z) > 0.5f) {
        return -1;
    }

    if (back_facing) {
        *back_facing = rd.y > 0;
    }
    if (intersection_point) {
        *intersection_point = get_point_on_ray(r, t);
    }
    if (normal) {
        *normal = glm::normalize(multiply_mv(plane.transform.inv_transpose, glm::vec4(0, 1, 0, 0)));
    }
    return t;
}

__device__ float sdf_intersection_test(const Geom& sdf, Ray r, float t_max,
                                       glm::vec3* intersection_point, glm::vec3* normal,
                                       bool* back_facing) {
    glm::vec3 ro = multiply_mv(sdf.transform.inverse, glm::vec4(r.org, 1.0f));
    glm::vec3 rd_obj = multiply_mv(sdf.transform.inverse, glm::vec4(r.dir, 0.0f));
    float rd_len = glm::length(rd_obj);
    if (rd_len == 0.f) {
        return -1.0f;
    }
    float inv_rd_len = 1.f / rd_len;
    glm::vec3 rd = rd_obj * inv_rd_len;

    float s = scene_intersect(Ray{ro, rd}, 0.f, t_max * rd_len, c_settings.sdf_hit_eps * rd_len);
    if (s <= 0.0f)
        return -1.0f;

    float t = s * inv_rd_len;

    if (intersection_point) {
        *intersection_point = r.org + r.dir * t;
    }
    if (normal || back_facing) {
        glm::vec3 p = ro + rd * s;
        glm::vec3 nor = scene_normal(p, c_settings.sdf_normal_eps * rd_len);
        if (back_facing) {
            *back_facing = glm::dot(rd, nor) > 0.f;
        }
        if (normal) {
            *normal = glm::normalize(multiply_mv(sdf.transform.inv_transpose, glm::vec4(nor, 0.f)));
        }
    }

    return t;
}

__device__ bool visible_to_light(Ray r, float light_dist, const Geom* geoms, int num_geoms) {
    float min_t = light_dist;
    float t;

    for (int i = 0; i < num_geoms; ++i) {
        const Geom& geom = geoms[i];
        bool outside = true;
        switch (geom.type) {
        case GeomType::Cube:
            t = box_intersection_test(geom, r, nullptr, nullptr, nullptr);
            break;
        case GeomType::Plane:
            t = plane_intersection_test(geom, r, nullptr, nullptr, &outside);
            // only intersect with one side
            if (outside) {
                continue;
            }
            break;
        case GeomType::Sphere:
            t = sphere_intersection_test(geom, r, nullptr, nullptr, nullptr);
            break;
        case GeomType::Sdf:
            t = sdf_intersection_test(geom, r, min_t, nullptr, nullptr, nullptr);
        }
        if (t > 0 && t < min_t) {
            return false;
        }
    }
    return true;
}

__global__ void compute_intersections(int num_paths, const PathSegment* path_segments,
                                      const Geom* geoms, int num_geoms,
                                      ShadeableIntersection* intersections, MatId* isect_mat_ids) {
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths) {
        Ray r = path_segments[path_index].ray;

        float t;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_normal;

        // naive parse through global geoms
        for (int i = 0; i < num_geoms; i++) {
            const Geom& geom = geoms[i];

            switch (geom.type) {
            case Cube:
                t = box_intersection_test(geom, r, nullptr, &tmp_normal, &outside);
                break;
            case Sphere:
                t = sphere_intersection_test(geom, r, nullptr, &tmp_normal, &outside);
                break;
            case Plane:
                t = plane_intersection_test(geom, r, nullptr, &tmp_normal, &outside);
                // only intersect with one side
                if (outside) {
                    continue;
                }
                break;
            case Sdf:
                t = sdf_intersection_test(geom, r, t_min, nullptr, &tmp_normal, &outside);
                break;
            }

            // Compute the minimum t from the intersection tests to determine
            // what scene geometry object was hit first.
            if (t > 0.0f && t_min > t) {
                t_min = t;
                hit_geom_index = i;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1) {
            // hit env map
            intersections[path_index].t = -1.0f;
            isect_mat_ids[path_index] = MAT_MISS;
        } else {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].surface_normal = normal;
            switch (geoms[hit_geom_index].role) {
            case GeomRole::Light:
                isect_mat_ids[path_index] = MAT_LIGHT;
                intersections[path_index].light_idx = geoms[hit_geom_index].id;
                break;
            case GeomRole::Material:
                intersections[path_index].light_idx = -1;
                isect_mat_ids[path_index] = geoms[hit_geom_index].id;
                break;
            }
        }
    }
}
