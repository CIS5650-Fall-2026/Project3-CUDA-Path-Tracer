#include "../src/pathtrace.h"
#include "../src/mesh.h"
#include "../src/camera.h"
#include "../src/lighting.h"
#include "../src/interactions.h"
#include "../src/intersections.h"

#include <cmath>
#include <chrono>
#include <iostream>
#include <stdexcept>

void check(bool passed, const char* message) {
    if (!passed) {
        throw std::runtime_error(message);
    }
    std::cout << "PASS: " << message << std::endl;
}

bool closeEnough(glm::vec3 a, glm::vec3 b) {
    return glm::length(a - b) < 0.0001f;
}

Geom sphere(float size, int material) {
    Geom geom{};
    geom.type = SPHERE;
    geom.materialid = material;
    geom.transform = glm::mat4(size);
    geom.transform[3][3] = 1.0f;
    geom.inverseTransform = glm::inverse(geom.transform);
    geom.invTranspose = glm::transpose(geom.inverseTransform);
    return geom;
}

glm::vec3 render(Scene& scene, int samples = 1) {
    uchar4* pixels = nullptr;
    cudaMalloc(&pixels, sizeof(uchar4));
    pathtraceInit(&scene);
    for (int sample = 1; sample <= samples; sample++) {
        pathtrace(pixels, 0, sample);
    }
    glm::vec3 result = scene.state.image[0] / float(samples);
    pathtraceFree();
    cudaFree(pixels);
    return result;
}

void testScattering() {
    Material material{};
    material.color = glm::vec3(0.5f, 0.25f, 0.75f);
    thrust::default_random_engine rng(42);
    float cosineSum = 0.0f;
    for (int i = 0; i < 10000; i++) {
        PathSegment path{};
        path.color = glm::vec3(0.8f);
        path.ray.direction = glm::vec3(0, -1, 0);
        scatterRay(path, glm::vec3(0), glm::vec3(0, 1, 0), material, rng);
        if (path.ray.direction.y < 0 || std::abs(glm::length(path.ray.direction) - 1) > 0.0001f
            || path.ray.origin.y <= 0 || !closeEnough(path.color, material.color * 0.8f)) {
            throw std::runtime_error("invalid diffuse bounce");
        }
        cosineSum += path.ray.direction.y;
    }
    check(std::abs(cosineSum / 10000 - 2.0f / 3.0f) < 0.02f,
        "diffuse directions follow the cosine-weighted hemisphere");
}

void testBox() {
    Geom box = sphere(1.0f, 0);
    box.type = CUBE;
    glm::vec3 point, normal;
    bool outside;
    float distance = boxIntersectionTest(box, {glm::vec3(0), glm::vec3(1, 0, 0)}, point, normal, outside);
    check(std::abs(distance - 0.5f) < 0.0001f && !outside && closeEnough(normal, glm::vec3(1, 0, 0)),
        "ray inside a box finds the exit face");
    distance = boxIntersectionTest(box, {glm::vec3(1, 0, 2), glm::vec3(0, 0, -1)}, point, normal, outside);
    check(distance < 0, "parallel ray outside a box misses");
    distance = boxIntersectionTest(box, {glm::vec3(0, 0, 2), glm::vec3(0, 0, -1)}, point, normal, outside);
    check(std::abs(distance - 1.5f) < 0.0001f && outside && closeEnough(normal, glm::vec3(0, 0, 1)),
        "outside box hit has an exact distance and normal");
}

void testGlassAndMirror() {
    Material material{};
    material.color = glm::vec3(1);
    material.hasReflective = 1;
    PathSegment path{};
    path.color = glm::vec3(1);
    path.ray.direction = glm::normalize(glm::vec3(1, -1, 0));
    thrust::default_random_engine rng(42);
    scatterRay(path, glm::vec3(0), glm::vec3(0, 1, 0), material, rng);
    check(closeEnough(path.ray.direction, glm::normalize(glm::vec3(1, 1, 0)))
        && path.ray.origin.y > 0, "mirror reflects at the incident angle");

    material.hasReflective = 0;
    material.hasRefractive = 1;
    material.indexOfRefraction = 1.5f;
    int reflected = 0;
    int transmitted = 0;
    for (int i = 0; i < 20000; i++) {
        path.color = glm::vec3(1);
        path.ray.direction = glm::vec3(0, -1, 0);
        scatterRay(path, glm::vec3(0), glm::vec3(0, 1, 0), material, rng);
        if (path.ray.direction.y > 0) {
            reflected++;
        } else {
            transmitted++;
            if (path.ray.origin.y >= 0 || !closeEnough(path.color, glm::vec3(1.0f / 2.25f))) {
                throw std::runtime_error("incorrect glass transmission");
            }
        }
    }
    check(transmitted > 0 && std::abs(reflected / 20000.0f - 0.04f) < 0.01f,
        "normal-incidence glass reflects about four percent of paths");

    bool checkedSnell = false;
    for (int i = 0; i < 100; i++) {
        path.color = glm::vec3(1);
        path.ray.direction = glm::vec3(0.6f, -0.8f, 0);
        scatterRay(path, glm::vec3(0), glm::vec3(0, 1, 0), material, rng);
        if (path.ray.direction.y < 0) {
            checkedSnell = std::abs(path.ray.direction.x - 0.4f) < 0.0001f;
            break;
        }
    }
    check(checkedSnell, "glass transmission follows Snell's law");
    path.color = glm::vec3(1);
    path.ray.direction = glm::vec3(0.8f, 0.6f, 0);
    scatterRay(path, glm::vec3(0), glm::vec3(0, 1, 0), material, rng);
    check(closeEnough(path.ray.direction, glm::vec3(0.8f, -0.6f, 0))
        && path.ray.origin.y < 0 && closeEnough(path.color, glm::vec3(1)),
        "total internal reflection stays inside the object");

    Geom glass = sphere(2, 0);
    glm::vec3 point, normal;
    bool outside;
    sphereIntersectionTest(glass, {glm::vec3(0), glm::vec3(0, 1, 0)}, point, normal, outside);
    check(!outside && closeEnough(normal, glm::vec3(0, 1, 0)),
        "sphere exit keeps its outward normal");
}

std::vector<glm::vec3> renderImage(Scene& scene, int samples, int seedOffset = 0) {
    int count = scene.state.camera.resolution.x * scene.state.camera.resolution.y;
    uchar4* pixels = nullptr;
    cudaMalloc(&pixels, count * sizeof(uchar4));
    pathtraceInit(&scene);
    for (int sample = 1; sample <= samples; sample++) {
        pathtrace(pixels, 0, sample + seedOffset);
    }
    std::vector<glm::vec3> image = scene.state.image;
    pathtraceFree();
    cudaFree(pixels);
    return image;
}

void testPathOrdering(const char* filename) {
    Scene scene(filename);
    scene.state.camera.pixelLength *= float(scene.state.camera.resolution.x) / 48.0f;
    scene.state.camera.resolution = glm::ivec2(48);
    scene.state.image.resize(48 * 48);
    scene.state.recordPathCounts = true;

    for (bool antialiasing : {false, true}) {
        scene.state.antialiasing = antialiasing;
        scene.state.streamCompaction = false;
        scene.state.materialSorting = false;
        auto reference = renderImage(scene, 16);
        auto referenceCounts = scene.state.activePaths;
        for (bool compaction : {false, true}) {
            for (bool sorting : {false, true}) {
                scene.state.streamCompaction = compaction;
                scene.state.materialSorting = sorting;
                auto image = renderImage(scene, 16);
                for (int i = 0; i < image.size(); i++) {
                    if (!closeEnough(reference[i], image[i])) {
                        throw std::runtime_error("reordering paths changed a pixel");
                    }
                }
                auto counts = scene.state.activePaths;
                counts.resize(referenceCounts.size(), 0);
                if (counts != referenceCounts) {
                    throw std::runtime_error("reordering paths changed the active counts");
                }
            }
        }
        check(true, antialiasing
            ? "sorting and compaction preserve all pixels and counts with antialiasing"
            : "sorting and compaction preserve all pixels and counts without antialiasing");
    }
}

void testAntialiasing(const char* filename) {
    Scene scene(filename);
    Camera& camera = scene.state.camera;
    camera.apertureRadius = 0.0f;
    camera.resolution = glm::ivec2(1);
    camera.position = glm::vec3(0, 0, 2);
    camera.view = glm::vec3(0, 0, -1);
    camera.right = glm::vec3(1, 0, 0);
    camera.up = glm::vec3(0, 1, 0);
    camera.pixelLength = glm::vec2(0.2f);
    scene.state.image.resize(1);
    scene.state.traceDepth = 1;

    Material light{};
    light.color = glm::vec3(1);
    light.emittance = 1;
    scene.materials = {light};
    Geom box = sphere(1, 0);
    box.type = CUBE;
    // The right edge passes through the middle of the only pixel.
    box.transform[3][0] = -0.5f;
    box.inverseTransform = glm::inverse(box.transform);
    box.invTranspose = glm::transpose(box.inverseTransform);
    scene.geoms = {box};
    scene.state.antialiasing = false;
    glm::vec3 center = render(scene, 16);
    check(closeEnough(center, glm::vec3(1)), "disabled antialiasing samples the pixel center");

    scene.state.antialiasing = true;
    glm::vec3 coverage = render(scene, 2048);
    check(std::abs(coverage.x - 0.5f) < 0.04f && closeEnough(coverage, glm::vec3(coverage.x)),
        "jittered rays recover fifty-percent coverage at a half-pixel edge");
    check(closeEnough(coverage, render(scene, 2048)), "camera samples repeat after accumulation resets");
}
void testMeshLoading() {
    for (const char* file : {"external.gltf", "interleaved.gltf", "embedded.gltf", "binary.glb", "unindexed.gltf"}) {
        Geom geom{};
        geom.type = MESH;
        geom.transform = glm::mat4(1);
        std::vector<Triangle> triangles;
        loadGltf(std::string("tests/meshes/") + file, geom, triangles);
        check(triangles.size() == 1 && closeEnough(triangles[0].a, glm::vec3(-1, -1, 0))
            && closeEnough(triangles[0].normal, glm::vec3(0, 0, 1)), file);
        glm::vec3 normal;
        float hit = meshIntersectionTest(geom, triangles.data(),
            {glm::vec3(0, 0, 2), glm::vec3(0, 0, -1)}, normal, true, 100);
        check(std::abs(hit - 2.0f) < 0.0001f, "mesh intersection finds the triangle");
        float miss = meshIntersectionTest(geom, triangles.data(),
            {glm::vec3(3, 0, 2), glm::vec3(0, 0, -1)}, normal, true, 100);
        check(miss < 0, "parallel ray outside mesh bounds misses");
    }
    Geom geom{};
    geom.transform = glm::mat4(1);
    std::vector<Triangle> triangles;
    loadGltf("tests/meshes/transformed.gltf", geom, triangles);
    check(closeEnough(triangles[0].a, glm::vec3(4, -3, 0))
        && closeEnough(triangles[0].normal, glm::vec3(0, 0, 1)),
        "node hierarchy and mirrored nonuniform scale preserve outward normals");
    for (const char* file : {"invalid-range.gltf", "cycle.gltf"}) {
        bool rejected = false;
        try {
            std::vector<Triangle> invalid;
            loadGltf(std::string("tests/meshes/") + file, geom, invalid);
        } catch (const std::exception&) {
            rejected = true;
        }
        check(rejected, file);
    }
    Triangle nearTriangle = {glm::vec3(-1, -1, 1), glm::vec3(1, -1, 1),
        glm::vec3(0, 1, 1), glm::vec3(0, 0, 1)};
    triangles = { {glm::vec3(-1, -1, 0), glm::vec3(1, -1, 0),
        glm::vec3(0, 1, 0), glm::vec3(0, 0, 1)}, nearTriangle };
    geom.triangleStart = 0;
    geom.triangleCount = 2;
    geom.boundsMin = glm::vec3(-1, -1, -0.001f);
    geom.boundsMax = glm::vec3(1, 1, 1.001f);
    glm::vec3 normal;
    Ray ray{glm::vec3(0, 0, 2), glm::vec3(0, 0, -1)};
    check(std::abs(meshIntersectionTest(geom, triangles.data(), ray, normal, true, 100) - 1) < 0.0001f,
        "mesh intersection selects the closest triangle");
    check(intersectsBounds({glm::vec3(0), glm::vec3(0, 0, 1)}, geom.boundsMin, geom.boundsMax, 100),
        "ray starting inside mesh bounds is retained");
}

__global__ void generateRayFromCamera(Camera camera, int iteration, int depth,
    bool antialiasing, PathSegment* paths);

void testLens() {
    Camera camera{};
    camera.resolution = glm::ivec2(17, 9);
    camera.position = glm::vec3(0, 0, 5);
    camera.view = glm::vec3(0, 0, -1);
    camera.right = glm::vec3(1, 0, 0);
    camera.up = glm::vec3(0, 1, 0);
    camera.pixelLength = glm::vec2(0.1f);
    camera.focalDistance = 5;
    Ray center{camera.position, camera.view};
    check(closeEnough(sampleLens(camera, center, 0.8f, 0.3f).origin, center.origin)
        && closeEnough(sampleLens(camera, center, 0.8f, 0.3f).direction, center.direction),
        "zero aperture preserves the pinhole ray");

    camera.apertureRadius = 0.5f;
    PathSegment* devicePaths = nullptr;
    cudaMalloc(&devicePaths, 17 * 9 * sizeof(PathSegment));
    std::vector<PathSegment> paths(17 * 9);
    float squaredRadiusSum = 0;
    for (int sample = 1; sample <= 16; sample++) {
        generateRayFromCamera<<<dim3(3, 2), dim3(8, 8)>>>(camera, sample, 4, false, devicePaths);
        if (cudaMemcpy(paths.data(), devicePaths, paths.size() * sizeof(PathSegment),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("could not read GPU lens rays");
        }
        for (int y = 0; y < 9; y++) {
            for (int x = 0; x < 17; x++) {
                const Ray& ray = paths[x + y * 17].ray;
                glm::vec3 offset = ray.origin - camera.position;
                float radiusSquared = glm::dot(offset, offset);
                squaredRadiusSum += radiusSquared;
                glm::vec3 focus = ray.origin + ray.direction * (-ray.origin.z / ray.direction.z);
                glm::vec3 expected(-(x + 0.5f - 8.5f) * 0.5f, -(y + 0.5f - 4.5f) * 0.5f, 0);
                if (radiusSquared > 0.25001f || std::abs(offset.z) > 0.0001f || !closeEnough(focus, expected)) {
                    throw std::runtime_error("lens ray left the aperture or missed its focus point");
                }
            }
        }
    }
    cudaFree(devicePaths);
    check(std::abs(squaredRadiusSum / (16 * 17 * 9) - 0.125f) < 0.01f,
        "lens samples cover the disk uniformly and meet at the focal plane");
}

void testMeshRendering() {
    Scene scene("scenes/benchmark-mesh.json");
    scene.state.camera.pixelLength *= float(scene.state.camera.resolution.x) / 32.0f;
    scene.state.camera.resolution = glm::ivec2(32, 24);
    scene.state.image.resize(32 * 24);
    scene.state.meshCulling = false;
    auto reference = renderImage(scene, 8);
    scene.state.meshCulling = true;
    auto culled = renderImage(scene, 8);
    float brightness = 0;
    for (int i = 0; i < reference.size(); i++) {
        brightness += reference[i].x;
        if (!closeEnough(reference[i], culled[i])) throw std::runtime_error("culling changed a pixel");
    }
    check(brightness > 0, "mesh scene produces a nonblack image");
    check(true, "mesh culling preserves every rendered pixel");
}
Geom boxAt(glm::vec3 position, glm::vec3 scale, int material) {
    Geom box{};
    box.type = CUBE;
    box.materialid = material;
    box.transform = utilityCore::buildTransformationMatrix(position, glm::vec3(0), scale);
    box.inverseTransform = glm::inverse(box.transform);
    box.invTranspose = glm::transpose(box.inverseTransform);
    return box;
}

void testLightSampling() {
    Geom box = boxAt(glm::vec3(0), glm::vec3(2, 3, 4), 0);
    LightSample face = sampleLightSurface(box, nullptr, 0.2f, 0.5f, 0.5f);
    check(closeEnough(face.position, glm::vec3(1, 0, 0))
        && std::abs(face.areaPdf - 1.0f / 72.0f) < 0.00001f,
        "box light PDF includes face selection and transformed area");

    check(std::abs(face.areaPdf - lightAreaPdf(box, nullptr, face.position, -1)) < 0.00001f,
        "box light-hit PDF matches the surface sampler");
    check(std::abs(powerWeight(0.3f, 0.7f) + powerWeight(0.7f, 0.3f) - 1.0f) < 0.00001f
        && std::isfinite(powerWeight(1e30f, 1e-30f)),
        "MIS weights are complementary and remain finite for very different PDFs");
    Geom ellipsoid = box;
    ellipsoid.type = SPHERE;
    LightSample point = sampleLightSurface(ellipsoid, nullptr, 0, 0.5f, 0);
    check(closeEnough(point.position, glm::vec3(1, 0, 0))
        && std::abs(point.areaPdf - 1.0f / (12.0f * PI)) < 0.00001f,
        "ellipsoid light PDF includes its surface area Jacobian");

    check(std::abs(point.areaPdf - lightAreaPdf(ellipsoid, nullptr, point.position, -1)) < 0.00001f,
        "ellipsoid light-hit PDF matches the surface sampler");
    Triangle triangle{glm::vec3(0), glm::vec3(2, 0, 0), glm::vec3(0, 2, 0), glm::vec3(0, 0, 1)};
    Geom mesh{};
    mesh.type = MESH;
    mesh.triangleCount = 1;
    LightSample sample = sampleLightSurface(mesh, &triangle, 0, 0.25f, 0.5f);
    check(closeEnough(sample.position, glm::vec3(0.5f, 0.5f, 0))
        && std::abs(sample.areaPdf - 0.5f) < 0.00001f,
        "mesh light sampling uses triangle area and barycentric coordinates");
    check(std::abs(sample.areaPdf - lightAreaPdf(mesh, &triangle, sample.position, 0)) < 0.00001f,
        "mesh light-hit PDF matches the surface sampler");
    mesh.boundsMin = glm::vec3(-0.001f);
    mesh.boundsMax = glm::vec3(2.001f, 2.001f, 0.001f);
    Ray shadow{glm::vec3(0.5f, 0.5f, 1), glm::vec3(0, 0, -1)};
    check(shadowBlocked(shadow, 2, &mesh, 1, &triangle, true)
        && shadowBlocked(shadow, 2, &mesh, 1, &triangle, false)
        && !shadowBlocked(shadow, 0.5f, &mesh, 1, &triangle, true),
        "mesh shadow tests respect culling and segment length");
    Geom ball = sphere(2, 0);
    Ray throughBall{glm::vec3(0, 0, 3), glm::vec3(0, 0, -1)};
    check(shadowBlocked(throughBall, 3.9998f, &ball, 1, nullptr, true)
        && !shadowBlocked(throughBall, 1.9998f, &ball, 1, nullptr, true),
        "a sphere blocks samples on its far side but not its visible surface");
}
void testDirectLighting(const char* filename) {
    Scene scene(filename);
    Camera& camera = scene.state.camera;
    camera.resolution = glm::ivec2(1);
    camera.position = glm::vec3(0, 0, 0.5f);
    camera.view = glm::vec3(0, 0, -1);
    camera.right = glm::vec3(1, 0, 0);
    camera.up = glm::vec3(0, 1, 0);
    camera.pixelLength = glm::vec2(0);
    camera.apertureRadius = 0;
    scene.state.traceDepth = 2;
    scene.state.image.resize(1);
    scene.state.directLighting = true;

    Material diffuse{};
    diffuse.color = glm::vec3(0.5f, 0.25f, 0.75f);
    Material light{};
    light.color = glm::vec3(1, 0.8f, 0.6f);
    light.emittance = 2;
    scene.materials = {diffuse, light};
    scene.triangles = {
        {glm::vec3(-1, -1, 2), glm::vec3(1, -1, 2), glm::vec3(1, 1, 2), glm::vec3(0, 0, 1)},
        {glm::vec3(-1, -1, 2), glm::vec3(1, 1, 2), glm::vec3(-1, 1, 2), glm::vec3(0, 0, 1)}
    };
    Geom emitter{};
    emitter.type = MESH;
    emitter.materialid = 1;
    emitter.triangleCount = 2;
    emitter.boundsMin = glm::vec3(-1, -1, 1.999f);
    emitter.boundsMax = glm::vec3(1, 1, 2.001f);
    scene.geoms = {boxAt(glm::vec3(0, 0, -0.5f), glm::vec3(4, 4, 1), 0), emitter};

    // Independent midpoint integration over the square light.
    double irradiance = 0;
    for (int y = 0; y < 200; y++) {
        for (int x = 0; x < 200; x++) {
            double px = (x + 0.5) * 0.01 - 1;
            double py = (y + 0.5) * 0.01 - 1;
            double distanceSquared = px * px + py * py + 4;
            irradiance += 4.0 / (distanceSquared * distanceSquared) * 0.0001 / PI;
        }
    }
    glm::vec3 expected = diffuse.color * light.color * light.emittance * float(irradiance);
    auto matches = [&](glm::vec3 actual, float tolerance) {
        return glm::length(actual - expected) < tolerance * glm::length(expected);
    };
    glm::vec3 sampled = render(scene, 2048);
    check(matches(sampled, 0.025f), "direct light matches independent quadrature without doubled emission");
    scene.state.directLighting = false;
    check(matches(render(scene, 8192), 0.07f), "BSDF-only lighting agrees with the same reference");
    scene.state.directLighting = true;

    // Splitting the square into two light objects must not halve or double its energy.
    scene.geoms[1].triangleCount = 1;
    emitter.triangleStart = 1;
    emitter.triangleCount = 1;
    scene.geoms.push_back(emitter);
    check(matches(render(scene, 2048), 0.025f), "light selection probability preserves energy with multiple lights");
    scene.geoms.resize(2);
    scene.geoms[1].triangleCount = 2;

    scene.geoms.push_back(boxAt(glm::vec3(0, 0, 3), glm::vec3(4, 4, 0.1f), 0));
    check(closeEnough(render(scene, 2048), sampled), "geometry beyond the light does not block its shadow ray");
    scene.geoms.back() = boxAt(glm::vec3(0, 0, 1), glm::vec3(4, 4, 0.1f), 0);
    check(closeEnough(render(scene, 128), glm::vec3(0)), "box occluder blocks direct light");
    scene.geoms.pop_back();

    scene.state.traceDepth = 1;
    check(closeEnough(render(scene, 8), glm::vec3(0)), "direct sampling respects the surface depth limit");
    scene.state.traceDepth = 2;
    scene.materials[0].hasReflective = 1;
    check(closeEnough(render(scene, 16), diffuse.color * light.color * light.emittance),
        "mirror-to-light paths keep their emission with direct sampling enabled");
    scene.materials[0].hasReflective = 0;
    scene.geoms.erase(scene.geoms.begin());
    camera.view = glm::vec3(0, 0, 1);
    check(closeEnough(render(scene, 4), light.color * light.emittance),
        "camera-visible lights keep their emission with direct sampling enabled");

    camera.view = glm::vec3(0, 0, -1);
    scene.geoms = {boxAt(glm::vec3(0, 0, -0.5f), glm::vec3(4, 4, 1), 0), sphere(10, 1)};
    glm::vec3 sphereResult = render(scene, 8192);
    glm::vec3 sphereExpected = diffuse.color * light.color * light.emittance;
    check(glm::length(sphereResult - sphereExpected) < 0.05f * glm::length(sphereExpected),
        "sphere light sampling agrees with a uniform enclosing emitter");
    scene.materials[0].hasRefractive = 1;
    scene.materials[0].indexOfRefraction = 1.5f;
    scene.materials[0].color = glm::vec3(1);
    scene.state.traceDepth = 16;
    check(closeEnough(render(scene, 128), light.color * light.emittance),
        "glass-to-light paths keep their emission with MIS enabled");
    scene.geoms.pop_back();
    check(closeEnough(render(scene, 4), glm::vec3(0)), "scenes without emitters remain black");
}
void compareLighting(const char* filename) {
    Scene scene(filename);
    scene.state.camera.pixelLength *= float(scene.state.camera.resolution.x) / 64.0f;
    scene.state.camera.resolution = glm::ivec2(64);
    scene.state.image.resize(64 * 64);
    scene.state.directLighting = true;
    auto reference = renderImage(scene, 4096, 10000);
    std::cout << "Reference: 64x64, 4096 direct-light samples, seeds 10001 through 14096" << std::endl;
    std::cout << "direct_lighting,samples,milliseconds,mse,mean_radiance" << std::endl;
    for (bool enabled : {false, true}) {
        scene.state.directLighting = enabled;
        auto start = std::chrono::steady_clock::now();
        auto image = renderImage(scene, 128);
        double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        double error = 0, mean = 0;
        for (int i = 0; i < image.size(); i++) {
            glm::vec3 value = image[i] / 128.0f;
            glm::vec3 difference = value - reference[i] / 4096.0f;
            error += glm::dot(difference, difference);
            mean += value.x + value.y + value.z;
        }
        std::cout << enabled << ",128," << milliseconds << "," << error / (3 * image.size())
            << "," << mean / (3 * image.size()) << std::endl;
    }
}
void benchmark(const char* filename, const std::string& feature) {
    Scene scene(filename);
    int count = scene.state.camera.resolution.x * scene.state.camera.resolution.y;
    uchar4* pixels = nullptr;
    cudaMalloc(&pixels, count * sizeof(uchar4));
    float aperture = scene.state.camera.apertureRadius > 0 ? scene.state.camera.apertureRadius : 0.6f;
    const auto originalMaterials = scene.materials;
    std::cout << "round," << feature << ",milliseconds_per_sample" << std::endl;
    for (int round = 0; round < 3; round++) {
        for (int mode = 0; mode < 2; mode++) {
            bool enabled = (mode + round) % 2 != 0;
            if (feature == "sorting") {
                scene.state.materialSorting = enabled;
            } else if (feature == "culling") {
                scene.state.meshCulling = enabled;
            } else if (feature == "lighting") {
                scene.state.directLighting = enabled;
            } else if (feature == "dof") {
                scene.state.camera.apertureRadius = enabled ? aperture : 0.0f;
            } else if (feature == "refraction") {
                scene.materials = originalMaterials;
                if (!enabled) {
                    for (auto& material : scene.materials) {
                        material.hasRefractive = 0.0f;
                    }
                }
            } else {
                scene.state.streamCompaction = enabled;
            }
            pathtraceInit(&scene);
            for (int sample = 1; sample <= 5; sample++) {
                pathtrace(pixels, 0, sample);
            }
            auto start = std::chrono::steady_clock::now();
            for (int sample = 6; sample <= 55; sample++) {
                pathtrace(pixels, 0, sample);
            }
            double milliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count() / 50;
            std::cout << round << "," << enabled << "," << milliseconds << std::endl;
            pathtraceFree();
        }
    }
    scene.state.recordPathCounts = true;
    scene.state.streamCompaction = false;
    pathtraceInit(&scene);
    pathtrace(pixels, 0, 1);
    std::cout << "bounce,active_paths" << std::endl;
    for (int i = 0; i < scene.state.activePaths.size(); i++) {
        std::cout << i + 1 << "," << scene.state.activePaths[i] << std::endl;
    }
    pathtraceFree();
    cudaFree(pixels);
}
int main(int argc, char** argv) {
    if (argc == 3) {
        std::string option = argv[2];
        if (option == "--compare-lighting") { compareLighting(argv[1]); return 0; }
        if (option == "--benchmark") benchmark(argv[1], "compaction");
        else if (option == "--benchmark-sorting") benchmark(argv[1], "sorting");
        else if (option == "--benchmark-culling") benchmark(argv[1], "culling");
        else if (option == "--benchmark-dof") benchmark(argv[1], "dof");
        else if (option == "--benchmark-lighting") benchmark(argv[1], "lighting");
        else if (option == "--benchmark-refraction") benchmark(argv[1], "refraction");
        else return 1;
        return 0;
    }
    if (argc != 2) {
        std::cerr << "Usage: pathtrace_checks SCENEFILE.json\n";
        return 1;
    }
    try {
        testLightSampling();
        testDirectLighting(argv[1]);
        testMeshLoading();
        testLens();
        testMeshRendering();
        testGlassAndMirror();
        testPathOrdering(argv[1]);
        testAntialiasing(argv[1]);
        testScattering();
        testBox();
        Scene scene(argv[1]);
        scene.state.directLighting = false;
        Camera& camera = scene.state.camera;
        camera.apertureRadius = 0.0f;
    camera.resolution = glm::ivec2(1);
        camera.position = glm::vec3(0, 0, 2);
        camera.view = glm::vec3(0, 0, -1);
        camera.right = glm::vec3(1, 0, 0);
        camera.up = glm::vec3(0, 1, 0);
        camera.pixelLength = glm::vec2(0);
        scene.state.image.resize(1);
        Material diffuse{};
        diffuse.color = glm::vec3(0.5f, 0.25f, 0.75f);
        Material light{};
        light.color = glm::vec3(1.0f, 0.8f, 0.6f);
        light.emittance = 2.0f;
        scene.materials = {diffuse, light};
        scene.geoms = {sphere(1, 1)};
        scene.state.traceDepth = 4;
        check(closeEnough(render(scene, 3), light.color * light.emittance),
            "light hits terminate and accumulation averages correctly");
        scene.geoms[0].materialid = 0;
        check(closeEnough(render(scene), glm::vec3(0)), "diffuse ray escaping the scene is black");
        scene.geoms.push_back(sphere(20, 1));
        check(closeEnough(render(scene, 8), diffuse.color * light.color * light.emittance),
            "diffuse bounce reaches enclosing light without hitting itself");
        scene.materials[0].hasReflective = 1;
        check(closeEnough(render(scene, 8), diffuse.color * light.color * light.emittance),
            "mirror bounce reaches the light on the GPU");
        scene.materials[0].hasReflective = 0;
        scene.materials[0].hasRefractive = 1;
        scene.materials[0].indexOfRefraction = 1.5f;
        scene.materials[0].color = glm::vec3(1);
        scene.state.traceDepth = 16;
        check(closeEnough(render(scene, 128), light.color * light.emittance),
            "glass entry and exit preserve radiance in an enclosing light");
        scene.state.traceDepth = 1;
        check(closeEnough(render(scene), glm::vec3(0)), "depth limit discards unfinished throughput");
        scene.state.traceDepth = 0;
        check(closeEnough(render(scene), glm::vec3(0)), "zero depth contributes no light");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
