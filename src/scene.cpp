#include "scene.h"

#include "sceneStructs.h"
#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

// TODO: make unique ptrs for the geometry
using namespace std;
using json = nlohmann::json;

#define BVH_MAX_DEPTH 16
#define BVH_MAX_LEAF_TRIS 4
#define RAY_DEPTH 64

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        buildBVH(); 

        return;
    }
    else if (ext == ".obj")
    {
        loadFromOBJ(filename);
        //set_up_cornell_box(20.0);
        set_up_default_lights(glm::vec3(1.0));
        // put the triangles in a bvh
        buildBVH(); // maxdepth is 16
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::set_up_camera_default(int resx, 
                                int resy, 
                                float fovy, 
                                int iterations,
                                int depth,
                                const std::string& fileName,
                                glm::vec3 camPos,
                                glm::vec3 camLookAt,
                                glm::vec3 camUp
)
{
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = resx;
    camera.resolution.y = resy;
    state.iterations = iterations;
    state.traceDepth = depth;
    state.imageName = fileName;

    camera.position = camPos;
    camera.lookAt = camLookAt;
    camera.up = camUp;

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

}

void Scene::set_up_default_lights(glm::vec3 color)
{
    // modify as necessary. There is a point and area light in here
    Light default_area_light;
    default_area_light.type = AREALIGHT;
    default_area_light.color = color;
    default_area_light.intensity = 10;

    default_area_light.translation = glm::vec3(0.0, 6.99, 5.0);
    default_area_light.rotation = glm::vec3(240.0, 0.0, 0.0);
    default_area_light.scale = glm::vec3(3.f, 0.01f, 3.f);

    default_area_light.transform = utilityCore::buildTransformationMatrix(
        default_area_light.translation, default_area_light.rotation, default_area_light.scale);
    default_area_light.inverseTransform = glm::inverse(default_area_light.transform);
    default_area_light.invTranspose = glm::inverseTranspose(default_area_light.transform);
    default_area_light.normal = glm::normalize(glm::vec3(default_area_light.transform[1]));
    lights.push_back(default_area_light);


   Light default_point_light;
    default_point_light.type = POINTLIGHT;
    default_point_light.pointLight.decay = 10.0;
    default_point_light.pointLight.range = 20.0;
    default_point_light.color = glm::vec3(0.3, 0.6, 0.9);
    default_point_light.intensity = 20;
    default_point_light.translation = glm::vec3(-5.0, 4.99, 0.0);
    default_point_light.rotation = glm::vec3(0.0, 0.0, 0.0);
    default_point_light.scale = glm::vec3(1.f);

    default_point_light.transform = utilityCore::buildTransformationMatrix(
        default_point_light.translation, default_point_light.rotation, default_point_light.scale);
    default_point_light.inverseTransform = glm::inverse(default_point_light.transform);
    default_point_light.invTranspose = glm::inverseTranspose(default_point_light.transform);
    default_point_light.normal = glm::normalize(glm::vec3(default_point_light.transform[1]));
    lights.push_back(default_point_light);

    Light light3;
    light3.type = AREALIGHT;
    light3.color = glm::vec3(0.8, 0.6, 0.3);
    light3.intensity = 20;

    light3.translation = glm::vec3(13.0, 5.99, 0.0);
    light3.rotation = glm::vec3(0.0, 20.0, 120.0);
    light3.scale = glm::vec3(5.f, 0.01f, 5.f);

    light3.transform = utilityCore::buildTransformationMatrix(
        light3.translation, light3.rotation, light3.scale);
    light3.inverseTransform = glm::inverse(light3.transform);
    light3.invTranspose = glm::inverseTranspose(light3.transform);
    light3.normal = glm::normalize(glm::vec3(light3.transform[1]));
    lights.push_back(light3);


}

void Scene::set_up_cornell_box(float scale)
{
    // room interior dimensions and wall thickness
    const float W = scale;
    const float H = scale;
    const float D = scale;
    const float T = scale;   // wall thickness

    // materials
    const int base = (int)materials.size();

    Material white{};
    white.color = glm::vec3(1.0);
    Material red{};
    red.color = glm::vec3(0.85f, 0.10f, 0.10f);
    Material green{};
    green.color = glm::vec3(0.10f, 0.85f, 0.10f);

    materials.push_back(white);
    materials.push_back(red);
    materials.push_back(green);
    const int WHITE = base + 0;
    const int RED = base + 1;
    const int GREEN = base + 2;

    const glm::vec3 noRot(0.f);

    // floor: top face at y = 0
    Geom floor_(CUBE, glm::vec3(0.f, -T / 2.f, 0.f), noRot, glm::vec3(W + 2 * T, T, D), WHITE);

    // ceiling: bottom face at y = H
    Geom ceiling(CUBE, glm::vec3(0.f, H + T / 2.f, 0.f), noRot, glm::vec3(W + 2 * T, T, D), WHITE);

    // left wall (red): inner face at x = -W/2
    Geom leftWall(CUBE, glm::vec3(-(W / 2.f + T / 2.f), H / 2.f, 0.f), noRot, glm::vec3(T, H, D), RED);

    // right wall (green): inner face at x = +W/2
    Geom rightWall(CUBE, glm::vec3(W / 2.f + T / 2.f, H / 2.f, 0.f), noRot, glm::vec3(T, H, D), GREEN);

    // back wall: inner face at z = -D/2, covers the edges of the other walls
    Geom backWall(CUBE, glm::vec3(0.f, H / 2.f, -(D / 2.f + T / 2.f)), noRot, glm::vec3(W + 2 * T, H + 2 * T, T), WHITE);

    geoms.push_back(floor_);
    geoms.push_back(ceiling);
    geoms.push_back(leftWall);
    geoms.push_back(rightWall);
    geoms.push_back(backWall);

    // modify as necessary. There is a point and area light in here
    Light default_area_light;
    default_area_light.type = AREALIGHT;
    default_area_light.color = glm::vec3(1.0);
    default_area_light.intensity = 20;

    default_area_light.translation = glm::vec3(0.0, scale - 1, 0.0);
    default_area_light.rotation = glm::vec3(180.0, 0.0, 0.0);
    default_area_light.scale = glm::vec3(scale/3.f, 0.01f, scale/ 3.f);

    default_area_light.transform = utilityCore::buildTransformationMatrix(
        default_area_light.translation, default_area_light.rotation, default_area_light.scale);
    default_area_light.inverseTransform = glm::inverse(default_area_light.transform);
    default_area_light.invTranspose = glm::inverseTranspose(default_area_light.transform);
    default_area_light.normal = glm::normalize(glm::vec3(default_area_light.transform[1]));
    lights.push_back(default_area_light);
}

void Scene::set_up_render_cam(Camera& camera, RenderState& state)
{
    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

}

void Scene::loadFromJSON(const std::string& jsonName)
{

    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& lightsData = data["Lights"];
    for (const auto& p : lightsData)
    {
        const auto& type = p["TYPE"];
        Light newLight;

        if (p["TYPE"] == "AREA")
        {
            newLight.type = AREALIGHT;
        }
        else if (p["TYPE"] == "POINT")
        {
            newLight.type = POINTLIGHT;
            newLight.pointLight.decay = p["DECAY"];
            newLight.pointLight.range = p["RANGE"];
        }
        const auto& col = p["RGB"];
        newLight.color = glm::vec3(col[0], col[1], col[2]);
        newLight.intensity = p["INTENSITY"];

        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newLight.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newLight.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newLight.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newLight.transform = utilityCore::buildTransformationMatrix(
        newLight.translation, newLight.rotation, newLight.scale);
        newLight.inverseTransform = glm::inverse(newLight.transform);
        newLight.invTranspose = glm::inverseTranspose(newLight.transform);
        newLight.normal = glm::normalize(glm::vec3(newLight.transform[1]));
        lights.push_back(newLight);
    }

    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom = type == "cube"? Geom(CUBE) : Geom(SPHERE);

        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    set_up_render_cam(camera, state);

}

void Scene::load_materials(std::vector<tinyobj::material_t> objmaterials)
{
    
    if (objmaterials.size() == 0)
    {
        Material defaultMat{}; // add a default material to the scene just in case ya know
        defaultMat.color = glm::vec3(0.8f);
        materials.push_back(defaultMat);
    }

    for (const auto& om : objmaterials)
    {
        Material mat{};
        mat.color = glm::vec3(om.diffuse[0], om.diffuse[1], om.diffuse[2]);

        float emission = glm::max(om.emission[0], glm::max(om.emission[1], om.emission[2]));
        glm::vec3 specular = glm::vec3(om.specular[0], om.specular[1], om.specular[2]);
        glm::vec3 tf = glm::vec3(om.transmittance[0], om.transmittance[1], om.transmittance[2]);

        if (emission > 0.f)
        {
            mat.color = glm::vec3(om.emission[0], om.emission[1], om.emission[2]) / emission;
            mat.emittance = emission;
        }
        else if (glm::dot(tf, tf) > 0.f) // refers to dielectric stuff
        {
            mat.isDielectric = 1.f;
            mat.indexOfRefraction = (om.ior > 1.f) ? om.ior : 1.5f; // set to 1.5 by default unless specified, or less than 1

        } else if (glm::length(specular) > 0.01f)
        {
            mat.hasReflective = 1.f;
        }

        materials.push_back(mat);
    }
}

void Scene::populateBuffers(const tinyobj::attrib_t& attrib, 
    std::vector<glm::vec3>& vertices, 
    std::vector<glm::vec2>& uvs)
{

    for (int i = 0; i < attrib.vertices.size(); i+= 3)
    {
        float v1 = attrib.vertices[i];
        float v2 = attrib.vertices[i + 1];
        float v3 = attrib.vertices[i + 2];
        vertices.push_back(glm::vec3(v1, v2, v3));
    }
    
    for (int i = 0; i < attrib.texcoords.size(); i += 2)
    {
        float uv1 = attrib.texcoords[i];
        float uv2 = attrib.texcoords[i + 1];
        uvs.push_back(glm::vec2(uv1, uv2));
    }

}

void Scene::load_triangles(const tinyobj::shape_t& shape, 
    const std::vector<glm::vec3>& vertices,
    const std::vector<glm::vec2>& uvs)
{
    printf("shape here");
    const vector<tinyobj::index_t> & indices = shape.mesh.indices;
    const vector<int> & mat_ids = shape.mesh.material_ids;
    std::cout << "Loading " << mat_ids.size() << " triangles..." << std::endl;
    // populate with face data
    for (size_t faceidx = 0; faceidx < mat_ids.size(); faceidx++)
    {

        //vertex positions
        int vidx1 = indices[3 * faceidx].vertex_index;
        int vidx2 = indices[3 * faceidx + 1].vertex_index;
        int vidx3 = indices[3 * faceidx + 2].vertex_index;

        glm::vec3 v1 = vertices[vidx1];
        glm::vec3 v2 = vertices[vidx2];
        glm::vec3 v3 = vertices[vidx3];

        Geom t = Geom(v1, v2, v3);

        // uvs
        if (uvs.size() > 0) { 
            int uvidx1 = indices[3 * faceidx].texcoord_index;
            int uvidx2 = indices[3 * faceidx + 1].texcoord_index;
            int uvidx3 = indices[3 * faceidx + 2].texcoord_index;

            t.uv1 = uvs[uvidx1];
            t.uv2 = uvs[uvidx2];
            t.uv3 = uvs[uvidx3];
            t.hasUVs = true;
        }
        
        // material
        if (mat_ids[faceidx] != -1) // else it is 0 by default
            t.materialid = mat_ids[faceidx];
      
        geoms.push_back(t);
    }
}

void Scene::loadFromOBJ(const std::string& filenameOBJ)
{
    tinyobj::ObjReaderConfig reader_config;
    std::filesystem::path objPath(filenameOBJ);
    reader_config.mtl_search_path =
        objPath.has_parent_path()
        ? objPath.parent_path().string() + "/"
        : "./";
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filenameOBJ, reader_config)) {
    if (!reader.Error().empty()) {
        std::cerr << "TinyObjReader: " << reader.Error();
    }
    exit(1);
    }

    if (!reader.Warning().empty()) {
    std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& objmaterials = reader.GetMaterials();

    load_materials(objmaterials);

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> uvs;

    // populate a vertex, normal and uv buffer for triangles
    populateBuffers(attrib, vertices, uvs);

    printf("size of vertices buffer %d, uvs, %d", vertices.size(), uvs.size());

    // populate the geoms buffer with geom structs, per object in the scene
    for (auto shape = shapes.begin(); shape < shapes.end(); shape++)
        load_triangles(*shape, vertices, uvs);
    
    // hardcoded lights and camera in scene

    set_up_camera_default(1000, 800, 45.f, 5000, RAY_DEPTH, filenameOBJ, 
                                glm::vec3(0.f, 0.f, 7.5),
                                glm::vec3(0.f, 2.f, 0.f),
                                glm::vec3(0.f, 1.f, 0.f));

    //required for display: set up render camera stuff
    set_up_render_cam(state.camera, state);
    printf("number of triangles in the scene: %d", geoms.size());
}

void Scene::buildBVH() {
    nodes.clear();
    std::vector<Geom*> tris;
    for(auto& g : geoms) 
        if (g.type == TRIANGLE) tris.push_back(&g);
    
    if (tris.empty()) { // no tris, no bvh
        bvhRootIdx = -1;
        return;
    }
    int numLeafNodes = 0;
    bvhRootIdx = recursiveBVHBuild(tris, 0, tris.size(), 0, &numLeafNodes);

    // reorder geoms so that position i == tris[i]
    std::vector<Geom> reorderedGeoms;
    reorderedGeoms.reserve(geoms.size());
    for (Geom* t : tris) reorderedGeoms.push_back(*t);
    for (auto& g : geoms)
        if (g.type != TRIANGLE) reorderedGeoms.push_back(g); // push back non triangle data last
    geoms = std::move(reorderedGeoms);

        std::cout << "Number of triangles in mesh: " << tris.size() << std::endl;
    std::cout << "Number of leaf nodes: " << numLeafNodes << std::endl;

}

BVHBounds Scene::Union(const BVHBounds& a, const BVHBounds &b) {
    glm::vec3 maxvec = glm::max(a.maxCorner, b.maxCorner);
    glm::vec3 minvec = glm::min(a.minCorner, b.minCorner);

    BVHBounds ab = BVHBounds(minvec, maxvec);
    return ab;
}

// TODO: make a depth limit. need to implement start and end indices and test it

int Scene::recursiveBVHBuild(std::vector<Geom*> &triangles, int start, int end, int depth, int* numLeafNodes)
{
    int nodeIdx = (int)nodes.size();
    nodes.emplace_back();

    BVHBounds currentLayerBounds(glm::vec3(FLT_MAX),  glm::vec3(-FLT_MAX));

    // build up our current bounding box
    for (int i = start; i < end; i++)
        currentLayerBounds = Union(triangles[i]->bbox, currentLayerBounds);
    
    // theres only one triangle to consider, or we reached maxdepth, so build a leaf node
    int count = end - start;
    if (count <= BVH_MAX_LEAF_TRIS || depth >= BVH_MAX_DEPTH)
    {
        nodes[nodeIdx].isLeaf = true;
        nodes[nodeIdx].tri_start = start;
        nodes[nodeIdx].tri_count = count;
        nodes[nodeIdx].bbox = currentLayerBounds;
        nodes[nodeIdx].child_L = -1;
        nodes[nodeIdx].child_R = -1;

        (*numLeafNodes)++;
        return nodeIdx;
    }

    // recursive case
     
    // find longest axis to split on
    glm::vec3 extent = currentLayerBounds.maxCorner - currentLayerBounds.minCorner;

    int splitAxis = 0;
    if (extent.y > extent.x && extent.y > extent.z)
        splitAxis = 1;
    else if (extent.z > extent.x && extent.z > extent.y)
        splitAxis = 2;
    
    int midIdx = (start + end) / 2;

    // choose a median split point (along longest axis)
    std::nth_element(triangles.begin() + start,
                    triangles.begin() + midIdx,
                    triangles.begin() + end,
                    [splitAxis](const Geom *a, const Geom *b) {
                             return a->centroid[splitAxis] < b->centroid[splitAxis];
                            }
                    );
    //recurse
    int childLidx = recursiveBVHBuild(triangles, start, midIdx, depth + 1,numLeafNodes);
    int childRidx = recursiveBVHBuild(triangles, midIdx, end, depth + 1, numLeafNodes);

    // build up our current node
    nodes[nodeIdx].child_L = childLidx;
    nodes[nodeIdx].child_R = childRidx;
    nodes[nodeIdx].bbox = currentLayerBounds;
    nodes[nodeIdx].isLeaf = false;

    return nodeIdx;
}