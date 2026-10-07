#pragma once

#include "sceneStructs.h"
#include "tiny_obj_loader.h"

#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void set_up_camera_default(int resx,
        int resy,
        float fovy,
        int iterations,
        int depth,
        const std::string& fileName,
        glm::vec3 camPos,
        glm::vec3 camLookAt,
        glm::vec3 camUp
    );
    void set_up_cornell_box(float scale);
    void set_up_default_lights(glm::vec3 color);
    void set_up_render_cam(Camera& camera, RenderState& state);
    void populateBuffers(const tinyobj::attrib_t& attrib,
        std::vector<glm::vec3>& vertices,
        std::vector<glm::vec2>& uvs);

        // populates geoms
    void load_triangles(const tinyobj::shape_t& shape,
        const std::vector<glm::vec3>& vertices,
        const std::vector<glm::vec2>& uvs);

        // populates materials
    void load_materials(std::vector<tinyobj::material_t> objmaterials);
    void loadFromOBJ(const std::string& filenameOBJ);

    
    //essentially this populates the nodes std::vector
    void buildBVH();
    int recursiveBVHBuild(std::vector<Geom*> &triangles, int start, int end, int depth, int* numLeafNodes);
    BVHBounds Union(const BVHBounds& a, const BVHBounds &b);


public:
    Scene(std::string filename);


    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Light> lights;
    std::vector<BVHNode> nodes;
    int bvhRootIdx;

    RenderState state;
};
