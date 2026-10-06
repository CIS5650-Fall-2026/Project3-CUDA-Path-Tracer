#pragma once

#include "sceneStructs.h"
#include <vector>

    // need forward declaration
    struct tg3_model;
    struct tg3_primitive;
    struct tg3_accessor;

class Scene
{
private:
    // Index struct to support different indices for vtx/normal/texcoord.
    // -1 means not used.
    struct index_t {
        int vertex_index;
        int normal_index;
        int texcoord_index;
    };

    struct mesh_t {
        std::vector<float> vertices;
        std::vector<float> normals;
        std::vector<float> texcoords;
        std::vector<index_t> indices;
        std::vector<unsigned int>
            num_face_vertices;          // The number of vertices per
                                        // face. 3 = triangle, 4 = quad, ...
        std::vector<int> material_ids;  // per-face material ID
        std::vector<unsigned int> smoothing_group_ids;  // per-face smoothing group
                                                        // ID(0 = off. positive value
                                                        // = group id)
    };

    mesh_t mesh;
    
    void loadFromJSON(const std::string& jsonName);
    void loadFromGLTF(const std::string& gltfName);
    void loadFromOBJ(const std::string& objName);

    // helpers for gltf parsing (i hate this)
    static int findAttribute(const tg3_primitive &primitive, const char *name);
    
    static const uint8_t* getAccessorData(
        const tg3_model &model,
        const tg3_accessor &accessor,
        int32_t &stride
    );

    static int appendFloatAttribute(
        const tg3_model &model,
        int accessorIdx,
        int componentCount,
        std::vector<float> &dest
    );

    unsigned long int appendGLTFIndices(
        const tg3_model &model,
        const tg3_primitive &primitive,
        int vertex_offset,
        int normal_offset,
        int texcoord_offset
    );

    // obj/gltf mesh rendering
    void appendMeshToRender();
    void initMeshCamera();

    // procedural shell rendering
    glm::vec3 shellPoint(float u, float v);
    glm::vec3 vasePoint(float u, float v);
    void addProceduralTriangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, int materialId);
    void generateShell(int materialId, const glm::mat4& transform);
    void generateVase(int materialId, const glm::mat4& transform);

public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    std::vector<uchar4> texPixels;
};
