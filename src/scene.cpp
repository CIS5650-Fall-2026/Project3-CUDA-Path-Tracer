#include "scene.h"

#define TINYGLTF3_ENABLE_FS

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

bool Scene::loadModelGLTF(const std::string& filename, tg3_model& model)
{
    tg3_error_stack errors;
    tg3_error_stack_init(&errors);

    tg3_parse_options options;
    tg3_parse_options_init(&options);

    tg3_error_code result = tg3_parse_file(
        &model,
        &errors,
        filename.c_str(),
        static_cast<uint32_t>(filename.size()),
        &options
    );

    if (result != TG3_OK)
    {
        std::cerr << "Failed to load glTF: "
            << filename << std::endl;

        for (uint32_t i = 0; i < errors.count; i++)
        {
            std::cerr << "[" << (int)errors.entries[i].severity << "] "
                << (errors.entries[i].message
                    ? errors.entries[i].message
                    : "(null)")
                << std::endl;
        }

        tg3_error_stack_free(&errors);
        return false;
    }

    tg3_error_stack_free(&errors);
    return true;
}

std::vector<glm::vec3> loadVertexDataFromModel(
    const tg3_model& model,
    const tg3_primitive& primitive)
{
    std::vector<glm::vec3> vertices;

    int32_t positionAccessorIndex = -1;

    for (uint32_t a = 0; a < primitive.attributes_count; a++)
    {
        const tg3_str_int_pair& attribute = primitive.attributes[a];

        if (strcmp(attribute.key.data, "POSITION") == 0)
        {
            positionAccessorIndex = attribute.value;
            break;
        }
    }

    if (positionAccessorIndex < 0)
        return vertices;

    const tg3_accessor& accessor =
        model.accessors[positionAccessorIndex];

    const tg3_buffer_view& view =
        model.buffer_views[accessor.buffer_view];

    const tg3_buffer& buffer =
        model.buffers[view.buffer];

    const float* data = reinterpret_cast<const float*>(
        buffer.data.data + view.byte_offset + accessor.byte_offset);

    vertices.resize(accessor.count);

    for (uint32_t i = 0; i < accessor.count; i++)
    {
        vertices[i] = glm::vec3(
            data[i * 3],
            data[i * 3 + 1],
            data[i * 3 + 2]
        );
    }

    return vertices;
}
std::vector<uint32_t> loadIndexDataFromModel(
    const tg3_model& model,
    const tg3_primitive& primitive)
{
    std::vector<uint32_t> indices;

    if (primitive.indices < 0)
        return indices;

    const tg3_accessor& accessor =
        model.accessors[primitive.indices];

    const tg3_buffer_view& view =
        model.buffer_views[accessor.buffer_view];

    const tg3_buffer& buffer =
        model.buffers[view.buffer];

    const uint32_t* data = reinterpret_cast<const uint32_t*>(
        buffer.data.data + view.byte_offset + accessor.byte_offset);

    indices.resize(accessor.count);

    for (uint32_t i = 0; i < accessor.count; i++)
        indices[i] = data[i];

    return indices;
}

BoundingBox Union(const BoundingBox& b1, const BoundingBox& b2) {
    BoundingBox result;
    result.min = glm::min(b1.min, b2.min);
    result.max = glm::max(b1.max, b2.max);
    return result;
}

BoundingBox GetTriangleBoundingBox(const Triangle& triangle) {
    BoundingBox bbox;
    bbox.min = glm::min(glm::min(triangle.v0, triangle.v1), triangle.v2);
    bbox.max = glm::max(glm::max(triangle.v0, triangle.v1), triangle.v2);
    return bbox;
}

//returns id of node
int recursiveBVHBuild(int start, int end, BVHTree& bvh) {

    BVHNode node;
    node.leftChild = -1;
    node.rightChild = -1;
    node.triangleIndex = -1;
    node.isLeaf = false;

    // case leaf node
    if (end - start <= 1) {
        node.isLeaf = true;
        node.triangleIndex = start;
		node.geomIndex = -1;
		node.materialId = bvh.materialId;
        node.bbox = GetTriangleBoundingBox(bvh.triangles[start]);
        bvh.nodes.push_back(node);
        int nodeId = bvh.nodes.size() - 1;
        return nodeId;
    }

    // case normal node
    else {

        // get bounding box for this node
        BoundingBox bbox = GetTriangleBoundingBox(bvh.triangles[start]);
        for (int i = start + 1; i < end; ++i) {
            bbox = Union(bbox, GetTriangleBoundingBox(bvh.triangles[i]));
        }
        node.bbox = bbox;

        //sort triangles by centroid along the longest axis
        glm::vec3 extent = bbox.max - bbox.min;
        int axis = 0;
        if (extent.y > extent.x && extent.y > extent.z) axis = 1;
        else if (extent.z > extent.x && extent.z > extent.y) axis = 2;

        std::sort(bvh.triangles.begin() + start, bvh.triangles.begin() + end,
            [axis](const Triangle& a, const Triangle& b)
            {
                glm::vec3 ca = (a.v0 + a.v1 + a.v2) / 3.0f;
                glm::vec3 cb = (b.v0 + b.v1 + b.v2) / 3.0f;

                return ca[axis] < cb[axis];
            });

		// recursively build left and right children
        int mid = (start + end) / 2;
        node.leftChild = recursiveBVHBuild(start, mid, bvh);
        node.rightChild = recursiveBVHBuild(mid, end, bvh);

        bvh.nodes.push_back(node);
        int nodeId = bvh.nodes.size() - 1;

		return nodeId;

    }
}

int recursiveTopLevelBVHBuild(
    int start,
    int end,
    std::vector<int>& rootNodesIndices,
    std::vector<BVHNode>& bvhNodes)
{
    // Leaf: this node is already a complete BVH/Geom
    if (end - start == 1)
        return rootNodesIndices[start];

    // Find bounding box containing all root nodes
    BoundingBox bbox =
        bvhNodes[rootNodesIndices[start]].bbox;

    for (int i = start + 1; i < end; ++i)
    {
        bbox = Union(
            bbox,
            bvhNodes[rootNodesIndices[i]].bbox
        );
    }

    // Find longest axis
    glm::vec3 extent = bbox.max - bbox.min;

    int axis = 0;
    if (extent.y > extent.x && extent.y > extent.z)
        axis = 1;
    else if (extent.z > extent.x && extent.z > extent.y)
        axis = 2;

    // Sort root nodes by their bounding-box centroid
    std::sort(
        rootNodesIndices.begin() + start,
        rootNodesIndices.begin() + end,
        [axis, &bvhNodes](int a, int b)
        {
            glm::vec3 ca =
                (bvhNodes[a].bbox.min + bvhNodes[a].bbox.max) * 0.5f;

            glm::vec3 cb =
                (bvhNodes[b].bbox.min + bvhNodes[b].bbox.max) * 0.5f;

            return ca[axis] < cb[axis];
        }
    );

    int mid = (start + end) / 2;

    int leftChild = recursiveTopLevelBVHBuild(
        start, mid, rootNodesIndices, bvhNodes);

    int rightChild = recursiveTopLevelBVHBuild(
        mid, end, rootNodesIndices, bvhNodes);

    // Create new internal node
    BVHNode node;
    node.bbox = bbox;
    node.leftChild = leftChild;
    node.rightChild = rightChild;
    node.triangleIndex = -1;
    node.geomIndex = -1;
    node.materialId = -1;
    node.isLeaf = false;

    int nodeId = bvhNodes.size();
    bvhNodes.push_back(node);

    return nodeId;
}

void buildBVH(BVHTree& bvh) {

	bvh.nodes.clear();
	int rootId = recursiveBVHBuild(0, bvh.triangles.size(), bvh);
	bvh.rootNodeIndex = rootId;

}

// Helper function to compute the bounding box of any given Geom object
BoundingBox GetGeomBoundingBox(const Geom& geom)
{
    glm::vec3 localMin;
    glm::vec3 localMax;

    if (geom.type == TRIANGLE)
    {
        localMin = glm::min(
            glm::min(geom.vertices[0], geom.vertices[1]),
            geom.vertices[2]);

        localMax = glm::max(
            glm::max(geom.vertices[0], geom.vertices[1]),
            geom.vertices[2]);
    }
    else
    {
        // Cube and sphere both have .5, .5
        localMin = glm::vec3(-0.5f);
        localMax = glm::vec3(0.5f);
    }

    glm::vec3 corners[8] =
    {
        glm::vec3(localMin.x, localMin.y, localMin.z),
        glm::vec3(localMax.x, localMin.y, localMin.z),
        glm::vec3(localMin.x, localMax.y, localMin.z),
        glm::vec3(localMax.x, localMax.y, localMin.z),

        glm::vec3(localMin.x, localMin.y, localMax.z),
        glm::vec3(localMax.x, localMin.y, localMax.z),
        glm::vec3(localMin.x, localMax.y, localMax.z),
        glm::vec3(localMax.x, localMax.y, localMax.z)
    };

    BoundingBox bbox;
    bbox.min = glm::vec3(FLT_MAX);
    bbox.max = glm::vec3(-FLT_MAX);

    for (int i = 0; i < 8; ++i)
    {
        glm::vec3 p =
            glm::vec3(geom.transform * glm::vec4(corners[i], 1.0f));

        bbox.min = glm::min(bbox.min, p);
        bbox.max = glm::max(bbox.max, p);
    }

    return bbox;
}

std::vector<glm::vec3> loadNormalDataFromModel(
    const tg3_model& model,
    const tg3_primitive& primitive)
{
    std::vector<glm::vec3> normals;

    int32_t normalAccessorIndex = -1;

    for (uint32_t a = 0; a < primitive.attributes_count; a++)
    {
        const tg3_str_int_pair& attribute = primitive.attributes[a];

        if (strcmp(attribute.key.data, "NORMAL") == 0)
        {
            normalAccessorIndex = attribute.value;
            break;
        }
    }

    if (normalAccessorIndex < 0)
        return normals;

    const tg3_accessor& accessor =
        model.accessors[normalAccessorIndex];

    const tg3_buffer_view& view =
        model.buffer_views[accessor.buffer_view];

    const tg3_buffer& buffer =
        model.buffers[view.buffer];

    const float* data = reinterpret_cast<const float*>(
        buffer.data.data +
        view.byte_offset +
        accessor.byte_offset);

    normals.resize(accessor.count);

    for (uint32_t i = 0; i < accessor.count; i++)
    {
        normals[i] = glm::vec3(
            data[i * 3],
            data[i * 3 + 1],
            data[i * 3 + 2]
        );
    }

    return normals;
}


void Scene::loadFromJSON(const std::string& jsonName)
{

    // bvh on / off setting
    bool useBVH = USE_BVH;

    std::vector<BVHTree> bvhTrees;

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
            newMaterial.type = DIFFUSE;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting" || p["TYPE"] == "Emissive")
        {
            newMaterial.type = EMISSIVE;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular" || p["TYPE"] == "Reflective")
        {
            newMaterial.type = REFLECTIVE;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Refractive" || p["TYPE"] == "Transmissive")
        {
            newMaterial.type = REFRACTIVE;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.indexOfRefraction = p["IOR"];
        }
        else if (p["TYPE"] == "Dielectric")
        {
            newMaterial.type = DIELECTRIC;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.indexOfRefraction = p["IOR"];
        }
        else if (p["TYPE"] == "Microfacet")
        {
            newMaterial.type = MICROFACET;
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& modelData = data["Models"];
    const auto& primitiveData = data["Primitives"];

	// Unpack models into primitive objects
    for (const auto& m : modelData)
    {

		BVHTree tree;
        tree.materialId = MatNameToID[m["MATERIAL"]];

		const auto& modelType = m["TYPE"];
		const auto& modelFile = m["FILE"].get<string>();

        if (modelType == "gltf") {

			// Load glTF model
            tg3_model model{};
			std::string filePath = "..\\..\\..\\models\\" + modelFile + "\\scene.gltf";

            if (loadModelGLTF(filePath, model))
            {
                std::cout << "Loaded \"" << filePath << "\"!" << std::endl;

                for (uint32_t mo = 0; mo < model.meshes_count; mo++)
                {
                    const tg3_mesh& mesh = model.meshes[mo];

                    for (uint32_t p = 0; p < mesh.primitives_count; p++)
                    {
                        const tg3_primitive& primitive = mesh.primitives[p];

						std::vector<glm::vec3> vertices = loadVertexDataFromModel(model, primitive);
						std::vector<uint32_t> indices = loadIndexDataFromModel(model, primitive);
						std::vector<glm::vec3> normals = loadNormalDataFromModel(model, primitive);

                        
                        for (uint32_t i = 0; i < indices.size(); i += 3)
                        {

                            glm::vec3 v0 = vertices[indices[i]];
                            glm::vec3 v1 = vertices[indices[i + 1]];
                            glm::vec3 v2 = vertices[indices[i + 2]];

                            if (useBVH) {
                                
                                const auto& trans = m["TRANS"];
                                const auto& rotat = m["ROTAT"];
                                const auto& scale = m["SCALE"];

                                glm::vec3 translation(trans[0], trans[1], trans[2]);
                                glm::vec3 rotation(rotat[0], rotat[1], rotat[2]);
                                glm::vec3 scaling(scale[0], scale[1], scale[2]);

                                glm::mat4 transform =
                                    utilityCore::buildTransformationMatrix(
                                        translation,
                                        rotation,
                                        scaling);

                                tree.triangles.push_back({
                                    glm::vec3(transform * glm::vec4(v0, 1.0f)),
                                    glm::vec3(transform * glm::vec4(v1, 1.0f)),
                                    glm::vec3(transform * glm::vec4(v2, 1.0f)),
                                    });

                            }
                            else {

                                Geom newGeom;
                                newGeom.type = TRIANGLE;
                                newGeom.vertices[0] = v0;
                                newGeom.vertices[1] = v1;
                                newGeom.vertices[2] = v2;
                                newGeom.materialid = MatNameToID[m["MATERIAL"]];
                                const auto& trans = m["TRANS"];
                                const auto& rotat = m["ROTAT"];
                                const auto& scale = m["SCALE"];
                                newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
                                newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
                                newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
                                newGeom.transform = utilityCore::buildTransformationMatrix(
                                newGeom.translation, newGeom.rotation, newGeom.scale);
                                newGeom.inverseTransform = glm::inverse(newGeom.transform);
                                newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

                                geoms.push_back(newGeom);

                            }
                        }
                    }
                }

            }

        }

        if (useBVH) {

			buildBVH(tree);
            bvhTrees.push_back(tree);

		}
    }

	// Load Primitive Objects
    for (const auto& p : primitiveData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
		else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "triangle")
        {
            newGeom.type = TRIANGLE;

            const auto& verts = p["VERTS"];
            newGeom.vertices[0] = glm::vec3(
                verts[0][0], verts[0][1], verts[0][2]);
            newGeom.vertices[1] = glm::vec3(
                verts[1][0], verts[1][1], verts[1][2]);
            newGeom.vertices[2] = glm::vec3(
                verts[2][0], verts[2][1], verts[2][2]);
        }
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

    // Combine all geom and bvh trees into a single list for rendering
    if (useBVH) {


		// combine all BVH trees into a single list of triangle and bvh nodes
		// keep track of the offset for each tree's nodes and triangles
		// keep track of all roots and their material ids (is now a node attribute)
        std::vector <int> rootNodesIndices;
        int nodeOffset = 0;
		int triangleOffset = 0;

        for (const auto& tree : bvhTrees) {

            triangles.insert(triangles.end(), tree.triangles.begin(), tree.triangles.end());

            for (const auto& oldNode : tree.nodes)
            {
                BVHNode node = oldNode;

                if (node.leftChild != -1)
                    node.leftChild += nodeOffset;

                if (node.rightChild != -1)
                    node.rightChild += nodeOffset;

                if (node.triangleIndex != -1)
                    node.triangleIndex += triangleOffset;

                bvhNodes.push_back(node);
            }

            BVHNode root = tree.nodes[tree.rootNodeIndex];

			rootNodesIndices.push_back(bvhNodes.size() - tree.nodes.size() + tree.rootNodeIndex);

            nodeOffset += tree.nodes.size();
            triangleOffset += tree.triangles.size();
        
        }

        // create nodes as each individual geom
        for (int i = 0; i < geoms.size(); ++i)
        {
            const auto& geom = geoms[i];

            BVHNode node;
            node.bbox = GetGeomBoundingBox(geom);
            node.leftChild = -1;
            node.rightChild = -1;
            node.triangleIndex = -1;
            node.geomIndex = i;
            node.materialId = geom.materialid;
            node.isLeaf = true;

            bvhNodes.push_back(node);
			rootNodesIndices.push_back(bvhNodes.size() - 1);
        }

		// recursively build a top-level BVH for all root nodes
        bvhRootNodeIndex = recursiveTopLevelBVHBuild( 0, rootNodesIndices.size(), rootNodesIndices, bvhNodes);

	}

	// Load Camera and Render State
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

    if (cameraData.contains("FOCALDISTANCE"))
    {
        camera.focalDistance = cameraData["FOCALDISTANCE"];
    }
    else
    {
        camera.focalDistance = 1.0f;
	}

    if (cameraData.contains("LENSRADIUS"))
    {
        camera.lensRadius = cameraData["LENSRADIUS"];
    }
    else
    {
        camera.lensRadius = 0.0f;
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

}
