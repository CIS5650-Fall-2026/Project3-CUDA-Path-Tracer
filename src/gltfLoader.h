#pragma once

#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <glm/glm.hpp>

struct EnvironmentData {
	int width = 0;
	int height = 0;

	//RGBA
	std::vector<float> pixels;
	//CDF(cyumulative distirbution function, stores a running total of probabilities
	// if
	// A B C
	// 2 5 3
	// 0.2 0.5 0.3
	// [0.0, 0.2) [0.2, 0.7) [0.7, 1.0)
	// CDF array: 0.0, 0.2, 0.7, 1.0
	// generate a random number between 0 and one: 0.13 -> pixel A, 0.45 -> pixel B, 0.92 -> pixelC
	// Pixel i's probability is cdf[i+1] - cdf[i]
	std::vector<double> cdf;
};

//image = pixel data
//texture = tells us which iamge to see and how to sample it
//both share 1 image
struct ImageData {
	int width = 0;
	int height = 0;
	int channels = 0;
	int bitsPerChannel = 0;

	std::vector<unsigned char> pixels;
};

struct TextureData {
	int imageIndex = -1;

	//Repeat along u and v
	int wrapS = 10497; //controls what happens outside the image's UV range
	int wrapT = 10497;

	int minFilter = -1; //controls how pixels are combined when sampling
	int magFilter = -1;
};

struct MeshData {
	std::vector<glm::vec3> positions; //World space
	std::vector<glm::vec3> normals;
	std::vector<std::array<std::uint32_t, 3>> triangles;
	std::vector<glm::vec2> texcoords;

	//Index into the glTF material array, not the renderer's array
	int gltfMaterialIndex = -1;
	int gltfBaseColorTextureIndex = -1;
	//Index into renderer's material array
	int rendererMaterialId = -1;
	float metallicFactor = 1.0f;
	float roughnessFactor = 1.0f;
	glm::vec4 baseColorFactor = glm::vec4(1.0f);

	std::vector<glm::vec4> tangents; //world-space xyz
	int gltfMetallicRoughnessTextureIndex = -1;
	int gltfNormalTextureIndex = -1;
	float normalScale = 1.0f;
};

bool loadGltf(const std::string& filename, std::vector<MeshData>& output, std::vector<ImageData>& outputImages, std::vector<TextureData>& outputTextures);
bool loadEnvironment(const std::string& filename, EnvironmentData& output);


//Box.gltf for reference
//{
//    "asset": {
//        "generator": "COLLADA2GLTF",
//            "version" : "2.0"
//    },
//        "scene" : 0,
//        "scenes" : [
//    {
//        "nodes": [
//            0
//        ]
//    }
//        ] ,
//        "nodes": [
//    {
//        "children": [
//            1
//        ] ,
//            "matrix" : [
//                1.0,
//                0.0,
//                0.0,
//                0.0,
//                0.0,
//                0.0,
//                -1.0,
//                0.0,
//                0.0,
//                1.0,
//                0.0,
//                0.0,
//                0.0,
//                0.0,
//                0.0,
//                1.0
//            ]
//    },
//        {
//            "mesh": 0
//        }
//        ],
//        "meshes": [
//    {
//        "primitives": [
//        {
//            "attributes": {
//                "NORMAL": 1, //first element of attributes is normal?
//                    "POSITION" : 2 //2nd element of attributes is position?
//            },
//                "indices" : 0,
//                "mode" : 4,
//                "material" : 0
//        }
//        ] ,
//            "name": "Mesh"
//    }
//        ],
//        "accessors": [ //accessor
//    {
//        "bufferView": 0, // accessor.at(0) = indices?
//            "byteOffset" : 0,
//            "componentType" : 5123,
//            "count" : 36,
//            "max" : [
//                23
//            ] ,
//            "min" : [
//                0
//            ] ,
//            "type" : "SCALAR"
//    },
//        {
//            "bufferView": 1, // accessor.at(1) = normals
//            "byteOffset" : 0,
//            "componentType" : 5126,
//            "count" : 24,
//            "max" : [
//                1.0,
//                1.0,
//                1.0
//            ] ,
//            "min" : [
//                -1.0,
//                -1.0,
//                -1.0
//            ] ,
//            "type" : "VEC3"
//        },
//        {
//            "bufferView": 1, // accesor.at(2) = position
//            "byteOffset" : 288,
//            "componentType" : 5126,
//            "count" : 24,
//            "max" : [
//                0.5,
//                0.5,
//                0.5
//            ] ,
//            "min" : [
//                -0.5,
//                -0.5,
//                -0.5
//            ] ,
//            "type" : "VEC3"
//        }
//        ],
//        "materials": [
//    {
//        "pbrMetallicRoughness": {
//            "baseColorFactor": [
//                0.800000011920929,
//                0.0,
//                0.0,
//                1.0
//            ] ,
//                "metallicFactor": 0.0
//        },
//            "name": "Red"
//    }
//        ],
//        "bufferViews": [ //what is this
	//    {
	//			"buffer": 0,
	//            "byteOffset" : 576,
	//            "byteLength" : 72,
	//            "target" : 34963
	//    },
//        {
//            "buffer": 0,
//            "byteOffset" : 0,
//            "byteLength" : 576,
//            "byteStride" : 12,
//            "target" : 34962
//        }
//        ],
//        "buffers": [
//    {
//        "byteLength": 648,
//            "uri" : "Box0.bin"
//    }
//        ]
//}
