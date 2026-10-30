// Compile TinyGLTF's implementation in this file only
#define TINYGLTF_IMPLEMENTATION

// Load images, but don't need TinyGLTF to write images
#define TINYGLTF_NO_STB_IMAGE_WRITE


#include "tiny_gltf.h"
#include "gltfLoader.h"
#include <iostream>
#include <cstring> //for std::memcpy
#include <cstdint> //std::uint16_t
#include <vector>
#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stdexcept>
#include <utility>
#include <cmath>

struct MeshInstance {
	int meshIndex;
	glm::mat4 worldTransform;
};

glm::mat4 getLocalTransform(const tinygltf::Node& node) {
	if (!node.matrix.empty()) {
		if (node.matrix.size() != 16) {
			throw std::runtime_error("Invalid node matrix size");
		}

		return glm::mat4(glm::make_mat4(node.matrix.data()));
	}

	glm::vec3 translation(0.0f);
	glm::vec3 scale(1.0f);
	glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);

	if (!node.translation.empty()) {
		translation = glm::vec3(node.translation.at(0), node.translation.at(1), node.translation.at(2));
	}

	if (!node.scale.empty()) {
		scale = glm::vec3(node.scale.at(0), node.scale.at(1), node.scale.at(2));
	}

	//gltf: xyzw, glm: wxyz
	if (!node.rotation.empty()) {
		rotation = glm::quat(static_cast<float>(node.rotation.at(3)), static_cast<float>(node.rotation.at(0)), static_cast<float>(node.rotation.at(1)), static_cast<float>(node.rotation.at(2)));
	}

	return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
}

void traverseMesh(const tinygltf::Model& model, int nodeIndex, const glm::mat4& parentWorld, std::vector <MeshInstance>& instances) {
	const tinygltf::Node& node = model.nodes.at(nodeIndex);

	const glm::mat4 world = parentWorld * getLocalTransform(node);

	if (node.mesh >= 0) {
		instances.push_back({ node.mesh, world });
	}

	for (int childIndex : node.children) {
		traverseMesh(model, childIndex, world, instances);
	}
}

//Current texture import limits:
//Every image must be 8 bit RGBA.
//Every texture uses sRGB decoding.
//Every texture uses linear filtering, regardless of the glTF filter settings.

static void validateMaterialTexture(const tinygltf::Model& model, const MeshData& mesh,
	int textureIndex, int texCoord, const char* label) {
	//ignore unused textures
	if (textureIndex < 0) return;
	//check if texture index is larger than total number of textures
	if (size_t(textureIndex) >= model.textures.size()) {
		throw std::runtime_error(std::string(label) + ": invalid texture index");
	}
	//texture must be mapped using the first UV channel - TEXCOORD_0, this means no complicated flattening/remapping
	if (texCoord != 0 || mesh.texcoords.size() != mesh.positions.size()) {
		throw std::runtime_error(std::string(label) + ": requires TEXCOORD_0");
	}
}

static void loadMeshTangents(const tinygltf::Model& model, const tinygltf::Primitive& primitive, 
	const glm::mat4& worldTransform, MeshData& mesh) {
	auto it = primitive.attributes.find("TANGENT");
	if (it == primitive.attributes.end()) {
		return;
	}

	const tinygltf::Accessor& acc = model.accessors.at(it->second);
	if (acc.type != TINYGLTF_TYPE_VEC4 || acc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
		acc.normalized || acc.sparse.isSparse || acc.bufferView < 0 ||
		acc.count != mesh.positions.size() || mesh.normals.size() != mesh.positions.size()) {
		throw std::runtime_error("Unsupported tangent accessor or missing normals");
	}

	const tinygltf::BufferView& view = model.bufferViews.at(acc.bufferView);
	const tinygltf::Buffer& buffer = model.buffers.at(view.buffer);

	const size_t elementBytes = 4 * sizeof(float);
	const size_t stride = view.byteStride ? view.byteStride : elementBytes;

	if (view.byteOffset > buffer.data.size() ||
		view.byteLength > buffer.data.size() - view.byteOffset ||
		acc.byteOffset > view.byteLength || stride < elementBytes) {
		throw std::runtime_error("Invalid tangent buffer layout");
	}

	const size_t available = view.byteLength - acc.byteOffset;
	if (acc.count > 0 && (available < elementBytes ||
		acc.count - 1 >(available - elementBytes) / stride)) {
		throw std::runtime_error("Tangent data exceeds buffer view");
	}

	const glm::mat3 linear(worldTransform);
	const float determinant = glm::determinant(linear);
	if (!std::isfinite(determinant) || determinant == 0.0f) {
		throw std::runtime_error("Singular tangent transform");
	}

	const float transformSign = determinant < 0.0f ? -1.0f : 1.0f;
	const size_t start = view.byteOffset + acc.byteOffset;
	mesh.tangents.resize(acc.count);

	for (size_t v = 0; v < acc.count; v++) {
		float xyzw[4]; //get a tangent
		std::memcpy(xyzw, buffer.data.data() + start + v * stride, sizeof(xyzw));
		if (!std::isfinite(xyzw[0]) || !std::isfinite(xyzw[1]) || !std::isfinite(xyzw[2]) ||
			(xyzw[3] != -1.0f && xyzw[3] != 1.0f)) {
			throw std::runtime_error("Invalid tangent");
		}

		glm::vec3 t = linear * glm::vec3(xyzw[0], xyzw[1], xyzw[2]); //tangents are directions so just use linear transform instead of invtrans, localToWorld
		glm::vec3 n = mesh.normals[v]; // already transformed and normalized
		
		t -= n * glm::dot(n, t);
		
		float lengthSquared = glm::dot(t, t);

		//must not be infinite and must be non zero
		if (!std::isfinite(lengthSquared) || lengthSquared <= 0.000000000001f) {
			throw std::runtime_error("Degenerate transformed tangent");
		}

		t = glm::normalize(t);
		mesh.tangents[v] = glm::vec4(t, xyzw[3] * transformSign);
	}
}

//Only TEXCOORD_0 is supported: acts as a guide on how to assign a 2D coordinate to every 3D vertex, limits baked lightmaps, detail/tiling maps, decals/logos
bool loadGltf(const std::string& filename, std::vector<MeshData>& output, std::vector<ImageData>& outputImages, std::vector<TextureData>& outputTextures) {
	std::vector<MeshData> loadedMeshes;

	tinygltf::TinyGLTF loader; //performs parcing
	tinygltf::Model model; //owns the loaded CPU data
	std::string error;
	std::string warning;
	//a gltf primitive is a drawable mesh section
	//a gltf primitive is a group of geometry within a mesh that uses one material

	// read .gltf file
	bool loaded = loader.LoadASCIIFromFile(&model, &error, &warning, filename);

	if (!warning.empty()) {
		std::cerr << "glTF warning: " << warning << '\n';
	}
	if (!error.empty()) {
		std::cerr << "glTF error: " << error << '\n';
	}
	if (!loaded) {
		std::cerr << "Failed to load: " << filename << '\n';
		return false;
	}
	std::cout << "Loaded: " << filename << '\n'
		<< "Scenes: " << model.scenes.size() << '\n'
		<< "Nodes: " << model.nodes.size() << '\n'
		<< "Meshes: " << model.meshes.size() << '\n'
		<< "Buffers: " << model.buffers.size() << '\n';

	//////////////////
	// Texture load
	//////////////////
	std::cout << "Images: " << model.images.size() << std::endl; //decoded pixel bytes
	std::cout << "Textures: " << model.textures.size() << std::endl;

	for (size_t i = 0; i < model.images.size(); i++) {
		const tinygltf::Image& image = model.images[i];

		std::cout << "Image " << i
			<< ": " << image.width << " x " << image.height
			<< ", channels: " << image.component
			<< ", bits per channel: " << image.bits
			<< ", decoded bytes: " << image.image.size()
			<< std::endl;
	}



	//////////////////
	// Scene check
	//////////////////
	if (model.scenes.empty()) {
		std::cerr << "glTF has no scenes\n";
		return false;
	}

	// Use the declared default scene, or choose scene 0 if none is declared
	const int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
	const tinygltf::Scene& gltfScene = model.scenes.at(sceneIndex); //Scene
	std::vector<MeshInstance> instances;

	for (int rootNode : gltfScene.nodes) {
		//traverse scene graph to combine node transforms
		traverseMesh(model, rootNode, glm::mat4(1.0f), instances);
	}

	// Decode and transform mesh data
	//////////////////
	// Model Loop
	//////////////////
	for (const MeshInstance& instance: instances) {
		const tinygltf::Mesh& mesh = model.meshes.at(instance.meshIndex); //Mesh

		std::cout << "Mesh " << instance.meshIndex << ": " << mesh.primitives.size() << " primitive(s)\n";

		//////////////////
		// Mesh Loop
		//////////////////
		for (size_t p = 0; p < mesh.primitives.size(); p++) {
			const tinygltf::Primitive& primitive = mesh.primitives[p]; //Primitive

			auto positionIt = primitive.attributes.find("POSITION");
			if (positionIt == primitive.attributes.end()) {
				std::cerr << "Primitive has no POSITION attribute\n";
				return false;
			}

			//Accessors describe how to interpret data (element count, type, and location in a buffer), it follows the gltf format
			const tinygltf::Accessor& positions = model.accessors.at(positionIt->second); //Accessor

			// Supports ordinary FLOAT VEC3 positions
			if (positions.type != TINYGLTF_TYPE_VEC3 ||
				positions.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
				positions.sparse.isSparse ||
				positions.bufferView < 0) {
				std::cerr << "Unsupported position accessor\n";
				return false;
			}

			//////////////////
			// Get positions using accessor
			//////////////////

			// Accessor -> buffer view & buffer, use buffer view to access buffer
			const tinygltf::BufferView& view = model.bufferViews.at(positions.bufferView); //A specified region within a buffer
			const tinygltf::Buffer& buffer = model.buffers.at(view.buffer); //Raw bytes loaded into memory

			// Each position contains 3 floats
			const size_t elementBytes = 3 * sizeof(float);
			// A missing byteStride means elements are tightly packed
			const size_t stride = view.byteStride != 0 ? view.byteStride : elementBytes;

			if (view.byteOffset > buffer.data.size() ||
				view.byteLength > buffer.data.size() - view.byteOffset ||
				positions.byteOffset > view.byteLength ||
				stride < elementBytes) {
				std::cerr << "Invalid position buffer layout\n";
				return false;
			}

			const size_t available = view.byteLength - positions.byteOffset;

			if (positions.count > 0 && (available < elementBytes || positions.count - 1 > (available - elementBytes) / stride)){
				std::cerr << "Position data exceeds buffer view\n";
				return false;
			}

			const size_t start = view.byteOffset + positions.byteOffset;

			//////////////////
			// Positions Loop
			//////////////////
			std::vector<glm::vec3> meshPositions;
			meshPositions.reserve(positions.count);

			for (size_t v = 0; v < positions.count; v++) {
				float xyz[3];

				//copy into xyz, buffer.data with offset start + v * stride, with increments of 1 to v
				std::memcpy(xyz, buffer.data.data() + start + v * stride, sizeof(xyz));

				meshPositions.push_back(glm::vec3(xyz[0], xyz[1], xyz[2]));

				//std::cout << "Position" << v << ": " << xyz[0] << ", " << xyz[1] << ", " << xyz[2] << "\n";
			}

			for (glm::vec3& position : meshPositions) {
				position = glm::vec3(instance.worldTransform * glm::vec4(position, 1.0f));
			}

			if (!meshPositions.empty()) {
				const glm::vec3 & first = meshPositions.front();

				std::cout << "First world position: " << first.x << ", " << first.y << ", " << first.z << '\n';
			}

			std::cout << " Primitive " << p << '\n'
					  << "    Mode: " << primitive.mode << '\n'
					  << "    Vertices: " << positions.count << '\n'
					  << "    Position type: " << positions.type << '\n'
					  << "    Position component type: " << positions.componentType << '\n';

			//////////////////
			// Get normals
			//////////////////
			std::vector<glm::vec3> meshNormals;

			auto normalIt = primitive.attributes.find("NORMAL");

			if (normalIt != primitive.attributes.end()) {
				const tinygltf::Accessor& normals = model.accessors.at(normalIt->second);
				if (normals.type != TINYGLTF_TYPE_VEC3 ||
					normals.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
					normals.sparse.isSparse ||
					normals.bufferView < 0 ||
					normals.count != positions.count) {
					std::cerr << "Unsupported normal accessor\n";
					return false;
				}

				std::cout << "    Normals: " << normals.count << '\n';

				const tinygltf::BufferView& normalView = model.bufferViews.at(normals.bufferView);
				const tinygltf::Buffer& normalBuffer = model.buffers.at(normalView.buffer);

				const size_t normalElementBytes = 3 * sizeof(float);
				const size_t normalStride = normalView.byteStride != 0 ? normalView.byteStride : normalElementBytes;

				// Validate
				if (normalView.byteOffset > normalBuffer.data.size() ||
					normalView.byteLength > normalBuffer.data.size() - normalView.byteOffset ||
					normals.byteOffset > normalView.byteLength ||
					normalStride < normalElementBytes) {
					std::cerr << "Invalid normal buffer layout\n";
					return false;
				}

				const size_t availableNormalBytes = normalView.byteLength - normals.byteOffset;

				if (normals.count > 0 && availableNormalBytes < normalElementBytes ||
					normals.count - 1 >(availableNormalBytes - normalElementBytes) / normalStride) {
					std::cerr << "Normal data exceeds buffer view\n";
					return false;
				}

				const size_t normalStart = normalView.byteOffset + normals.byteOffset;

				meshNormals.reserve(normals.count);

				//////////////////
				// Normals Loop
				//////////////////
				for (size_t v = 0; v < normals.count; v++) {
					float xyz[3];

					std::memcpy(xyz, normalBuffer.data.data() + normalStart + v * normalStride, sizeof(xyz));

					meshNormals.push_back(glm::vec3(xyz[0], xyz[1], xyz[2]));
				}

				const glm::mat3 linearTransform = glm::mat3(instance.worldTransform);
				if (glm::determinant(linearTransform) == 0.0f) {
					std::cerr << "Cannot transform normasl with a singular transform\n";
					return false;
				}

				const glm::mat3 normalMatrix = glm::transpose(glm::inverse(linearTransform)); //inverseTranspose undoes rotation, only scale
				
				for (glm::vec3& normal : meshNormals) {
					glm::vec3 worldNormal = normalMatrix * normal;

					if (glm::dot(worldNormal, worldNormal) == 0.0f) {
						std::cerr << "Invalid zero-length normal\n";
						return false;
					}
					normal = glm::normalize(worldNormal);
				}
			}

			//////////////////
			// Get texture coordinates
			//////////////////
			std::vector<glm::vec2> meshTexcoords;

			auto texcoordIt = primitive.attributes.find("TEXCOORD_0");

			if (texcoordIt != primitive.attributes.end()) {
				const tinygltf::Accessor& texcoords = model.accessors.at(texcoordIt->second);

				//Non sparse, float VEC2 coords
				if (texcoords.type != TINYGLTF_TYPE_VEC2 ||
					texcoords.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
					texcoords.normalized || texcoords.sparse.isSparse || texcoords.bufferView < 0 ||
					texcoords.count != positions.count) {
					std::cerr << "Unsupported texture coordinate accessor\n";
					return false;
				}

				const tinygltf::BufferView& texcoordView = model.bufferViews.at(texcoords.bufferView);
				const tinygltf::Buffer& texcoordBuffer = model.buffers.at(texcoordView.buffer);
				
				const size_t texcoordElementBytes = 2 * sizeof(float);
				const size_t texcoordStride = texcoordView.byteStride != 0 ? texcoordView.byteStride : texcoordElementBytes;

				if (texcoordView.byteOffset > texcoordBuffer.data.size() ||
					texcoordView.byteLength > texcoordBuffer.data.size() - texcoordView.byteOffset ||
					texcoords.byteOffset > texcoordView.byteLength || texcoordStride < texcoordElementBytes) {
					std::cerr << "Invalid texture coordinate buffer layout\n";
					return false;
				}

				const size_t availableTexcoordBytes = texcoordView.byteLength - texcoords.byteOffset;

				if(texcoords.count > 0 && (availableTexcoordBytes < texcoordElementBytes || texcoords.count - 1 > (availableTexcoordBytes - texcoordElementBytes) / texcoordStride)) {
					std::cerr << "Texture coordinates exceed buffer view\n";
					return false;
				}

				const size_t texcoordStart = texcoordView.byteOffset + texcoords.byteOffset;

				meshTexcoords.reserve(texcoords.count);

				for (size_t v = 0; v < texcoords.count; ++v) {
					float uv[2];

					std::memcpy(uv, texcoordBuffer.data.data() + texcoordStart + v * texcoordStride, sizeof(uv));

					meshTexcoords.push_back(glm::vec2(uv[0], uv[1]));
				}
			}

			//////////////////
			// Get Indices
			//////////////////
			std::vector<std::array<std::uint32_t, 3>> meshTriangles;

			if (primitive.indices < 0) {
				std::cerr << "Non-indexed geometry is not supported yet\n";
				return false;
			}

			if (primitive.indices >= 0) {
				const tinygltf::Accessor& indices = model.accessors.at(primitive.indices);

				std::cout << "    Indices: " << indices.count << '\n'
						  << "    Index component type: "
						  << indices.componentType << '\n';

				//Indexed triangles with unsigned 16 bit indices
				if (primitive.mode != TINYGLTF_MODE_TRIANGLES ||
					indices.type != TINYGLTF_TYPE_SCALAR ||
					indices.sparse.isSparse ||
					indices.bufferView < 0 ||
					indices.count % 3 != 0) {
					std::cerr << "Unsupported triangle index format\n";
					return false;
				}

				const tinygltf::BufferView& indexView = model.bufferViews.at(indices.bufferView);
				const tinygltf::Buffer& indexBuffer = model.buffers.at(indexView.buffer);

				size_t indexBytes = 0;

				switch (indices.componentType) {
				case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
					indexBytes = sizeof(std::uint8_t);
					break;
					
				case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
					indexBytes = sizeof(std::uint16_t);
					break;

				case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
					indexBytes = sizeof(std::uint32_t);
					break;

				default:
					std::cerr << "Unsupported index component type\n";
					return false;
				}

				//Index data is tightly packed
				if (indexView.byteStride != 0 ||
					indexView.byteOffset > indexBuffer.data.size() ||
					indexView.byteLength > indexBuffer.data.size() - indexView.byteOffset ||
					indices.byteOffset > indexView.byteLength) {
					std::cerr << "Invalid index buffer layout\n";
					return false;
				}

				const size_t availableIndexBytes = indexView.byteLength - indices.byteOffset;

				if (indices.count > availableIndexBytes / indexBytes) {
					std::cerr << "Index data exceeds buffer view\n";
					return false;
				}

				const size_t indexStart = indexView.byteOffset + indices.byteOffset;

				meshTriangles.reserve(indices.count / 3);

				//GLTF has different types of requirements
				for (size_t t = 0; t < indices.count / 3; ++t) {
					std::uint32_t triangle[3]; //triangle is just a set of 3 shorts from indices
					for (size_t corner = 0; corner < 3; corner++) {
						const unsigned char* source = indexBuffer.data.data() + indexStart + (t * 3 + corner) * indexBytes;

						switch (indices.componentType) {
						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
							std::uint8_t value;
							std::memcpy(&value, source, sizeof(value));
							triangle[corner] = value;
							break;
						}

						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
							std::uint16_t value;
							std::memcpy(&value, source, sizeof(value));
							triangle[corner] = value;
							break;
						}

						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
							std::uint32_t value;
							std::memcpy(&value, source, sizeof(value));
							triangle[corner] = value;
							break;
						}
						}
					}

					if (triangle[0] >= positions.count ||
						triangle[1] >= positions.count ||
						triangle[2] >= positions.count) {
						std::cerr << "Triangle references an invalid vertex\n";
						return false;
					}

					meshTriangles.push_back({ triangle[0], triangle[1], triangle[2] });

					//std::cout << "Triangle " << t << ": " << triangle[0] << ", " << triangle[1] << ", " << triangle[2] << '\n';
				}
			}
			std::cout << "Stored " << meshPositions.size() << " positions and " << meshTriangles.size() << " triangles\n";

			// don't copy, just move mem ownership
			MeshData meshData;
			meshData.positions = std::move(meshPositions);
			meshData.normals = std::move(meshNormals);
			meshData.texcoords = std::move(meshTexcoords);
			std::cout << "    Texture coordinates: " << meshData.texcoords.size() << '\n';
			meshData.triangles = std::move(meshTriangles);
			meshData.gltfMaterialIndex = primitive.material;

			if (primitive.material >= 0) {
				if (static_cast<size_t>(primitive.material) >= model.materials.size()) {
					std::cerr << "Invalid material index\n";
					return false;
				}

				//get material property
				const tinygltf::Material& material = model.materials.at(primitive.material);

				const auto& baseColorTexture = material.pbrMetallicRoughness.baseColorTexture;
				
				if (baseColorTexture.index >= 0) {
					if (size_t(baseColorTexture.index) >= model.textures.size()) {
						std::cerr << "Invalid base color texture index\n";
						return false;
					}

					//test, only check TEXCOORD_0
					if (baseColorTexture.texCoord != 0) {
						std::cerr << "Base color texture requires an unsupported UV set\n";
						return false;
					}

					if (meshData.texcoords.empty()) {
						std::cerr << "Base color texture requires TEXCOORD_0\n";
						return false;
					}

					meshData.gltfBaseColorTextureIndex = baseColorTexture.index;
				}

				//get base color from materials
				const auto& color = material.pbrMetallicRoughness.baseColorFactor;

				if (color.size() != 4) {
					std::cerr << "Invalid base color factor\n";
					return false;
				}

				meshData.baseColorFactor = glm::vec4(static_cast<float>(color[0]), static_cast<float>(color[1]), static_cast<float>(color[2]), static_cast<float>(color[3]));
				
				const auto& pbr = material.pbrMetallicRoughness;
				meshData.metallicFactor = glm::clamp(static_cast<float>(pbr.metallicFactor), 0.0f, 1.0f);
				meshData.roughnessFactor = glm::clamp(static_cast<float>(pbr.roughnessFactor), 0.0f, 1.0f);
			
				const tinygltf::TextureInfo& mr = pbr.metallicRoughnessTexture;
				validateMaterialTexture(model, meshData, mr.index, mr.texCoord, "Metallic-roughness texture");
				meshData.gltfMetallicRoughnessTextureIndex = mr.index;

				const tinygltf::NormalTextureInfo& normalMap = material.normalTexture;
				validateMaterialTexture(model, meshData, normalMap.index, normalMap.texCoord, "Normal texture");
				meshData.gltfNormalTextureIndex = normalMap.index;
				meshData.normalScale = static_cast<float>(normalMap.scale);

				//Leave it at this for now, no KHR
				if (pbr.baseColorTexture.extensions.count("KHR_texture_transform") != 0 ||
					mr.extensions.count("KHR_texture_transform") != 0 ||
					normalMap.extensions.count("KHR_texture_transform") != 0) {
					throw std::runtime_error("KHR_texture_transform is not supported yet");
				}
			}

			std::cout << "Base color: "
				<< meshData.baseColorFactor.r << ", "
				<< meshData.baseColorFactor.g << ", "
				<< meshData.baseColorFactor.b << ", "
				<< meshData.baseColorFactor.a << '\n';

			std::cout << "Base color texture index: " << meshData.gltfBaseColorTextureIndex << std::endl;

			loadMeshTangents(model, primitive, instance.worldTransform, meshData);

			loadedMeshes.push_back(std::move(meshData));
		}
	}

	std::vector<ImageData> loadedImages;
	loadedImages.reserve(model.images.size());

	for (tinygltf::Image& image : model.images) {
		if (image.width <= 0 || image.height <= 0 || image.image.empty()) {
			std::cerr << "Image has no decoded pixel data\n";
			return false;
		}

		if (image.pixel_type != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE || image.bits != 8 || image.component != 4) {
			throw std::runtime_error("Material textures require decoded unsigned 8-bit RGBA images");
		}

		ImageData imageData;
		imageData.width = image.width;
		imageData.height = image.height;
		imageData.channels = image.component;
		imageData.bitsPerChannel = image.bits;
		imageData.pixels = std::move(image.image);

		loadedImages.push_back(std::move(imageData));
	}

	std::vector<TextureData> loadedTextures;
	loadedTextures.reserve(model.textures.size());

	for (tinygltf::Texture& texture : model.textures) {
		if (texture.source < 0 || size_t(texture.source) >= loadedImages.size()) {
			std::cerr << "Texture has no supported image source \n";
			return false;
		}

		TextureData textureData;
		textureData.imageIndex = texture.source;

		if (texture.sampler >= 0) {
			if (size_t(texture.sampler) >= model.samplers.size()) {
				std::cerr << "Invalid texture sampler index\n";
				return false;
			}

			tinygltf::Sampler& sampler = model.samplers[texture.sampler];

			textureData.wrapS = sampler.wrapS;
			textureData.wrapT = sampler.wrapT;
			textureData.minFilter = sampler.minFilter;
			textureData.magFilter = sampler.magFilter;
		}

		loadedTextures.push_back(textureData);
	}

	output = std::move(loadedMeshes); //lets the destination vectors take ownership of the existing allocations instead of copying every vertex and triangle
	outputImages = std::move(loadedImages);
	outputTextures = std::move(loadedTextures);
	return true;
}

double getEnvironmentLuminance(const EnvironmentData& environment, size_t pixelIndex) {
	size_t offset = pixelIndex * 4;
	//numbers convert RGB into a luminance estimat
	//standard luminance coefficients for linear sRGB primaries (green contributres most, followed by red, blue)
	double luminance = 0.2126 * environment.pixels[offset] + 0.7152 * environment.pixels[offset + 1] + 0.0722 * environment.pixels[offset + 2];
	
	if (!std::isfinite(luminance) || luminance <= 0.0) {
		return 0.0;
	}

	return luminance;
}

//build CDF table for every pixel in the HDR
//1. calculate each pixel's brightness from RGB values, alpha ignored
//2. Compute average brightness, then 0.001 x average to scale down to small sampling area
//3. Calculate the solid angle of each pixel (pixels near the pole cover less area)
//4. Assigne each pixel this weight
//5. Store running sums of weights
//6. Divide every cumulative sum by the final total, covnerting range to [0,1]
void buildEnvironmentDistribution(EnvironmentData& environment) {
	int width = environment.width;
	int height = environment.height;
	size_t pixelCount = (size_t)width * height;

	double pi = 3.14159265358979323846;

	double luminanceSum = 0.0;
	for (size_t i = 0; i < pixelCount; i++) {
		luminanceSum += getEnvironmentLuminance(environment, i);
	}

	//give dark pixels a small sampling probability, for an entirely black image, use area based sampling
	double floorLuminance;
	if (luminanceSum > 0.0) {
		floorLuminance = 0.001 * luminanceSum / (double)pixelCount;
	}
	else {
		floorLuminance = 1.0;
	}

	environment.cdf.resize(pixelCount + 1);
	environment.cdf[0] = 0.0;

	double totalWeight = 0.0;

	for (int y = 0; y < height; y++) {
		double thetaTop = pi * y / height;
		double thetaBottom = pi * (y + 1) / height;

		double pixelSolidAngle = (2.0 * pi / width) * (std::cos(thetaTop) - std::cos(thetaBottom));

		for (int x = 0; x < width; x++) {
			size_t i = size_t(y) * width + x;
			double weight = (getEnvironmentLuminance(environment, i) + floorLuminance) * pixelSolidAngle;

			totalWeight += weight;
			environment.cdf[i + 1] = totalWeight;
		}
	}

	for (size_t i = 1; i <= pixelCount; i++) {
		environment.cdf[i] /= totalWeight;
	}

	environment.cdf.back() = 1.0;

	std::cout << " Built environment CDF: " << pixelCount << " pixels, total weight: " << totalWeight << '\n';
}

//Parsing using stb_image, exposed by tiny_gltf.h
bool loadEnvironment(const std::string& filename, EnvironmentData& output) {
	if (!stbi_is_hdr(filename.c_str())) {
		std::cerr << "Expected a readable Radiance HDR image: " << filename << std::endl;
		return false;
	}

	int width = 0;
	int height = 0;
	int sourceChannels = 0;

	//returns floating point pixels
	float* pixels = stbi_loadf( 
		filename.c_str(),
		&width,
		&height,
		&sourceChannels,
		4); //requests RGBA output even if the file contains RGB

	if (pixels == nullptr) {
		const char* reason = stbi_failure_reason();

		std::cerr << "Failed to load environment: " << filename << '\n'
			<< (reason ? reason : "Unknown iamge-loading error")
			<< '\n';

		return false;
	}

	if (width <= 0 || height <= 0) {
		stbi_image_free(pixels);
		std::cerr << "Invalid environment dimensions\n";
		return false;
	}

	EnvironmentData environment;
	environment.width = width;
	environment.height = height;

	size_t valueCount = (size_t)width * (size_t)height * 4;

	environment.pixels.assign(pixels, pixels + valueCount);
	stbi_image_free(pixels); //releases allocation after we copy the pixels into our vector

	buildEnvironmentDistribution(environment);
	output = std::move(environment);

	std::cout << "Loaded environment: " << filename << '\n'
		<< "Size: " << output.width << " x "
		<< output.height << ", RGBA float\n";

	return true;
}