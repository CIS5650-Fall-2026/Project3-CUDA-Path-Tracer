// Textures are not used, so image decoding is disabled (the bundled
// stb_image.h is also too old for tinygltf's loader). Keep these defines
// in sync with scene.cpp.
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#include "tiny_gltf.h"
