#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// tinyexr decodes and encodes ZIP chunks with stb's zlib (environment.cpp)
// and declares this function extern "C". stb_image_write v0.98 defines it
// without a declaration of its own, so this one gives the definition below
// C linkage. The PNG writer calls it the same way either way.
extern "C" unsigned char* stbi_zlib_compress(unsigned char* data, int data_len, int* out_len, int quality);

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
