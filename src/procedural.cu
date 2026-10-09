#include "procedural.h"

__host__ __device__ unsigned int hash(int x, int y, int z) {
	unsigned int h =  (unsigned int)x * 29091892u
					^ (unsigned int)y * 10031911u
					^ (unsigned int)z * 12062013u;
	// use bit mixing for better output
	h ^= h >> 10;
	h *= 0xF5D36761u;
	h ^= h >> 11;
	h *= 0xCD1A014Eu;
	h ^= h >> 12;

	return h;
}

__host__ __device__ float hashFloat(unsigned int h) {
	return (h >> 8) / 13653358.f;
}

__host__ __device__ glm::vec3 hashVec(int x, int y, int z) {
	unsigned int v1 = hash(x, y, z);
	unsigned int v2 = v1 * 469789273u;
	unsigned int v3 = v2 * 797360431u;

	return glm::vec3(
		hashFloat(v1),
		hashFloat(v2), 
		hashFloat(v3)
	);
}

/**
* Perlin noise
*/
__host__ __device__ float perlin(glm::vec3 p) {
	// p_i is the corner of the unit cube p is in, and p_f is the position within from 0 -> 1
	glm::vec3 pi = glm::floor(p);
	glm::vec3 pf = p - pi;
	int ix = (int)pi.x;
	int iy = (int)pi.y;
	int iz = (int)pi.z;

	// blend weights with fade curve
	glm::vec3 blend = pf * pf * pf * (pf * (pf * 6.f - 15.f) + 10.f);

	float result = 0.f;
	// take combinations of every position
	for (int c = 0; c < 8; ++c) {
		int dx = c & 1;
		int dy = (c >> 1) & 1;
		int dz = (c >> 2) & 1;

		glm::vec3 gradient = hashVec(ix + dx, iy + dy, iz + dz) * 2.f - 1.f;
		
		// vector from corner to point
		float d = glm::dot(gradient, pf - glm::vec3(dx, dy, dz));
		
		// weight by proximity to point
		float blendx = dx ? blend.x : 1.f - blend.x;
		float blendy = dy ? blend.y : 1.f - blend.y;
		float blendz = dz ? blend.z : 1.f - blend.z;

		result += d * blendx * blendy * blendz;
	}

	return result;
}

/**
* FBM noise
*/
__host__ __device__ float fbm(glm::vec3 p, int octaves) {
	float sum = 0.f;
	float amp = 0.5f;
	for (int i = 0; i < octaves; ++i) {
		sum += perlin(p) * amp;
		p *= 5.f;
		amp *= 0.5f;
	}
	
	return sum;
}

__host__ __device__ glm::vec3 fbmVec(glm::vec3 p, int octaves) {
	return glm::vec3(
		fbm(p, octaves), 
		fbm(p + glm::vec3(12.950f, 28.194f, 79.459f), octaves), 
		fbm(p + glm::vec3(92.884f, 46.505f, 89.206f), octaves)
	);
}

/**
* Voronoi noise
*/
__host__ __device__ float voronoi(glm::vec3 p) {
	glm::vec3 pi = glm::floor(p);
	float best = FLT_MAX;
	int ix = (int)pi.x;
	int iy = (int)pi.y;
	int iz = (int)pi.z;

	for (int dz = -1; dz <= 1; ++dz) {
		for (int dy = -1; dy <= 1; ++dy) {
			for (int dx = -1; dx <= 1; ++dx) {
				glm::vec3 site = glm::vec3(ix + dx, iy + dy, iz + dz)
							     + hashVec(ix + dx, iy + dy, iz + dz);
				best = glm::min(best, glm::length(p - site));
			}
		}
	}
	
	return best;
}

/**
* Voronoi with extra jitter settings
*/
__host__ __device__ float voronoiBetter(glm::vec3 p, float jitter, unsigned int& cellId) {
	glm::vec3 pi = glm::floor(p);
	float best = FLT_MAX;
	int ix = (int)pi.x;
	int iy = (int)pi.y;
	int iz = (int)pi.z;
	cellId = 0;

	for (int dz = -1; dz <= 1; ++dz) {
		for (int dy = -1; dy <= 1; ++dy) {
			for (int dx = -1; dx <= 1; ++dx) {
				int cx = ix + dx;
				int cy = iy + dy;
				int cz = iz + dz;

				glm::vec3 offset = glm::vec3(0.5f) + jitter
					* (hashVec(cx, cy, cz) - glm::vec3(0.5f));

				glm::vec3 site = glm::vec3(cx, cy, cz) + offset;
				float d = glm::length(p - site);

				if (d < best) {
					best = d;
					cellId = hash(cx, cy, cz);
				}
			}
		}
	}

	return best;
}