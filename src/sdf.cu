
#include "sdf.h"
#include "utilities.h"
#include <glm/gtc/matrix_transform.hpp>


// sdfs from iq
static __host__ __device__ float boxSDF(glm::vec3 p, glm::vec3 b)
{
	glm::vec3 q = glm::abs(p) - b;
	return glm::length(glm::max(q, glm::vec3(0.0f))) + glm::min(glm::max(q.x, glm::max(q.y, q.z)), 0.0f);
}

static __host__ __device__ float sphereSDF(glm::vec3 p)
{
	return glm::length(p) - 1.8f;
}

static __host__ __device__ float octahedronSDF(glm::vec3 p, float s)
{
	p = glm::abs(p);
	float m = p.x + p.y + p.z - s;
	glm::vec3 q;
	if (3.0f * p.x < m) { q = glm::vec3(p.x, p.y, p.z); }
	else if (3.0f * p.y < m) { q = glm::vec3(p.y, p.z, p.x); }
	else if (3.0f * p.z < m) { q = glm::vec3(p.z, p.x, p.y); }
	else { return m * 0.57735027f; }

	float k = glm::clamp(0.5f * (q.z - q.y + s), 0.0f, s);
	return glm::length(glm::vec3(q.x, q.y - s + k, q.z - k));
}

static __host__ __device__ float hexPrismSDF(glm::vec3 p, glm::vec2 h)
{
	const glm::vec3 k = glm::vec3(-0.8660254f, 0.5f, 0.57735f);
	p = glm::abs(p);

	float kxyDotPxy = k.x * p.x + k.y * p.y;
	float t = 2.0f * glm::min(kxyDotPxy, 0.0f);
	p.x -= t * k.x;
	p.y -= t * k.y;

	glm::vec2 pxy = glm::vec2(p.x, p.y);
	glm::vec2 clampedPoint = glm::vec2(glm::clamp(p.x, -k.z * h.x, k.z * h.x), h.x);
	float dx = glm::length(pxy - clampedPoint) * glm::sign(p.y - h.x);
	float dy = p.z - h.y;
	glm::vec2 d = glm::vec2(dx, dy);

	return glm::min(glm::max(d.x, d.y), 0.0f) + glm::length(glm::max(d, glm::vec2(0.0f)));
}

static __host__ __device__ float cutHollowSphereSDF(glm::vec3 p, float r, float h, float t)
{
	float w = sqrt(r * r - h * h);
	glm::vec2 q = glm::vec2(glm::length(glm::vec2(p.x, p.z)), p.y);

	return ((h * q.x < w * q.y) ? glm::length(q - glm::vec2(w, h))
	                             : glm::abs(glm::length(q) - r)) - t;
}

static __host__ __device__ float triacontahedronSDF(glm::vec3 p)
{
	float size = 2.0f;
	p /= size;
	float c = cos(PI / 5.);
	float s = sqrt(0.75 - c * c);

	glm::vec3 n = glm::vec3(-0.5, -c, s); 

	p = glm::abs(p);
	p -= glm::vec3(2.0f * glm::min(glm::dot(n, p), 0.0f)) * n;

	p.x = glm::abs(p.x);
	p.y = glm::abs(p.y);
	p -= glm::vec3(2.0f * glm::min(glm::dot(n, p), 0.0f)) * n;

	p.x = glm::abs(p.x);
	p.y = glm::abs(p.y);
	p -= glm::vec3(2.0f * glm::min(glm::dot(n, p), 0.0f)) * n;

	float d = p.z-1.0f;

	return d*size;
}

static __host__ __device__ inline glm::vec3 random33(glm::vec3 p)
{
	return glm::fract(glm::sin(glm::vec3(
		glm::dot(p, glm::vec3(127.1f, 311.7f, 74.7f)),
		glm::dot(p, glm::vec3(269.5f, 183.3f, 246.1f)),
		glm::dot(p, glm::vec3(113.5f, 271.9f, 124.6f))
	)) * 43758.5453f);
}

static __host__ __device__ inline float random12(glm::vec2 p)
{
	return glm::fract(sinf(glm::dot(p, glm::vec2(127.1f, 311.7f))) * 43758.5453f);
}

static __host__ __device__ inline float voronoi3D(glm::vec3 xyz, float gridSize)
{
	glm::vec3 stw = xyz * gridSize;

	glm::vec3 i = glm::floor(stw);
	glm::vec3 f = glm::fract(stw);

	float minDist = 100.0f;
	for (int x = -1; x <= 1; x++) {
		for (int y = -1; y <= 1; y++) {
			for (int z = -1; z <= 1; z++) {
				glm::vec3 offset = glm::vec3(float(x), float(y), float(z));
				glm::vec3 randomPt = random33(i + offset);
				float currDist = glm::length(f - (randomPt + offset));

				minDist = glm::min(minDist, currDist);
			}
		}
	}
	return minDist;
}

static __host__ __device__ inline float sminPoly(float a, float b, float k)
{
	float h = glm::max(k - fabsf(a - b), 0.0f) / k;
	return glm::min(a, b) - h * h * k * 0.25f;
}

static __host__ __device__ inline glm::vec2 rot2(glm::vec2 v, float a)
{
	float c = cosf(a), s = sinf(a);
	return glm::vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

static __host__ __device__ inline float torusSDF(glm::vec3 p, glm::vec2 t)
{
	glm::vec2 q = glm::vec2(glm::length(glm::vec2(p.x, p.z)) - t.x, p.y);
	return glm::length(q) - t.y;
}

static __host__ __device__ inline float tetraPodSDF(glm::vec3 p)
{
	p.y -= 1.1f;
	float k = 0.8f;

	p *= 2.0f;

	glm::vec2 xy = rot2(glm::vec2(p.x, p.y), PI / 5.0f);
	p.x = xy.x; p.y = xy.y;

	float d = glm::length(p) - 1.0f;
	d = sminPoly(d, glm::length(p - glm::vec3(1, 1, 1) * 0.75f) - 0.6f, k);
	d = sminPoly(d, glm::length(p - glm::vec3(1, -1, -1) * 0.75f) - 0.6f, k);
	d = sminPoly(d, glm::length(p - glm::vec3(-1, 1, -1) * 0.75f) - 0.6f, k);
	d = sminPoly(d, glm::length(p - glm::vec3(-1, -1, 1) * 0.75f) - 0.6f, k);

	return d / 2.0f;
}

__host__ __device__ float sceneSDFTest(glm::vec3 p) {

	glm::vec3 v0 = glm::vec3(0.00f, 0.0f, 1.00f);
	glm::vec3 v1 = glm::vec3(-0.87f, 0.0f, -0.50f);
	glm::vec3 v2 = glm::vec3(0.87f, 0.0f, -0.50f);

	float tet = tetraPodSDF(p - v0 * 2.0f);
	float sph = glm::length(p - v2 * 2.0f - glm::vec3(0.0f, 0.9f, 0.0f)) - 0.6f;
	float oct = octahedronSDF(p - v1 * 2.0f - glm::vec3(0.0f, 1.0f, 0.0f), 1.0f);

	return glm::min(tet, glm::min(sph, oct));
}

static __host__ __device__ float ringStructure(glm::vec3 p)
{
	p.y -= 2.0f;
	const float rad = 0.1f;
	float brad = 2.0f;

	float d = 100000.0f;
	for (int k = 0; k < 3; k++)
	{
		float i = float(k);
		brad += i * 0.8f;

		glm::vec2 xz = rot2(glm::vec2(p.x, p.z), i);
		p.x = xz.x; p.z = xz.y;
		glm::vec2 xy = rot2(glm::vec2(p.x, p.y), i);
		p.x = xy.x; p.y = xy.y;

		float d0 = torusSDF(p, glm::vec2(brad, rad));

		glm::vec3 p1 = p;
		glm::vec2 xy1 = rot2(glm::vec2(p1.x, p1.y), 0.5f * PI);
		p1.x = xy1.x; p1.y = xy1.y;
		float d1 = torusSDF(p1, glm::vec2(brad, rad));

		glm::vec2 yz = rot2(glm::vec2(p.y, p.z), 0.5f * PI);
		p.y = yz.x; p.z = yz.y;
		float d2 = torusSDF(p, glm::vec2(brad, rad));

		d = glm::min(d, glm::min(d0, glm::min(d1, d2)));
	}

	return d;
}

static __host__ __device__ float ringStructureSmooth(glm::vec3 p)
{
	p.y -= 2.0f;
	const float rad = 0.45f;
	const float sm = 0.3f;

	float d = 100000.0f;
	for (int k = 1; k < 3; k++)
	{
		float i = float(k);
		float brad = i * 1.1f + 2.0f;

		glm::vec2 xz = rot2(glm::vec2(p.x, p.z), i);
		p.x = xz.x; p.z = xz.y;
		glm::vec2 xy = rot2(glm::vec2(p.x, p.y), i);
		p.x = xy.x; p.y = xy.y;

		float d0 = torusSDF(p, glm::vec2(brad, rad));

		glm::vec3 p1 = p;
		glm::vec2 xy1 = rot2(glm::vec2(p1.x, p1.y), 0.5f * PI);
		p1.x = xy1.x; p1.y = xy1.y;
		float d1 = torusSDF(p1, glm::vec2(brad, rad));

		glm::vec2 yz = rot2(glm::vec2(p.y, p.z), 0.5f * PI);
		p.y = yz.x; p.z = yz.y;
		float d2 = torusSDF(p, glm::vec2(brad, rad));

		d = glm::min(d, sminPoly(d0, sminPoly(d1, d2, sm), sm));
	}

	return d;
}

static __host__ __device__ float rippleLensSDF(glm::vec3 p)
{
	glm::vec2 yz = rot2(glm::vec2(p.y, p.z), 0.35f);
	p.y = yz.x; p.z = yz.y;
	const float R = 3.0f;
	const float h = 0.45f;
	float lens = glm::max(glm::length(p - glm::vec3(0.0f, h - R, 0.0f)) - R,
		glm::length(p - glm::vec3(0.0f, R - h, 0.0f)) - R);

	float r = glm::length(glm::vec2(p.x, p.z));
	float wTop = glm::smoothstep(-h, h, p.y);
	float wRim = 1.0f - glm::smoothstep(1.0f, 1.45f, r);
	float ripple = 0.04f * cosf(9.0f * r) * wTop * wRim;
	return (lens - ripple) * 0.68f;
}

static __host__ __device__ float glassPanels(glm::vec3 p)
{
	p.y -= 2.0f;
	const float num = 5.0f;
	const float s = 2.0f;
	float idz = glm::clamp(glm::round(p.z / s), 0.0f, num);
	p.z -= s * idz;

	glm::vec2 xy = rot2(glm::vec2(p.x, p.y), idz);

	float h = random12(glm::vec2(idz, 0.0f)) * 0.1f;
	glm::vec2 xz = rot2(glm::vec2(p.x, p.z), h);

	float dnoise = voronoi3D(p, 1.);
	p.z -= dnoise * 0.1 * glm::smoothstep(0.0f, 0.2f, p.z);

	float sm = 1.0f - glm::smoothstep(0.0f, num, idz);
	sm = 1.0;
	float d2 = boxSDF(p, glm::vec3(3.0f * sm, 4.0f * sm, 0.3f));

	return d2 * 0.3f;
}

// Value noise 3D by iq
// https://www.shadertoy.com/view/4sfGzS
static __host__ __device__ inline float hash31(glm::ivec3 p)
{
	// 3D -> 1D
	unsigned int n = (unsigned int)(p.x * 3 + p.y * 113 + p.z * 311);

	// 1D hash by Hugo Elias
	n = (n << 13) ^ n;
	n = n * (n * n * 15731u + 789221u) + 1376312589u;
	return float(n & 0x0fffffffu) / float(0x0fffffff);
}

static __host__ __device__ inline float valueNoise3D(glm::vec3 x)
{
	glm::ivec3 i = glm::ivec3(glm::floor(x));
	glm::vec3 f = glm::fract(x);
	f = f * f * (3.0f - 2.0f * f);

	return glm::mix(glm::mix(glm::mix(hash31(i + glm::ivec3(0, 0, 0)),
	                                  hash31(i + glm::ivec3(1, 0, 0)), f.x),
	                         glm::mix(hash31(i + glm::ivec3(0, 1, 0)),
	                                  hash31(i + glm::ivec3(1, 1, 0)), f.x), f.y),
	                glm::mix(glm::mix(hash31(i + glm::ivec3(0, 0, 1)),
	                                  hash31(i + glm::ivec3(1, 0, 1)), f.x),
	                         glm::mix(hash31(i + glm::ivec3(0, 1, 1)),
	                                  hash31(i + glm::ivec3(1, 1, 1)), f.x), f.y), f.z);
}


static __host__ __device__ float sphereFieldCell(glm::vec3 p, glm::vec2 cell)
{
	const float num = 5.0f;
	const float s = 5.0f;

	float r = random12(cell);
	float dens = glm::smoothstep(num * sqrtf(2.0f) + 2.0f, 0.0f, glm::length(cell));
	if (r > dens) return 1e9f;

	float rr = glm::fract(r * 10.0f);
	p.x -= s * cell.x;
	p.z -= s * cell.y;
	p.y -= 1.0f;
	p += 2.0f * glm::vec3(r * 2.0f - 1.0f, 0.0f, rr * 2.0f - 1.0f);

	glm::vec2 xz = rot2(glm::vec2(p.x, p.z), rr);
	p.x = xz.x; p.z = xz.y;
	glm::vec2 xy = rot2(glm::vec2(p.x, p.y), rr);
	p.x = xy.x; p.y = xy.y;

	float sph = tetraPodSDF(p);
	if (r < 0.3f) sph += valueNoise3D(p * 6.0f) * 0.08f;
	return sph;
}

static __host__ __device__ float sphereFieldSDF(glm::vec3 p)
{
	const float num = 5.0f;
	const float s = 5.0f;
	glm::vec2 id = glm::vec2(glm::clamp(glm::round(p.x / s), -num, num),
	                         glm::clamp(glm::round(p.z / s), -num, num));

	float d = sphereFieldCell(p, id);
	for (int j = -1; j <= 1; j++)
	{
		for (int i = -1; i <= 1; i++)
		{
			glm::vec2 cell = id + glm::vec2(float(i), float(j));
			if ((i == 0 && j == 0) || glm::abs(cell.x) > num || glm::abs(cell.y) > num) continue;
			d = glm::min(d, sphereFieldCell(p, cell));
		}
	}
	d = glm::min(d, 3.0f);
	return d * 0.4f;
}

static __host__ __device__ float rippleLensSuperSDF(glm::vec3 p)
{

	

	p -= 5.0f;

	glm::vec2 xy = rot2(glm::vec2(p.x, p.y), PI / 4.0f);
	p.x = xy.x; p.y = xy.y;

	const float R = 8.0f;
	const float h = 0.55f;

	float lens = glm::max(
		glm::length(p - glm::vec3(0.0f, h - R, 0.0f)) - R,
		glm::length(p - glm::vec3(0.0f, R - h, 0.0f)) - R
	);

	float r = glm::length(glm::vec2(p.x, p.z));
	float wTop = glm::smoothstep(-h, h, p.y);
	float wRim = 1.0f - glm::smoothstep(4.0f, 5.0f, r);

	float ripple = 0.04f * cosf(9.0f * r) * wTop * wRim;

	xy = rot2(glm::vec2(p.x, p.y), PI/2.0f);
	p.x = xy.x; p.y = xy.y;

	float ring = 100000.0f;
	for (int i = 0; i < 20; i++)
	{
		float t = (float)i;
		float a = t * 0.01f;

		/*p.xz = rot(p.xz, a);
		p.xy = rot(p.xy, a);*/

		glm::vec2 xz = rot2(glm::vec2(p.x, p.z), a);
		p.x = xz.x; p.z = xz.y;
		glm::vec2 xy = rot2(glm::vec2(p.x, p.y), a);
		p.x = xy.x; p.y = xy.y;

		float r = torusSDF(p, glm::vec2(3.5f + t * 0.1f, 0.05f));
		ring = glm::min(ring, r);
	}

	float dd = (lens - ripple) * 0.68f;
	return glm::min(ring, dd);
}

__host__ __device__ float sceneSDFDef(glm::vec3 p) {

	p -= glm::vec3(0.0f, 2.f, 0.0f);
	const float scale = 0.6;

	float oct = octahedronSDF(p, 2.0f);

	p -= glm::vec3(0.0f, 0.f, -4.0f);
	
	float sph = sphereSDF(p);

	p += glm::vec3(0.0f, 0.f, 8.0f);
	float triacon = triacontahedronSDF(p);
	

	return glm::min(glm::min(oct, sph), triacon);
	
}

__constant__ int dev_sdfScene;

void setSDFScene(int i)
{
	cudaMemcpyToSymbol(dev_sdfScene, &i, sizeof(int));
}

__host__ __device__ float sceneSDF(glm::vec3 p)
{
#ifdef __CUDA_ARCH__
	switch (dev_sdfScene)
	{
	case 1: return sceneSDFTest(p);
	case 2: return ringStructureSmooth(p);
	case 3: return glassPanels(p);
	default: return sceneSDFDef(p);
	}
#else
	return sceneSDFDef(p);
#endif
}

static __host__ __device__ glm::vec3 getSDFNormal(glm::vec3 p)
{
	const float eps = 0.0001f;
	glm::vec3 n;
	n.x = sceneSDF(p + glm::vec3(eps, 0.0f, 0.0f)) - sceneSDF(p - glm::vec3(eps, 0.0f, 0.0f));
	n.y = sceneSDF(p + glm::vec3(0.0f, eps, 0.0f)) - sceneSDF(p - glm::vec3(0.0f, eps, 0.0f));
	n.z = sceneSDF(p + glm::vec3(0.0f, 0.0f, eps)) - sceneSDF(p - glm::vec3(0.0f, 0.0f, eps));
	return glm::normalize(n);
}



__host__ __device__ float sdfIntersectionTest(Ray r, glm::vec3& intersectionPoint, glm::vec3& normal) {
	glm::vec3 rd = glm::normalize(r.direction);

	const int maxSteps = 128;
	const float epsilon = 0.0001f;
	const float maxDist = 100.0f;

	float t = 0.0f;
	for (int i = 0; i < maxSteps; i++)
	{
		glm::vec3 p = r.origin + t * rd;
		float dist = sceneSDF(p);

		if (glm::abs(dist) < epsilon)
		{
			intersectionPoint = p;
			normal = getSDFNormal(p);
			return t;
		}

		t += glm::abs(dist); // glass too

		if (t > maxDist)
		{
			break;
		}
	}

	return -1;
}
