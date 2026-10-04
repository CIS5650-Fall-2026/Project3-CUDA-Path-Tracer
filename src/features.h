#define ENABLE_SPECULAR 1
#define REFRACTION 1
#define ANTIALIASING 1
#define DEPTH_OF_FIELD 1
#define STREAM_COMPACTION 1
#define SORT_BY_MATERIAL 0
#define RUSSIAN_ROULETTE 1
#define DIRECT_LIGHTING 1     // aim each path's final traced ray at a random point on a light
#define RUSSIAN_ROULETTE_MIN_DEPTH 3   // bounces every path gets before it can be culled
#define MESH_BOUNDING_VOLUME_CULLING 1
#define BVH 1
#define BVH_MAX_DEPTH 32   // deepest leaf; the GPU traversal stack is sized from this
#define BVH_LEAF_SIZE 4    // stop splitting once a node holds this many triangles
#define LOG_BOUNCE_COUNTS 0
