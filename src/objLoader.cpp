#include "objLoader.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace
{
float cross2(glm::vec2 a, glm::vec2 b)
{
    return a.x * b.y - a.y * b.x;
}
bool onSegment(glm::vec2 a, glm::vec2 b, glm::vec2 p)
{
    return fabs(cross2(b - a, p - a)) <= 1e-10f &&
           p.x >= std::min(a.x, b.x) && p.x <= std::max(a.x, b.x) &&
           p.y >= std::min(a.y, b.y) && p.y <= std::max(a.y, b.y);
}
bool edgesIntersect(glm::vec2 a, glm::vec2 b, glm::vec2 c, glm::vec2 d)
{
    float abC = cross2(b - a, c - a), abD = cross2(b - a, d - a);
    float cdA = cross2(d - c, a - c), cdB = cross2(d - c, b - c);
    return (((abC > 0 && abD < 0) || (abC < 0 && abD > 0)) &&
            ((cdA > 0 && cdB < 0) || (cdA < 0 && cdB > 0))) ||
           onSegment(a, b, c) || onSegment(a, b, d) ||
           onSegment(c, d, a) || onSegment(c, d, b);
}
// Ear clipping supports simple concave polygons as well as triangles/quads.
void triangulate(const std::vector<glm::vec3> &face, std::vector<Triangle> &out)
{
    if (face.size() < 3)
        throw std::runtime_error("OBJ face has fewer than three vertices");
    glm::vec3 normal(0);
    for (size_t i = 0; i < face.size(); ++i)
        normal += glm::cross(face[i], face[(i + 1) % face.size()]);
    glm::vec3 an = glm::abs(normal);
    int drop = an.x > an.y ? (an.x > an.z ? 0 : 2) : (an.y > an.z ? 1 : 2);
    if (glm::length(normal) < 1e-12f)
        throw std::runtime_error("Degenerate OBJ polygon");
    std::vector<glm::vec2> p;
    for (auto v : face)
        p.emplace_back(v[(drop + 1) % 3], v[(drop + 2) % 3]);
    // Ear clipping assumes a simple boundary. A crossing polygon can have
    // nonzero signed area and still yield plausible but invalid triangles.
    for (size_t i = 0; i < p.size(); ++i)
        for (size_t j = i + 1; j < p.size(); ++j)
        {
            size_t nextI = (i + 1) % p.size(), nextJ = (j + 1) % p.size();
            if (nextI == j || nextJ == i)
                continue; // Adjacent edges share their endpoint by definition.
            if (edgesIntersect(p[i], p[nextI], p[j], p[nextJ]))
                throw std::runtime_error("Self-intersecting OBJ polygon");
        }
    float area = 0;
    for (size_t i = 0; i < p.size(); ++i)
        area += cross2(p[i], p[(i + 1) % p.size()]);
    float sign = area > 0 ? 1.0f : -1.0f;
    std::vector<int> indices(face.size());
    std::iota(indices.begin(), indices.end(), 0);
    while (indices.size() > 3)
    {
        bool found = false;
        for (size_t j = 0; j < indices.size(); ++j)
        {
            int a = indices[(j + indices.size() - 1) % indices.size()], b = indices[j],
                c = indices[(j + 1) % indices.size()];
            if (sign * cross2(p[b] - p[a], p[c] - p[b]) <= 1e-10f)
                continue;
            bool contains = false;
            for (int k : indices)
                if (k != a && k != b && k != c)
                {
                    if (sign * cross2(p[b] - p[a], p[k] - p[a]) >= -1e-10f &&
                        sign * cross2(p[c] - p[b], p[k] - p[b]) >= -1e-10f &&
                        sign * cross2(p[a] - p[c], p[k] - p[c]) >= -1e-10f)
                    {
                        contains = true;
                        break;
                    }
                }
            if (contains)
                continue;
            out.push_back({face[a], face[b], face[c]});
            indices.erase(indices.begin() + j);
            found = true;
            break;
        }
        if (!found)
            throw std::runtime_error("OBJ polygon is degenerate or self-intersecting");
    }
    auto a = face[indices[0]], b = face[indices[1]], c = face[indices[2]];
    if (glm::length(glm::cross(b - a, c - a)) < 1e-12f)
        throw std::runtime_error("Degenerate OBJ triangle");
    out.push_back({a, b, c});
}
} // namespace
void loadObj(const std::string &filename, std::vector<Triangle> &triangles, Geom &geom)
{
    std::ifstream file(filename);
    if (!file)
        throw std::runtime_error("Cannot open OBJ: " + filename);
    std::vector<glm::vec3> vertices;
    geom.triangleStart = static_cast<int>(triangles.size());
    geom.boundsMin = glm::vec3(std::numeric_limits<float>::max());
    geom.boundsMax = -geom.boundsMin;
    std::string line;
    int lineNumber = 0;
    try
    {
        while (std::getline(file, line))
        {
            ++lineNumber;
            std::istringstream in(line);
            std::string tag;
            in >> tag;
            if (tag == "v")
            {
                glm::vec3 p;
                if (!(in >> p.x >> p.y >> p.z) || !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                    throw std::runtime_error("Invalid vertex");
                vertices.push_back(p);
            }
            else if (tag == "f")
            {
                std::vector<glm::vec3> face;
                std::string token;
                while (in >> token)
                {
                    if (token[0] == '#')
                        break;
                    std::string text = token.substr(0, token.find('/'));
                    size_t consumed = 0;
                    int index = std::stoi(text, &consumed);
                    if (consumed != text.size() || index == 0)
                        throw std::runtime_error("Invalid vertex index");
                    int resolved = index > 0 ? index - 1 : static_cast<int>(vertices.size()) + index;
                    if (resolved < 0 || resolved >= static_cast<int>(vertices.size()))
                        throw std::runtime_error("Vertex index out of range");
                    face.push_back(vertices[resolved]);
                }
                if (face.size() > 3 && face.front() == face.back())
                    face.pop_back();
                triangulate(face, triangles);
            }
        }
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error(filename + ":" + std::to_string(lineNumber) + ": " + e.what());
    }
    geom.triangleCount = static_cast<int>(triangles.size()) - geom.triangleStart;
    if (!geom.triangleCount)
        throw std::runtime_error("OBJ contains no faces: " + filename);
    for (int i = geom.triangleStart; i < static_cast<int>(triangles.size()); ++i)
        for (auto p : {triangles[i].a, triangles[i].b, triangles[i].c})
        {
            geom.boundsMin = glm::min(geom.boundsMin, p);
            geom.boundsMax = glm::max(geom.boundsMax, p);
        }
}
