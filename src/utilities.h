#pragma once

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <iosfwd>
#include <string>
#include <vector>

namespace utility_core {
extern bool replace_string(std::string& str, const std::string& from, const std::string& to);
extern glm::vec3 clamp_rgb(glm::vec3 color);
extern std::vector<std::string> tokenize_string(std::string str);
extern glm::mat4 build_transformation_matrix(glm::vec3 translation, glm::vec3 rotation,
                                             glm::vec3 scale);
extern std::string convert_int_to_string(int number);
extern std::istream& safe_getline(std::istream& is,
                                  std::string& t); // Thanks to http://stackoverflow.com/a/6089413
} // namespace utility_core
