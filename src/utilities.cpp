//  UTILITYCORE- A Utility Library by Yining Karl Li
//  This file is part of UTILITYCORE, Copyright (c) 2012 Yining Karl Li
//
//  File: utilities.cpp
//  A collection/kitchen sink of generally useful functions

#include "utilities.h"
#include "math_utils.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iterator>
#include <ostream>
#include <sstream>

#include <cstdio>
#include <iostream>

bool utility_core::replace_string(std::string &str, const std::string &from,
                                const std::string &to) {
    size_t start_pos = str.find(from);
    if (start_pos == std::string::npos) {
        return false;
    }
    str.replace(start_pos, from.length(), to);
    return true;
}

std::string utility_core::convert_int_to_string(int number) {
    std::stringstream ss;
    ss << number;
    return ss.str();
}

glm::vec3 utility_core::clamp_rgb(glm::vec3 color) {
    if (color[0] < 0) {
        color[0] = 0;
    } else if (color[0] > 255) {
        color[0] = 255;
    }

    if (color[1] < 0) {
        color[1] = 0;
    } else if (color[1] > 255) {
        color[1] = 255;
    }

    if (color[2] < 0) {
        color[2] = 0;
    } else if (color[2] > 255) {
        color[2] = 255;
    }

    return color;
}

glm::mat4 utility_core::build_transformation_matrix(glm::vec3 translation,
                                                 glm::vec3 rotation,
                                                 glm::vec3 scale) {
    glm::mat4 translation_mat = glm::translate(glm::mat4(1.f), translation);
    glm::mat4 rotation_mat = glm::rotate(
        glm::mat4(1.f), rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
    rotation_mat =
        rotation_mat * glm::rotate(glm::mat4(1.f), rotation.y * (float)PI / 180,
                                  glm::vec3(0, 1, 0));
    rotation_mat =
        rotation_mat * glm::rotate(glm::mat4(1.f), rotation.z * (float)PI / 180,
                                  glm::vec3(0, 0, 1));
    glm::mat4 scale_mat = glm::scale(glm::mat4(1.f), scale);
    return translation_mat * rotation_mat * scale_mat;
}

std::vector<std::string> utility_core::tokenize_string(std::string str) {
    std::stringstream strstr(str);
    std::istream_iterator<std::string> it(strstr);
    std::istream_iterator<std::string> end;
    std::vector<std::string> results(it, end);
    return results;
}

std::istream &utility_core::safe_getline(std::istream &is, std::string &t) {
    t.clear();

    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks,
    // such as thread synchronization and updating the stream state.

    std::istream::sentry se(is, true);
    std::streambuf *sb = is.rdbuf();

    for (;;) {
        int c = sb->sbumpc();
        switch (c) {
        case '\n':
            return is;
        case '\r':
            if (sb->sgetc() == '\n') {
                sb->sbumpc();
            }
            return is;
        case EOF:
            // Also handle the case when the last line has no line ending
            if (t.empty()) {
                is.setstate(std::ios::eofbit);
            }
            return is;
        default:
            t += (char)c;
        }
    }
}
