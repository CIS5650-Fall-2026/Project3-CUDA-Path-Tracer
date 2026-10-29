#pragma once

// Exit pupil table of the real lens: for each film radius, the rear-element box that rays can get out through.
// Film radius bins run from 0 to the film's half diagonal; the box for a pixel is the bin's box turned to its angle,
// which holds only while every rim and the stop are round (PBRT-v3 6.4.5).

#include "render/lens_trace.h"

#include <glm/glm.hpp>

#include <vector>

constexpr float EXIT_PUPIL_BIN_MM = 0.35f;
constexpr int EXIT_PUPIL_GRID = 256;  // aim points per side of the square of LensDevice::aimRadius, traced for each bin

int exitPupilBinCount(float halfDiagonal);

// Boxes as (xmin, ymin, xmax, ymax) in rear-element mm, an empty bin all zeros, and each bin's pupil area in mm^2
// (the aim points that got out). Host trace, for tests.
void exitPupilBuildHost(const LensDevice& lens, int grid, std::vector<glm::vec4>& boxes, std::vector<float>& areas);

// Same table into device memory of lens.pupilBinCount boxes and areas; returns the build time in ms.
float exitPupilBuildDevice(const LensDevice& lens, int grid, glm::vec4* boxes, float* areas);
