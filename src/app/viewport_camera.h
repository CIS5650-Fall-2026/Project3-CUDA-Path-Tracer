#pragma once

#include "scene/sceneStructs.h"

#include <glm/glm.hpp>

// Viewport navigation as math on a pose; the window and headless renders both build the render camera from one.

struct ViewportPose
{
    glm::vec3 position;
    float yaw;           // about world +Y: 0 looks down -Z, positive turns toward +X
    float pitch;         // up from the horizon, stopping just short of the poles
    float roll;          // about the view axis, positive tips the right side up; only the scene camera sets it
    float pivotDistance; // the orbit pivot's distance along the view axis
};

glm::vec3 viewDirection(const ViewportPose& pose);
glm::vec3 pivotOf(const ViewportPose& pose);

// The pose that sees what cam sees, pivot at cam.lookAt; a view straight up or down is tilted a hair off the pole.
ViewportPose poseFromCamera(const Camera& cam);

// Writes pose into cam: the basis is built against world +Y, then rolled.
void applyPose(const ViewportPose& pose, Camera& cam);

// dx, dy are mouse pixels; a mirrored camera's screen right is its world left.

void turn(ViewportPose& pose, double dx, double dy, bool mirrored);
// Turns, then moves back onto the line through the pivot, so the pivot stays put on screen.
void orbit(ViewportPose& pose, double dx, double dy, bool mirrored);
// Scales the pivot distance, so the camera never reaches the pivot.
void dollyToPivot(ViewportPose& pose, double dx, double dy);
// One pixel's width at the pivot's distance per pixel.
void pan(ViewportPose& pose, const Camera& cam, double dx, double dy);
void moveAlongGround(ViewportPose& pose, double dx, double dy, float flySpeed, bool mirrored);

// forward, right, up are -1, 0 or 1 along the view, the camera's right and world +Y; false when nothing moved.
bool fly(ViewportPose& pose, const Camera& cam, int forward, int right, int up, float distance);
void moveAlongView(ViewportPose& pose, float distance);
