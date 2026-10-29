#include "app/viewport_camera.h"

#include "utilities.h"

#include <cmath>

namespace
{
const float PITCH_LIMIT = 0.5f * PI - 0.001f;
// Mouse look and orbit, radians per pixel of mouse motion
const float TURN_PER_PIXEL = 0.0025f;
// LMB drag: a pixel of vertical motion moves as far as this much flying
const float GROUND_SECONDS_PER_PIXEL = 0.005f;
// Alt + RMB drag: the distance to the pivot scales by e^-x for x pixels
const float DOLLY_PER_PIXEL = 0.005f;
}  // namespace

glm::vec3 viewDirection(const ViewportPose& pose)
{
    return glm::vec3(std::sin(pose.yaw) * std::cos(pose.pitch), std::sin(pose.pitch),
        -std::cos(pose.yaw) * std::cos(pose.pitch));
}

glm::vec3 pivotOf(const ViewportPose& pose)
{
    return pose.position + pose.pivotDistance * viewDirection(pose);
}

ViewportPose poseFromCamera(const Camera& cam)
{
    ViewportPose p;
    p.position = cam.position;
    p.yaw = std::atan2(cam.view.x, -cam.view.z);
    p.pitch = glm::clamp(std::asin(glm::clamp(cam.view.y, -1.0f, 1.0f)), -PITCH_LIMIT, PITCH_LIMIT);
    // right is read before the mirror flip, which applyPose makes after rolling.
    p.roll = std::atan2(cam.mirrored ? -cam.right.y : cam.right.y, cam.up.y);
    p.pivotDistance = glm::length(cam.lookAt - cam.position);
    return p;
}

void applyPose(const ViewportPose& pose, Camera& cam)
{
    cam.position = pose.position;
    cam.view = viewDirection(pose);
    // Unit length: generateRayFromCamera scales its pixel offsets by right and up directly.
    glm::vec3 right = glm::normalize(glm::cross(cam.view, glm::vec3(0.0f, 1.0f, 0.0f)));
    glm::vec3 up = glm::normalize(glm::cross(right, cam.view));
    if (pose.roll != 0.0f)
    {
        const float c = std::cos(pose.roll);
        const float s = std::sin(pose.roll);
        const glm::vec3 level = right;
        right = c * level + s * up;
        up = c * up - s * level;
    }
    cam.up = up;
    cam.right = cam.mirrored ? -right : right;
    cam.lookAt = pose.position + pose.pivotDistance * cam.view;
}

void turn(ViewportPose& pose, double dx, double dy, bool mirrored)
{
    const float screenRight = mirrored ? -1.0f : 1.0f;
    pose.yaw = std::remainder(pose.yaw + screenRight * TURN_PER_PIXEL * (float)dx, 2.0f * PI);
    pose.pitch = glm::clamp(pose.pitch - TURN_PER_PIXEL * (float)dy, -PITCH_LIMIT, PITCH_LIMIT);
}

void orbit(ViewportPose& pose, double dx, double dy, bool mirrored)
{
    const glm::vec3 pivot = pivotOf(pose);
    turn(pose, dx, dy, mirrored);
    pose.position = pivot - pose.pivotDistance * viewDirection(pose);
}

void dollyToPivot(ViewportPose& pose, double dx, double dy)
{
    const glm::vec3 pivot = pivotOf(pose);
    pose.pivotDistance *= std::exp(-DOLLY_PER_PIXEL * (float)(dx - dy));
    pose.position = pivot - pose.pivotDistance * viewDirection(pose);
}

void pan(ViewportPose& pose, const Camera& cam, double dx, double dy)
{
    const float perPixel = pose.pivotDistance * cam.pixelLength.y;
    pose.position += perPixel * ((float)dx * cam.right - (float)dy * cam.up);
}

void moveAlongGround(ViewportPose& pose, double dx, double dy, float flySpeed, bool mirrored)
{
    turn(pose, dx, 0.0, mirrored);
    const glm::vec3 heading(std::sin(pose.yaw), 0.0f, -std::cos(pose.yaw));
    pose.position -= (float)dy * flySpeed * GROUND_SECONDS_PER_PIXEL * heading;
}

bool fly(ViewportPose& pose, const Camera& cam, int forward, int right, int up, float distance)
{
    const glm::vec3 direction = (float)forward * cam.view + (float)right * cam.right
        + (float)up * glm::vec3(0.0f, 1.0f, 0.0f);
    if (direction == glm::vec3(0.0f))
    {
        return false;
    }
    pose.position += distance * glm::normalize(direction);
    return true;
}

void moveAlongView(ViewportPose& pose, float distance)
{
    pose.position += distance * viewDirection(pose);
}
