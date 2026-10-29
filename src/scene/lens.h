#pragma once

// A lens prescription for the real-lens camera, read at runtime from scenes/lenses/.

#include <glm/glm.hpp>

#include <string>
#include <vector>

// One surface in mm, then the gap behind it; surfaces run from the scene side to the film. Plain data for kernels.
struct LensSurface
{
    float radius;          // positive bulges toward the scene; 0 = flat
    float thickness;       // along the axis to the next surface; on the last surface, to the film at infinity focus (0 = not known yet)
    float ior;             // n_d of what fills the gap behind the surface; 1 = air
    float abbe;            // V_d of that glass; 0 = not known, one index for every wavelength
    float apertureRadius;  // half the clear diameter
    float conic;           // k in the (1 + k) sag form: 0 is a sphere, -1 a paraboloid
    // A4, A6 ... in LensSystem::aspheric from asphericOffset on; count 0 on a spherical surface.
    int asphericOffset;
    int asphericCount;
    int isStop;            // 1 on the aperture stop, a flat opening with no glass of its own
};

struct LensSystem
{
    std::string name;
    std::string file;
    std::string source;
    std::string apertureSource;  // patents rarely print clear apertures
    std::vector<LensSurface> surfaces;  // scene side first
    // A4, A6 ... of every aspheric surface in a row (mm^-3, mm^-5 ...).
    std::vector<float> aspheric;
    int stopIndex = -1;
    float focalLength = 0.0f;    // as the file states it, mm; 0 = not stated
    float fNumber = 0.0f;        // wide open, as stated; 0 = not stated
    glm::vec2 sensorMm = glm::vec2(36.0f, 24.0f);  // film width and height
    int blades = 0;              // 0 = a round stop
    // Internal focusing, -1 when the film moves instead; focusGap is (at infinity, at the closest focus).
    int focusSurface = -1;
    glm::vec2 focusGap = glm::vec2(0.0f);
    float closeFocusMagnification = 0.0f;  // 0 = not stated

    // From a trace of the table, all in mm.
    struct Paraxial
    {
        float focalLength;          // effective focal length, from a ray close to the axis
        float backFocus;            // last surface to the paraxial focus of a distant object
        float filmDistance;         // front vertex to the film
        float entrancePupilZ;       // behind the front vertex
        float entrancePupilRadius;
        float fNumber;              // wide open
    } paraxial = {};
};

// Fills lens.paraxial; a last gap of 0 is set to focus at infinity. Throws when an axial ray cannot get through.
void analyzeLens(LensSystem& lens);

extern const std::string LENS_DIR;

// An existing file, else LENS_DIR/<argument>, with .json added when there is no extension.
std::string findLensFile(const std::string& argument);

std::vector<std::string> lensFiles();

// Format in scenes/lenses/README.md; throws std::runtime_error starting with the file's name.
LensSystem loadLens(const std::string& argument);

LensSystem parseLensJson(const std::string& text, const std::string& name);

float lensLength(const LensSystem& lens);

// Front vertex to the film that puts an axial object point at lens z objectZ (mm, negative in front) in paraxial
// focus; the infinity film distance when the point is too close to form an image.
float lensFilmDistance(const LensSystem& lens, float objectZ);

// Radius on the rear vertex's plane that camera rays are aimed within: the rear clear aperture plus the
// radial shift between that plane and the curved surface for the steepest ray from the film's corner.
float lensAimRadius(const LensSystem& lens, float halfDiagonal);
