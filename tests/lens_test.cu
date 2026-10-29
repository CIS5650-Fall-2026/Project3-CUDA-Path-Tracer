// Checks the lens loaders and the lens trace against PBRT-v3's double Gauss and the Noct table; nonzero exit on a failure.
// Run it from the repository root, where scenes/lenses/ is.
#include "scene/lens.h"
#include "render/lens_trace.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thrust/random.h>

static int checks = 0;
static int failures = 0;

static void check(bool ok, const char* what)
{
    ++checks;
    failures += ok ? 0 : 1;
    printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
}

static bool within(float a, float b, float tolerance = 1e-4f)
{
    return std::fabs(a - b) <= tolerance;
}

// PBRT-v3's dgauss.50mm.dat: radius, thickness, index, clear diameter in mm; the stop is radius 0 with index 0.
static const float pbrtDgauss[11][4] = {
    {29.475f, 3.76f, 1.67f, 25.2f},
    {84.83f, 0.12f, 1.0f, 25.2f},
    {19.275f, 4.025f, 1.67f, 23.0f},
    {40.77f, 3.275f, 1.699f, 23.0f},
    {12.75f, 5.705f, 1.0f, 18.0f},
    {0.0f, 4.5f, 0.0f, 17.1f},
    {-14.495f, 1.18f, 1.603f, 17.0f},
    {40.77f, 6.065f, 1.658f, 20.0f},
    {-20.385f, 0.19f, 1.0f, 20.0f},
    {437.065f, 3.22f, 1.717f, 20.0f},
    {-39.73f, 0.0f, 1.0f, 20.0f},
};

static void checkDgauss(const LensSystem& lens, const char* label)
{
    char line[256];
    snprintf(line, sizeof line, "%s: 11 surfaces (%d)", label, (int)lens.surfaces.size());
    check(lens.surfaces.size() == 11, line);
    if (lens.surfaces.size() != 11)
    {
        return;
    }
    snprintf(line, sizeof line, "%s: the stop is surface 6 (%d)", label, lens.stopIndex + 1);
    check(lens.stopIndex == 5 && lens.surfaces[5].isStop && lens.surfaces[5].radius == 0.0f, line);
    int stops = 0;
    for (const LensSurface& s : lens.surfaces)
    {
        stops += s.isStop;
    }
    check(stops == 1, "dgauss: exactly one surface is the stop");
    check(within(lens.surfaces[0].radius, 29.475f) && within(lens.surfaces[0].thickness, 3.76f)
        && within(lens.surfaces[0].ior, 1.67f), "dgauss: first surface r 29.475, d 3.76, n 1.67");
    check(within(lens.surfaces[0].apertureRadius, 12.6f) && within(lens.surfaces[5].apertureRadius, 8.55f),
        "dgauss: apertures are radii, half the file's diameters (12.6 front, 8.55 stop)");
    check(lens.surfaces[1].ior == 1.0f && lens.surfaces[5].ior == 1.0f && lens.surfaces[10].ior == 1.0f,
        "dgauss: air gaps and the stop read as index 1");
    check(within(lens.surfaces[10].radius, -39.73f), "dgauss: last surface r -39.73");
    float glass = 0.0f;
    for (size_t i = 0; i + 1 < lens.surfaces.size(); ++i)
    {
        glass += lens.surfaces[i].thickness;
    }
    snprintf(line, sizeof line, "%s: front vertex to the last surface 32.04 mm (%.3f)", label, glass);
    check(within(glass, 32.04f, 1e-3f), line);
    for (const LensSurface& s : lens.surfaces)
    {
        if (s.conic != 0.0f || s.asphericCount != 0)
        {
            check(false, "dgauss: a surface has aspheric terms");
            return;
        }
    }
    check(lens.aspheric.empty(), "dgauss: every surface is spherical, no coefficients stored");
}

static void checkDgaussTrace(const LensSystem& lens, const char* label)
{
    char line[256];
    const LensSystem::Paraxial& p = lens.paraxial;
    const LensSurface* s = lens.surfaces.data();
    const int n = (int)lens.surfaces.size();

    snprintf(line, sizeof line, "%s: traced focal length 50.36 mm (%.3f)", label, p.focalLength);
    check(within(p.focalLength, 50.36f, 0.05f), line);
    snprintf(line, sizeof line, "%s: film 68.146 mm behind the front vertex (%.3f), back focus 36.11 (%.3f)", label,
        p.filmDistance, p.backFocus);
    check(within(p.filmDistance, 68.146f, 0.01f) && within(p.backFocus, 36.106f, 0.01f), line);
    snprintf(line, sizeof line, "%s: entrance pupil radius 12.48 (%.3f) against a stop radius of 8.55, so f/2.02 (%.3f)",
        label, p.entrancePupilRadius, p.fNumber);
    check(within(p.entrancePupilRadius, 12.48f, 0.02f) && within(p.fNumber, 2.02f, 0.01f), line);
    // The front group magnifies the stop 1.46x, so the pupil lies behind the stop (16.885 mm).
    snprintf(line, sizeof line, "%s: entrance pupil %.2f mm behind the front vertex, behind the stop (16.89) and before the film",
        label, p.entrancePupilZ);
    check(p.entrancePupilZ > 16.885f && p.entrancePupilZ < p.filmDistance, line);

    // Film to scene, as the kernel traces: from a film point through a point of the rear element.
    thrust::default_random_engine rng(7);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    const float rearZ = p.filmDistance - s[n - 1].thickness;
    const float heights[3] = {0.0f, 18.0f, 21.63f};  // center, 18 mm, the 36x24 corner
    const float expected[3] = {0.85f, 0.29f, 0.16f};
    bool allFinite = true;
    for (int h = 0; h < 3; ++h)
    {
        const int N = 20000;
        int passed = 0;
        for (int k = 0; k < N; ++k)
        {
            const float r = s[n - 1].apertureRadius * sqrtf(u01(rng));
            const float phi = 6.2831853f * u01(rng);
            glm::vec3 o(heights[h], 0.0f, p.filmDistance);
            glm::vec3 d = glm::normalize(glm::vec3(r * cosf(phi), r * sinf(phi), rearZ) - o);
            if (traceLens(s, n - 1, 0, rearZ, o, d))
            {
                ++passed;
                const float len = glm::length(d);
                allFinite = allFinite && std::isfinite(o.x + o.y + o.z + d.x + d.y + d.z) && within(len, 1.0f, 1e-4f)
                    && d.z < 0.0f;
            }
        }
        const float share = (float)passed / N;
        snprintf(line, sizeof line, "%s: %.0f%% of rays from %.1f mm on the film get out (expected about %.0f%%)", label,
            100.0f * share, heights[h], 100.0f * expected[h]);
        check(within(share, expected[h], 0.05f), line);
    }
    check(allFinite, "dgauss: every ray that gets out is finite, unit length and heads toward the scene");

    // An 18 mm field fan across the entrance pupil clips at the front and rear rims, never at the stop.
    const float theta = atanf(18.0f / p.focalLength);
    const glm::vec3 dir(-sinf(theta), 0.0f, cosf(theta));
    int blockedAt[16] = {};
    int entered = 0;
    for (int k = -40; k <= 40; ++k)
    {
        const float h = p.entrancePupilRadius * k / 40.0f;
        glm::vec3 o = glm::vec3(h, 0.0f, p.entrancePupilZ) - 50.0f * dir;
        glm::vec3 d = dir;
        float z = 0.0f;
        int stoppedBy = -1;
        for (int i = 0; i < n; ++i)
        {
            if (!traceLens(s, i, i, z, o, d))
            {
                stoppedBy = i;
                break;
            }
            z += s[i].thickness;
        }
        if (stoppedBy != 0)
        {
            ++entered;
        }
        if (stoppedBy >= 0)
        {
            ++blockedAt[stoppedBy];
        }
    }
    std::string where;
    int passed = entered;
    for (int i = 1; i < n; ++i)
    {
        passed -= blockedAt[i];
        if (blockedAt[i] > 0)
        {
            char part[32];
            snprintf(part, sizeof part, " %d at surface %d,", blockedAt[i], i + 1);
            where += part;
        }
    }
    snprintf(line, sizeof line, "%s: an 18 mm field fan of 81 rays: %d blocked at the front rim,%s %d pass, none at the stop",
        label, blockedAt[0], where.c_str(), passed);
    check(blockedAt[0] > 0 && blockedAt[n - 1] > 0 && blockedAt[lens.stopIndex] == 0, line);

    glm::vec3 o(0.0f, 0.0f, -1.0f);
    glm::vec3 d(0.0f, 0.0f, 1.0f);
    check(traceLens(s, 0, n - 1, 0.0f, o, d) && within(d.x, 0.0f, 1e-6f) && within(d.y, 0.0f, 1e-6f) && within(o.x, 0.0f, 1e-6f),
        "dgauss: the axial ray leaves along the axis");
}

int main()
{
    char line[256];
    try
    {
        const LensSystem js = loadLens("dgauss.50mm");
        check(js.file == LENS_DIR + "dgauss.50mm.json", "dgauss: findLensFile resolved the bare name to the .json");
        checkDgauss(js, "dgauss");
        checkDgaussTrace(js, "dgauss");
        bool matchesPbrt = true;
        for (int i = 0; i < 11; ++i)
        {
            const float* row = pbrtDgauss[i];
            const LensSurface& s = js.surfaces[i];
            const bool same = within(s.radius, row[0]) && (i == 10 || within(s.thickness, row[1]))
                && within(s.ior, row[2] < 1.0f ? 1.0f : row[2]) && within(s.apertureRadius, 0.5f * row[3])
                && (s.isStop != 0) == (row[0] == 0.0f && row[2] == 0.0f);
            if (!same)
            {
                snprintf(line, sizeof line, "dgauss: surface %d differs from PBRT's row", i + 1);
                check(false, line);
                matchesPbrt = false;
            }
        }
        check(matchesPbrt, "dgauss: every surface matches PBRT-v3's dgauss.50mm.dat row");
        const float abbe[11] = {47.1f, 0, 47.1f, 30.1f, 0, 0, 38.4f, 57.0f, 0, 48.1f, 0};
        bool abbeOk = true;
        for (int i = 0; i < 11; ++i)
        {
            abbeOk = abbeOk && within(js.surfaces[i].abbe, abbe[i]);
            abbeOk = abbeOk && ((js.surfaces[i].ior > 1.0f) == (abbe[i] > 0.0f));
        }
        check(abbeOk, "dgauss: Abbe numbers on the six glass rows only");
        snprintf(line, sizeof line, "dgauss: film 68.146 mm behind the front vertex (%.3f)", lensLength(js));
        check(within(lensLength(js), 68.146f, 1e-3f), line);
        check(within(js.focalLength, 50.0f) && within(js.fNumber, 2.0f) && js.blades == 0, "dgauss: stated f 50, f/2, round stop");
        check(js.focusSurface == -1, "dgauss: no focusing gap");

        LensSystem noFilm = js;
        noFilm.surfaces.back().thickness = 0.0f;
        analyzeLens(noFilm);
        snprintf(line, sizeof line, "dgauss: a film gap of 0 becomes the paraxial back focus, 36.11 (%.3f)",
            noFilm.surfaces.back().thickness);
        check(within(noFilm.surfaces.back().thickness, 36.106f, 0.01f), line);

        const LensSystem noct = loadLens("noct-z58");
        snprintf(line, sizeof line, "noct: 28 surfaces (%d), stop at 14 (%d)", (int)noct.surfaces.size(), noct.stopIndex + 1);
        check(noct.surfaces.size() == 28 && noct.stopIndex == 13, line);
        if (noct.surfaces.size() == 28)
        {
            int aspheric = 0;
            for (int i = 0; i < 28; ++i)
            {
                const bool has = noct.surfaces[i].asphericCount != 0;
                aspheric += has;
                const bool expected = i == 0 || i == 19 || i == 27;
                if (has != expected)
                {
                    snprintf(line, sizeof line, "noct: surface %d aspheric", i + 1);
                    check(false, line);
                }
            }
            check(aspheric == 3, "noct: aspheres on surfaces 1, 20 and 28 only");
            const LensSurface& s1 = noct.surfaces[0];
            const LensSurface& s20 = noct.surfaces[19];
            const LensSurface& s28 = noct.surfaces[27];
            check(s1.asphericOffset == 0 && s1.asphericCount == 4 && s20.asphericOffset == 4 && s20.asphericCount == 4
                && s28.asphericOffset == 8 && s28.asphericCount == 6 && noct.aspheric.size() == 14,
                "noct: 4 + 4 + 6 coefficients, each surface's run following the last");
            check(s1.conic == 0.0f && within(noct.aspheric[s1.asphericOffset], -3.82177e-7f, 1e-12f)
                && within(noct.aspheric[s1.asphericOffset + 3], -1.32266e-18f, 1e-23f),
                "noct: surface 1 k 0, A4 -3.82177e-7, A10 -1.32266e-18");
            check(within(noct.aspheric[s20.asphericOffset + 1], -4.51771e-10f, 1e-15f), "noct: surface 20 A6 -4.51771e-10");
            check(within(noct.aspheric[s28.asphericOffset + 5], -1.70470e-19f, 1e-24f), "noct: surface 28 A14 -1.70470e-19");
            check(noct.focusSurface == 21 && within(noct.focusGap.x, 2.68f) && within(noct.focusGap.y, 21.29f)
                && within(noct.surfaces[21].thickness, 2.68f), "noct: surface 22 focuses, 2.68 mm at infinity, 21.29 close, table at infinity");
            check(within(noct.closeFocusMagnification, -0.194f), "noct: close focus magnification -0.194");
            check(within(noct.surfaces[0].ior, 1.90265f) && within(noct.surfaces[0].abbe, 35.77f), "noct: surface 1 n 1.90265, V 35.77");
            check(within(noct.surfaces[13].apertureRadius, 0.5f * 47.918f), "noct: stop radius 23.959");
            snprintf(line, sizeof line, "noct: 160.2 mm to the film (%.2f)", lensLength(noct));
            check(within(lensLength(noct), 160.2f, 1e-2f), line);
            check(noct.blades == 11 && within(noct.sensorMm.x, 36.0f) && within(noct.sensorMm.y, 24.0f)
                && within(noct.focalLength, 59.62f) && within(noct.fNumber, 0.98f), "noct: 11 blades, 36x24 sensor, f 59.62, f/0.98");
            check(noct.apertureSource == "estimated", "noct: apertures marked as estimated");
            // The paraxial ray never feels the aspheric terms; the f-number tolerance waits for the aspheric trace.
            snprintf(line, sizeof line, "noct: traced f 59.62 (%.2f), back focus 16.55 (%.2f), f/0.98 (%.3f)",
                noct.paraxial.focalLength, noct.paraxial.backFocus, noct.paraxial.fNumber);
            check(within(noct.paraxial.focalLength, 59.62f, 0.1f) && within(noct.paraxial.backFocus, 16.55f, 0.1f)
                && within(noct.paraxial.fNumber, 0.98f, 0.05f), line);
        }
    }
    catch (const std::exception& e)
    {
        printf("FAIL  %s\n", e.what());
        ++checks;
        ++failures;
    }

    struct Bad
    {
        const char* what;
        const char* text;
    };
    const Bad bad[] = {
        {"json without a stop", "{\"surfaces\": [{\"radius\": 10, \"thickness\": 1, \"aperture\": 5}, {\"radius\": -10, \"thickness\": 0, \"aperture\": 5}]}"},
        {"json with two stops", "{\"surfaces\": [{\"radius\": 0, \"thickness\": 1, \"aperture\": 5, \"stop\": true}, {\"radius\": 0, \"thickness\": 0, \"aperture\": 5, \"stop\": true}]}"},
        {"json with a zero gap inside", "{\"surfaces\": [{\"radius\": 10, \"thickness\": 0, \"aperture\": 5}, {\"radius\": 0, \"thickness\": 0, \"aperture\": 5, \"stop\": true}]}"},
        {"json whose aspheric is not an array", "{\"surfaces\": [{\"radius\": 0, \"thickness\": 1, \"aperture\": 5, \"stop\": true, \"aspheric\": 1e-6}]}"},
        {"json missing a radius", "{\"surfaces\": [{\"thickness\": 1, \"aperture\": 5, \"stop\": true}]}"},
        {"json that is not json", "{surfaces: ["},
    };
    for (const Bad& b : bad)
    {
        bool refused = false;
        std::string why;
        try
        {
            parseLensJson(b.text, "test");
        }
        catch (const std::runtime_error& e)
        {
            refused = true;
            why = e.what();
        }
        char line[512];
        snprintf(line, sizeof line, "refused %s: %s", b.what, why.c_str());
        check(refused, line);
    }
    bool missing = false;
    try
    {
        loadLens("no-such-lens");
    }
    catch (const std::runtime_error&)
    {
        missing = true;
    }
    check(missing, "refused a lens file that does not exist");

    printf("%d of %d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
