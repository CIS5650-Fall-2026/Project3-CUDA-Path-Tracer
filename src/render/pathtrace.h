#pragma once

#include "scene/scene.h"
#include "render/tonemap.h"

// What pathtrace() reports to the viewport's ImGui panel.
struct GuiDataContainer
{
    int tracedDepth = 0;
};

// Null (headless) means no report is written.
void setGuiData(GuiDataContainer* data);
void pathtraceInit(Scene *scene);
void pathtraceFree();
// Clears the accumulation buffer; allocates nothing.
void pathtraceReset();
// Swaps in a new environment and rebuilds the light list, whose shares depend on its power.
void pathtraceSetEnvironment(const Environment& env);
// Swaps in the scene's current lens (Scene::lens), focused at the camera's focus distance; also after a focus change.
void pathtraceSetLens();
// Copies the accumulation buffer into scene->state.image.
void pathtraceDownloadImage();
// Share of camera rays the real lens blocked since the last reset; 0 without a lens.
float pathtraceLensBlockedFraction();
int pathtraceExitPupilBins();
float pathtraceExitPupilBuildMs();
// Front vertex to the film as focused, mm.
float pathtraceLensFilmDistance();
// One sample per pixel into the accumulation buffer; with a pixel buffer, also the tone-mapped average.
void pathtrace(uchar4 *pbo, int iteration);

// Feature toggles (default on). Safe to flip between iterations.
void setRussianRoulette(bool enabled);
// Off, rays go through the pixel center; the random numbers are still drawn.
void setAntialiasing(bool enabled);
void setMaterialSort(bool enabled);
// Off, shading updates paths in place; same image, every bounce launches over every pixel.
void setCompaction(bool enabled);
// Off, real-lens rays aim at the whole rear element instead of the exit pupil box; same mean, more blocked rays.
void setExitPupil(bool enabled);
// Camera weight normalized by the film-to-rear distance (irradiance on the film) instead of the center pixel.
void setPhysicalCameraWeight(bool enabled);
// Takes effect only where pathtraceNeeAvailable says it can.
void setNextEventEstimation(bool enabled);
// NEE needs OptiX and a light the scene can sample.
bool pathtraceNeeAvailable();

// Startup options, read once by pathtraceInit.
void setOptix(bool enabled);
void setOptixValidation(bool enabled);

// View transform of the viewport kernel; saveImage applies the same one on the host.
void setToneMap(ToneMapMode mode, float exposure);

// With --timing: per-bounce path counts and stage times as "BOUNCE,..." CSV lines.
void pathtraceTimingReport();
