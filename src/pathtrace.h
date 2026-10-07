#pragma once

#include <cuda_runtime.h>

class GuiDataContainer;
class Scene;

void init_data_container(GuiDataContainer* gui_data);
void pathtrace_init(Scene* scene);
void pathtrace_reset(Scene* scene);
void pathtrace_upload_env(const Scene& s);
void pathtrace_free();
void pathtrace(uchar4 *pbo, int iteration);
void copy_image_to_host();
