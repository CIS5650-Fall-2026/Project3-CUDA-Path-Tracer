cmake_minimum_required(VERSION 3.24)
project(cis565_path_tracer LANGUAGES CUDA CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++17 for host code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(UNIX)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(GL_LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
else(UNIX)
    set(EXTERNAL "${CMAKE_SOURCE_DIR}/external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(GL_LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

set(headers
    src/image.h
    src/intersections.cuh
    src/glsl_utility.hpp
    src/pathtrace.h
    src/scene.h
    src/scene_structs.h
    src/utilities.h
    src/thrust_utils.h
    src/sampling.cuh
    src/light_sampling.cuh
    src/shading.cuh
    src/config.h
    src/display.cuh
    src/sdf/sdf_scene.cuh
    src/sdf/sdf_structs.cuh
    src/sdf/sdf_utils.cuh
    src/math_utils.h
    src/camera.h
    src/gui.h
    src/gui_data.h
    src/render_settings.cuh
    src/render_settings.h
    src/bsdf/bsdf.cuh
    src/bsdf/bxdf_utils.cuh
    src/bsdf/fresnel.cuh
    src/bsdf/fresnel.cuh
    src/bsdf/dielectric.cuh
    src/bsdf/conductor.cuh
    src/bsdf/disney_diffuse.cuh
    src/bsdf/disney_metal.cuh
    src/bsdf/disney_clearcoat.cuh
    src/bsdf/disney_glass.cuh
    src/bsdf/disney_sheen.cuh
    src/bsdf/disney_bsdf.cuh
    src/bsdf/microfacet.cuh
    src/bsdf/bsdf_structs.cuh
    src/light_sampler.cuh
)

set(sources
    src/main.cpp
    src/stb.cpp
    src/image.cpp
    src/glsl_utility.cpp
    src/pathtrace.cu
    src/intersections.cu
    src/bsdf/bsdf.cu
    src/thrust_utils.cu
    src/scene.cpp
    src/utilities.cpp
    src/light_sampling.cu
    src/shading.cu
    src/display.cu
    src/sdf/sdf_scene.cu
    src/sdf/sdf_structs.cu
    src/sdf/sdf_utils.cu
    src/camera.cpp
    src/gui.cpp
)

set(imgui_headers
    external/imgui/ImGui/imconfig.h
    external/imgui/ImGui/imgui.h
    external/imgui/ImGui/imgui_impl_glfw.h
    external/imgui/ImGui/imgui_impl_opengl3.h
    external/imgui/ImGui/imgui_impl_opengl3_loader.h
    external/imgui/ImGui/imgui_internal.h
    external/imgui/ImGui/imstb_rectpack.h
    external/imgui/ImGui/imstb_textedit.h
    external/imgui/ImGui/imstb_truetype.h
)

set(imgui_sources
    external/imgui/ImGui/imgui.cpp
    external/imgui/ImGui/imgui_demo.cpp
    external/imgui/ImGui/imgui_draw.cpp
    external/imgui/ImGui/imgui_impl_glfw.cpp
    external/imgui/ImGui/imgui_impl_opengl3.cpp
    external/imgui/ImGui/imgui_tables.cpp
    external/imgui/ImGui/imgui_widgets.cpp
)

list(SORT headers)
list(SORT sources)
list(SORT imgui_headers)
list(SORT imgui_sources)

source_group("Headers" FILES ${headers})
source_group("Sources" FILES ${sources})
source_group("ImGui\\Headers" FILES ${imgui_headers})
source_group("ImGui\\Sources" FILES ${imgui_sources})

# add_subdirectory(external/imgui)
# add_subdirectory(stream_compaction)  # TODO: uncomment if using your stream compaction
add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers} ${imgui_sources} ${imgui_headers})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/external/imgui/ImGui")
target_link_libraries(${CMAKE_PROJECT_NAME}
    ${GL_LIBRARIES}
    CUDA::cudart

    # stream_compaction  # TODO: uncomment if using your stream compaction
)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)

add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2")

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--split-compile=0;-lineinfo;-src-in-ptx>;")
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--prec-div=false>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=false>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>"
)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
    "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-dlto>"
)
target_link_options(${CMAKE_PROJECT_NAME} PRIVATE
    "$<$<CONFIG:Release>:$<DEVICE_LINK:-dlto>>"
)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})
