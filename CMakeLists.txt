cmake_minimum_required(VERSION 3.24)
project(cis565_path_tracer LANGUAGES C CUDA CXX)

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

# CUDA 13's CCCL headers (Thrust/CUB) refuse to build with MSVC's traditional
# preprocessor; they require the standards-conforming one.
if(MSVC)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>")
endif()

# glm 1.0 leaves a default-constructed vector uninitialized; the 0.9.6 the
# starter code shipped zeroed it, and the code was written against that. The
# OptiX programs' nvcc command below passes the same define.
add_compile_definitions(GLM_FORCE_CTOR_INIT)
# glm 1.0's vector arithmetic goes through std::plus, std::multiplies and the
# like, constexpr host functions. Without this flag nvcc only warns (#20013)
# that device code may not call them, and then compiles every vec * scalar on
# the GPU into garbage: glm::vec3(1, 2, 3) * 2.f came back as (1, 2, 0) and
# every render was black. The OptiX nvcc command passes the same flag.
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

find_package(OpenGL REQUIRED)

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

# OptiX. The API headers are vendored under external/include/optix (from the
# OptiX SDK 9.1.0). The implementation lives in the display driver and is
# loaded at run time by optixInit(), so there is nothing to link. Host code is
# src/optix_intersect.cpp; the device programs are src/optix_programs.cu, built
# separately below.
set(OPTIX_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/include/optix")
include_directories(${OPTIX_INCLUDE_DIR})

# MikkTSpace (Morten S. Mikkelsen, zlib license), vendored from
# github.com/mmikk/MikkTSpace: generates the tangents of glTF meshes that have
# a normal map and no TANGENT attribute (src/gltf_loader.cpp). Plain C, the
# reason the project enables the C language.
set(MIKKTSPACE_DIR "${CMAKE_SOURCE_DIR}/external/mikktspace")
include_directories(${MIKKTSPACE_DIR})

# src/ by role: app/ (command line, window, viewport camera, image files),
# scene/ (scene data and loaders), render/ (the path tracing kernels and what
# they read on the device), optix/ (the OptiX intersection stage), and the
# shared utilities at the top. Includes name the folder: "scene/scene.h".
set(headers
    src/app/cli.h
    src/app/glslUtility.hpp
    src/app/image.h
    src/app/viewport_camera.h
    src/optix/optix_intersect.h
    src/optix/optix_params.h
    src/render/bsdf.h
    src/render/exit_pupil.h
    src/render/environment.h
    src/render/intersections.h
    src/render/lens_trace.h
    src/render/mesh_hit.h
    src/render/nee.h
    src/render/pathtrace.h
    src/render/pbr_surface.h
    src/render/sampling.h
    src/render/textures.h
    src/render/tonemap.h
    src/render/wavefront_ops.h
    src/scene/environment.h
    src/scene/gltf_loader.h
    src/scene/lens.h
    src/scene/scene.h
    src/scene/sceneStructs.h
    src/timing.h
    src/utilities.h
)

set(sources
    src/app/cli.cpp
    src/app/glslUtility.cpp
    src/app/image.cpp
    src/app/main.cpp
    src/app/viewport_camera.cpp
    src/optix/optix_intersect.cpp
    src/render/bsdf.cu
    src/render/exit_pupil.cu
    src/render/intersections.cu
    src/render/pathtrace.cu
    src/render/textures.cpp
    src/render/wavefront_ops.cu
    src/scene/environment.cpp
    src/scene/gltf_loader.cpp
    src/scene/lens.cpp
    src/scene/lights.cpp
    src/scene/scene.cpp
    src/stb.cpp
    src/timing.cpp
    src/utilities.cpp
)

set(imgui_headers
    src/ImGui/imconfig.h
    src/ImGui/imgui.h
    src/ImGui/imgui_impl_glfw.h
    src/ImGui/imgui_impl_opengl3.h
    src/ImGui/imgui_impl_opengl3_loader.h
    src/ImGui/imgui_internal.h
    src/ImGui/imstb_rectpack.h
    src/ImGui/imstb_textedit.h
    src/ImGui/imstb_truetype.h
)

set(imgui_sources
    src/ImGui/imgui.cpp
    src/ImGui/imgui_demo.cpp
    src/ImGui/imgui_draw.cpp
    src/ImGui/imgui_impl_glfw.cpp
    src/ImGui/imgui_impl_opengl3.cpp
    src/ImGui/imgui_tables.cpp
    src/ImGui/imgui_widgets.cpp
)

list(SORT headers)
list(SORT sources)
list(SORT imgui_headers)
list(SORT imgui_sources)

source_group(TREE "${CMAKE_SOURCE_DIR}/src" PREFIX "src" FILES ${headers} ${sources})
source_group("ImGui\\Headers" FILES ${imgui_headers})
source_group("ImGui\\Sources" FILES ${imgui_sources})

add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers} ${imgui_sources} ${imgui_headers}
    ${MIKKTSPACE_DIR}/mikktspace.c ${MIKKTSPACE_DIR}/mikktspace.h)
source_group("MikkTSpace" FILES ${MIKKTSPACE_DIR}/mikktspace.c ${MIKKTSPACE_DIR}/mikktspace.h)
target_link_libraries(${CMAKE_PROJECT_NAME} ${GL_LIBRARIES})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE src)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-src-in-ptx>")

# Device link-time optimization. Separable compilation compiles each .cu on its
# own, so device functions defined in another file (the intersection tests,
# scatterPbr) stay real calls inside the kernels unless the device link step
# can inline them. Release only; Debug and RelWithDebInfo use -G.
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-dlto>")
target_link_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<CONFIG:Release>:$<DEVICE_LINK:-dlto>>")

# OptiX device programs. They are not linked like the other .cu files: the
# driver compiles them at run time from OptiX-IR. nvcc emits that IR here,
# cmake/EmbedFile.cmake turns the file into a byte array, and the array is
# compiled into the executable, so optixModuleCreate reads the programs from
# memory rather than from a file next to the binary. sm_75 is the oldest RTX
# generation; the driver specializes the IR for the GPU it runs on.
set(optix_programs "${CMAKE_SOURCE_DIR}/src/optix/optix_programs.cu")
set(optix_ir "${CMAKE_BINARY_DIR}/optix_programs.optixir")
set(optix_ir_cpp "${CMAKE_BINARY_DIR}/optix_programs_ir.cpp")
# nvcc also writes every header the programs include into a depfile (-MD),
# which CMake reads, so editing any of them rebuilds the IR.
add_custom_command(
    OUTPUT ${optix_ir}
    COMMAND ${CMAKE_CUDA_COMPILER} -optix-ir -arch=sm_75 -rdc=true -std=c++17 -m64 -lineinfo
            -ccbin "${CMAKE_CXX_COMPILER}" -DGLM_FORCE_CTOR_INIT --expt-relaxed-constexpr
            "-I${OPTIX_INCLUDE_DIR}" "-I${GLM_INCLUDE_DIRS}" "-I${CMAKE_SOURCE_DIR}/src"
            -MD -MF ${optix_ir}.d -MT ${optix_ir}
            -o ${optix_ir} ${optix_programs}
    DEPENDS ${optix_programs}
    DEPFILE ${optix_ir}.d
    COMMENT "nvcc -optix-ir optix_programs.cu"
    VERBATIM)
add_custom_command(
    OUTPUT ${optix_ir_cpp}
    COMMAND ${CMAKE_COMMAND} -DINPUT=${optix_ir} -DOUTPUT=${optix_ir_cpp} -DSYMBOL=optix_programs_ir
            -P "${CMAKE_SOURCE_DIR}/cmake/EmbedFile.cmake"
    DEPENDS ${optix_ir} "${CMAKE_SOURCE_DIR}/cmake/EmbedFile.cmake"
    COMMENT "embedding optix_programs.optixir"
    VERBATIM)
target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${optix_ir_cpp} ${optix_programs})
# Listed so it shows in the IDE; nvcc must not build it as an ordinary CUDA file.
set_source_files_properties(${optix_programs} PROPERTIES HEADER_FILE_ONLY ON)
source_group("src\\optix" FILES ${optix_programs})

# Host-side check of scatterPbr (tests/bsdf_test.cu), not part of the default
# build. It includes render/bsdf.cu itself, so it is one translation unit:
#   cmake --build build --config Release --target bsdf_test
#   build/bin/Release/bsdf_test.exe
# The exit code is nonzero when a check fails.
add_executable(bsdf_test EXCLUDE_FROM_ALL tests/bsdf_test.cu)
target_include_directories(bsdf_test PRIVATE src)
set_target_properties(bsdf_test PROPERTIES CUDA_ARCHITECTURES native)
if(MSVC)
    # C4819: the CUDA headers hold characters outside the system code page.
    target_compile_options(bsdf_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>")
endif()

# Host-side check of the lens loaders (tests/lens_test.cu), built the same way:
#   cmake --build build --config Release --target lens_test
#   build/bin/Release/lens_test.exe        (from the repository root, it reads scenes/lenses/)
add_executable(lens_test EXCLUDE_FROM_ALL tests/lens_test.cu src/scene/lens.cpp src/render/exit_pupil.cu src/utilities.cpp)
target_include_directories(lens_test PRIVATE src)
set_target_properties(lens_test PROPERTIES CUDA_ARCHITECTURES native)
if(MSVC)
    target_compile_options(lens_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>")
endif()

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})

