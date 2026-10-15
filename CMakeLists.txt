cmake_minimum_required(VERSION 3.24)
project(cis565_path_tracer LANGUAGES C CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(EXTERNAL "${CMAKE_SOURCE_DIR}/external")

#--------------------------------------------------------------------
# Dependencies
#--------------------------------------------------------------------

# Vulkan SDK: headers + glslc. Vulkan::Headers (not Vulkan::Vulkan) because volk
# loads the loader at runtime; we never link vulkan-1.lib.
find_package(Vulkan REQUIRED)

if(UNIX)
    find_package(glfw3 REQUIRED)
    set(GLFW_LIBRARIES glfw)
else()
    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)
    set(GLFW_LIBRARIES ${GLFW_LIBRARY})
endif()

set(GLM_ROOT_DIR ${EXTERNAL})
find_package(GLM REQUIRED)

#--------------------------------------------------------------------
# Vendored third-party libraries (see external/VERSIONS.md)
#--------------------------------------------------------------------

# volk: loads Vulkan function pointers, including the RT extension entry points.
# VK_NO_PROTOTYPES is PUBLIC so every consumer sees it before including vulkan.h.
add_library(volk STATIC ${EXTERNAL}/volk/volk.c)
target_include_directories(volk PUBLIC ${EXTERNAL}/volk)
target_compile_definitions(volk PUBLIC VK_NO_PROTOTYPES)
if(WIN32)
    target_compile_definitions(volk PUBLIC VK_USE_PLATFORM_WIN32_KHR)
endif()
target_link_libraries(volk PUBLIC Vulkan::Headers)

# vk-bootstrap: instance/device/swapchain creation.
add_library(vk-bootstrap STATIC ${EXTERNAL}/vk-bootstrap/VkBootstrap.cpp)
target_include_directories(vk-bootstrap PUBLIC ${EXTERNAL}/vk-bootstrap)
target_link_libraries(vk-bootstrap PUBLIC volk)

# VMA and tinygltf are header-only; their implementations are compiled in
# src/ files.
add_library(vma INTERFACE)
target_include_directories(vma INTERFACE ${EXTERNAL}/vma)
target_link_libraries(vma INTERFACE volk)

add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE ${EXTERNAL}/tinygltf)

# Dear ImGui + GLFW/Vulkan backends.
set(imgui_sources
    ${EXTERNAL}/imgui/imgui.cpp
    ${EXTERNAL}/imgui/imgui_demo.cpp
    ${EXTERNAL}/imgui/imgui_draw.cpp
    ${EXTERNAL}/imgui/imgui_tables.cpp
    ${EXTERNAL}/imgui/imgui_widgets.cpp
    ${EXTERNAL}/imgui/imgui_impl_glfw.cpp
    ${EXTERNAL}/imgui/imgui_impl_vulkan.cpp
)
add_library(imgui STATIC ${imgui_sources})
target_include_directories(imgui PUBLIC ${EXTERNAL}/imgui ${GLFW_INCLUDE_DIR})

# Tells the Vulkan backend to call through volk's pointers instead of prototypes.
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK)
target_link_libraries(imgui PUBLIC volk ${GLFW_LIBRARIES})

set_target_properties(volk vk-bootstrap imgui PROPERTIES FOLDER "external")

#--------------------------------------------------------------------
# Project sources
#--------------------------------------------------------------------

set(headers
    src/icosphere.h
    src/cube.h
    src/image.h
    src/scene.h
    src/sceneStructs.h
    src/sceneUtils.h
    src/utilities.h
    src/gpu/shared.h
    src/vk/context.h
    src/vk/swapchain.h
    src/vk/renderer.h
    src/vk/pipeline.h
    src/vk/gui.h
    src/vk/gpuscene.h
    src/vk/utils/buffer.h
)

set(sources
    src/main.cpp
    src/image.cpp
    src/scene.cpp
    src/sceneUtils.cpp
    src/stb.cpp
    src/utilities.cpp
    src/vk/context.cpp
    src/vk/swapchain.cpp
    src/vk/vma_impl.cpp
    src/vk/renderer.cpp
    src/vk/pipeline.cpp
    src/vk/gui.cpp
    src/vk/gpuscene.cpp
    src/vk/utils/buffer.cpp
)

add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})

source_group(TREE ${CMAKE_SOURCE_DIR}/src FILES ${sources} ${headers})

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${GLM_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIR}
    ${EXTERNAL}/include
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
    volk
    vk-bootstrap
    vma
    tinygltf
    imgui
    ${GLFW_LIBRARIES}
)

if(WIN32)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

if(MSVC)
    # enable parallel compilation
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE /MP)

    # Silence third party build warnings
    set_source_files_properties(src/vk/vma_impl.cpp PROPERTIES COMPILE_OPTIONS "/w")
    set_source_files_properties(src/stb.cpp PROPERTIES COMPILE_OPTIONS "/w")
endif()

# Run from the source dir so relative scene paths work, and pass a
# default scene
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    VS_DEBUGGER_COMMAND_ARGUMENTS "scenes/cornellGlassSphere.json"
)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})

#--------------------------------------------------------------------
# Shader compilation (GLSL -> SPIR-V)
#--------------------------------------------------------------------

if(NOT Vulkan_GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found; check the Vulkan SDK installation")
endif()

set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/shaders)

# SPIR-V goes in the build tree, and its absolute path is compiled in as SHADER_DIR, so the
# program finds its shaders regardless of the working directory
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)

# List shader entry points (excluding files that are only #included, because glslc compiles them
# as part of whatever shader actually includes them)
set(shader_entry_points
    raygen.rgen
    miss.rmiss
    mesh.rchit
)

if(shader_entry_points)
    file(MAKE_DIRECTORY ${SHADER_OUT_DIR})
    set(spv_outputs)
    set(shader_sources)
    foreach(shader ${shader_entry_points})
        set(spv ${SHADER_OUT_DIR}/${shader}.spv)
        set(dep ${SHADER_OUT_DIR}/${shader}.d)

        # -I src: for #include "gpu/shared.h"
        # -g (every configuration): lets Nsight Graphics map profiling data to GLSL source
        # -MD/-MF/-MT: also writes a Makefile-style file listing every #included file; DEPFILE
        # reruns this command when any of them changes, not just the entry point
        add_custom_command(
            OUTPUT ${spv}
            COMMAND ${Vulkan_GLSLC_EXECUTABLE}
                    --target-env=vulkan1.3
                    -I ${SHADER_SRC_DIR}
                    -I ${CMAKE_SOURCE_DIR}/src
                    -g
                    -MD -MF ${dep} -MT ${spv}
                    ${SHADER_SRC_DIR}/${shader}
                    -o ${spv}
            DEPENDS ${SHADER_SRC_DIR}/${shader}
            DEPFILE ${dep}
            COMMENT "glslc ${shader}"
            VERBATIM
        )
        list(APPEND spv_outputs ${spv})
        list(APPEND shader_sources ${SHADER_SRC_DIR}/${shader})
    endforeach()

    # Add GLSL helpers that do not contain entry points here so they're
    # visible in the VS solution
    list(APPEND shader_sources ${SHADER_SRC_DIR}/common.glsl)
    list(APPEND shader_sources ${SHADER_SRC_DIR}/math.glsl)
    list(APPEND shader_sources ${SHADER_SRC_DIR}/sampleWarping.glsl)
    list(APPEND shader_sources ${SHADER_SRC_DIR}/rng.glsl)
    list(APPEND shader_sources ${SHADER_SRC_DIR}/bsdf.glsl)

    add_custom_target(shaders DEPENDS ${spv_outputs} SOURCES ${shader_sources})
    add_dependencies(${CMAKE_PROJECT_NAME} shaders)
    set_target_properties(shaders PROPERTIES FOLDER "shaders")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE SHADER_DIR="${SHADER_OUT_DIR}")
endif()
