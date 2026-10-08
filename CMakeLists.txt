cmake_minimum_required(VERSION 3.24)
project(cis565_path_tracer LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

# CUDA Setup
find_package(CUDAToolkit REQUIRED)

# Thrust Setup
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

# GLFW Setup
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "")
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "")
set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "")
add_subdirectory(external/glfw)

# GLM Setup
add_subdirectory(external/glm)

# JSON Setup
add_subdirectory(external/json)

# Native File Dialogs Setup
add_subdirectory(external/NFDe)

# Vulkan Setup
find_package(Vulkan REQUIRED)

file(GLOB_RECURSE SOURCE_FILES
	"src/*.cpp"
	"src/*.cu"
	"src/*.h"
	"src/*.hpp"
)

# ImGui Sources
file(GLOB IMGUI_SOURCES
	external/imgui/*.cpp
	external/imgui/backends/imgui_impl_glfw.cpp
	external/imgui/backends/imgui_impl_vulkan.cpp
)

add_executable(cis565_path_tracer ${SOURCE_FILES} ${IMGUI_SOURCES})

set_target_properties(cis565_path_tracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
	src/
	external/stb/
	external/imgui/
	external/imgui/backends
	external/tinygltf/
	external/tinyobjloader/
	${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE 
	GLFW_INCLUDE_VULKAN 
	GLM_FORCE_PURE
	$<$<CONFIG:Debug>:-G> 
	NOMINMAX 
)

target_link_libraries(${CMAKE_PROJECT_NAME} 
PRIVATE 
	Vulkan::Vulkan
	glfw
	glm::glm
	CUDA::cudart
	CUDA::cuda_driver
	Thrust
	nlohmann_json::nlohmann_json
	nfd
)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
		--expt-relaxed-constexpr
		--diag-suppress=20012 # Hide warnings from glm
		-Xcompiler=/std:c++17
		-Xcompiler=/Zc:preprocessor
        $<$<CONFIG:Debug>:-G>
        $<$<CONFIG:Release>:-O3>
        --generate-line-info
    >
)

add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/scenes"
            "$<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>/scenes"
    COMMENT "Copying scene files to the build destination directory"
)

if(WIN32)
	target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE VK_USE_PLATFORM_WIN32_KHR)
elseif(UNIX AND NOT APPLE)
	target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE VK_USE_PLATFORM_XLIB_KHR)
endif()	