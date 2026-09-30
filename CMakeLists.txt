cmake_minimum_required(VERSION 3.24)

project(cis565_path_tracer LANGUAGES CXX)

option(BUILD_RENDERER "Build the CUDA renderer and headless executable" ON)
option(BUILD_PREVIEW "Build the interactive OpenGL preview" ON)
set(PROJECT3_BUILD_ID "" CACHE STRING
    "Override the renderer source-content identifier recorded in outputs")

include(CTest)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release MinSizeRel RelWithDebInfo)
endif()

add_library(cis565_dependencies INTERFACE)
target_include_directories(cis565_dependencies SYSTEM INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/external/include")
target_include_directories(cis565_dependencies INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}")

set(cis565_core_sources
    src/checkpoint.cpp
    src/geometry.cpp
    src/image.cpp
    src/scene.cpp
    src/stb.cpp
    src/utilities.cpp)

add_library(cis565_core STATIC ${cis565_core_sources})
target_link_libraries(cis565_core PUBLIC cis565_dependencies)
target_compile_features(cis565_core PUBLIC cxx_std_17)
set_target_properties(cis565_core PROPERTIES FOLDER "Libraries")

if(MSVC)
    target_compile_options(cis565_core PRIVATE /W4)
else()
    target_compile_options(cis565_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_RENDERER)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "BUILD_RENDERER=ON requires a CUDA compiler. Configure with "
            "-DBUILD_RENDERER=OFF for host-only libraries and tests.")
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    set(cis565_renderer_sources
        src/pathtrace.cu
        src/interactions.cu
        src/intersections.cu
        stream_compaction/device_scan.cu)

    set(cis565_build_fingerprint_sources
        CMakeLists.txt
        src/checkpoint.cpp
        src/checkpoint.h
        src/geometry.cpp
        src/geometry.cuh
        src/geometry.h
        src/headless.cpp
        src/image.cpp
        src/image.h
        src/interactions.cu
        src/interactions.h
        src/intersections.cu
        src/intersections.h
        src/main.cpp
        src/pathtrace.cu
        src/pathtrace.h
        src/scene.cpp
        src/scene.h
        src/sceneStructs.h
        src/stb.cpp
        src/utilities.cpp
        src/utilities.h
        stream_compaction/device_scan.cu
        stream_compaction/device_scan.cuh)
    set(cis565_source_hash_material "")
    foreach(cis565_source IN LISTS cis565_build_fingerprint_sources)
        set(cis565_source_path "${CMAKE_CURRENT_SOURCE_DIR}/${cis565_source}")
        if(NOT EXISTS "${cis565_source_path}")
            message(FATAL_ERROR "Required source is missing: ${cis565_source}")
        endif()
        file(SHA256 "${cis565_source_path}" cis565_file_hash)
        string(APPEND cis565_source_hash_material
            "${cis565_source}:${cis565_file_hash}\n")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${cis565_source_path}")
    endforeach()
    string(SHA256 cis565_content_hash "${cis565_source_hash_material}")
    if(PROJECT3_BUILD_ID STREQUAL "")
        set(cis565_source_identifier "sha256-${cis565_content_hash}")
    else()
        if(NOT PROJECT3_BUILD_ID MATCHES "^[A-Za-z0-9._+-]+$")
            message(FATAL_ERROR
                "PROJECT3_BUILD_ID may contain only letters, digits, '.', '_', '+', and '-'")
        endif()
        set(cis565_source_identifier "${PROJECT3_BUILD_ID}")
    endif()

    add_library(cis565_renderer STATIC ${cis565_renderer_sources})
    target_link_libraries(cis565_renderer
        PUBLIC cis565_core CUDA::cudart)
    target_include_directories(cis565_renderer PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/stream_compaction")
    target_compile_features(cis565_renderer PUBLIC cxx_std_17 cuda_std_17)
    if(CMAKE_CUDA_ARCHITECTURES)
        string(REPLACE ";" "," cis565_cuda_architecture_id
            "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set(cis565_cuda_architecture_id "toolkit-default")
    endif()
    set_target_properties(cis565_renderer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
        FOLDER "Libraries")

    set(cis565_renderer_compile_options
        "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:-G;-src-in-ptx>"
        "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-lineinfo;-src-in-ptx>")
    target_compile_options(cis565_renderer PRIVATE
        ${cis565_renderer_compile_options})

    set(cis565_numerical_material
        "cxx_standard=${CMAKE_CXX_STANDARD}\ncuda_standard=${CMAKE_CUDA_STANDARD}\n"
        "cuda_architectures=${CMAKE_CUDA_ARCHITECTURES}\n"
        "renderer_options=${cis565_renderer_compile_options}\n"
        "cuda_separable=ON\ncuda_resolve_device_symbols=ON\n")
    string(APPEND cis565_numerical_material
        "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}\n"
        "CMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}\n")
    set(cis565_flag_configurations
        DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    foreach(cis565_configuration IN LISTS CMAKE_CONFIGURATION_TYPES)
        string(TOUPPER "${cis565_configuration}" cis565_configuration_upper)
        list(APPEND cis565_flag_configurations "${cis565_configuration_upper}")
    endforeach()
    if(CMAKE_BUILD_TYPE)
        string(TOUPPER "${CMAKE_BUILD_TYPE}" cis565_build_type_upper)
        list(APPEND cis565_flag_configurations "${cis565_build_type_upper}")
    endif()
    list(REMOVE_DUPLICATES cis565_flag_configurations)
    foreach(cis565_configuration IN LISTS cis565_flag_configurations)
        set(cis565_cxx_flags_variable
            "CMAKE_CXX_FLAGS_${cis565_configuration}")
        set(cis565_cuda_flags_variable
            "CMAKE_CUDA_FLAGS_${cis565_configuration}")
        string(APPEND cis565_numerical_material
            "${cis565_cxx_flags_variable}=${${cis565_cxx_flags_variable}}\n"
            "${cis565_cuda_flags_variable}=${${cis565_cuda_flags_variable}}\n")
    endforeach()
    get_directory_property(cis565_directory_options COMPILE_OPTIONS)
    get_directory_property(cis565_directory_definitions COMPILE_DEFINITIONS)
    string(APPEND cis565_numerical_material
        "directory_options=${cis565_directory_options}\n"
        "directory_definitions=${cis565_directory_definitions}\n")
    foreach(cis565_target IN ITEMS
            cis565_dependencies cis565_core cis565_renderer CUDA::cudart)
        foreach(cis565_property IN ITEMS
                COMPILE_OPTIONS COMPILE_DEFINITIONS
                INTERFACE_COMPILE_OPTIONS INTERFACE_COMPILE_DEFINITIONS)
            get_target_property(cis565_property_value
                ${cis565_target} ${cis565_property})
            if(cis565_property_value)
                string(APPEND cis565_numerical_material
                    "${cis565_target}.${cis565_property}=${cis565_property_value}\n")
            endif()
        endforeach()
    endforeach()
    string(SHA256 cis565_numerical_flags_hash
        "${cis565_numerical_material}")
    set(cis565_renderer_build_id
        "cis565-project3-renderer-v1_source-${cis565_source_identifier}_cxx-${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}_cuda-${CMAKE_CUDA_COMPILER_ID}-${CMAKE_CUDA_COMPILER_VERSION}_config-$<CONFIG>_arch-${cis565_cuda_architecture_id}_flags-${cis565_numerical_flags_hash}")
    target_compile_definitions(cis565_renderer PRIVATE
        "CIS565_RENDERER_BUILD_ID=\"${cis565_renderer_build_id}\"")

    add_executable(cis565_path_tracer_headless src/headless.cpp)
    target_link_libraries(cis565_path_tracer_headless PRIVATE
        cis565_renderer cis565_core CUDA::cudart)
    target_compile_features(cis565_path_tracer_headless PRIVATE cxx_std_17)
    set_target_properties(cis565_path_tracer_headless PROPERTIES
        FOLDER "Applications")

    if(BUILD_PREVIEW)
        find_package(OpenGL REQUIRED)

        if(WIN32)
            find_library(cis565_glfw_library NAMES glfw3
                PATHS
                    "${CMAKE_CURRENT_SOURCE_DIR}/external/lib"
                    "${CMAKE_CURRENT_SOURCE_DIR}/external/lib/vc2015"
                NO_DEFAULT_PATH)
            find_library(cis565_glew_library NAMES glew32s
                PATHS
                    "${CMAKE_CURRENT_SOURCE_DIR}/external/lib"
                    "${CMAKE_CURRENT_SOURCE_DIR}/external/lib/vc2015"
                NO_DEFAULT_PATH)
            if(NOT cis565_glfw_library OR NOT cis565_glew_library)
                message(FATAL_ERROR "Starter GLFW/GLEW libraries were not found")
            endif()
            set(cis565_preview_gl_libraries
                ${cis565_glew_library} ${cis565_glfw_library} ${OPENGL_LIBRARIES}
                user32 gdi32 shell32)
            set(cis565_preview_gl_includes "${CMAKE_CURRENT_SOURCE_DIR}/external/include")
            set(cis565_preview_definitions GLEW_STATIC)
        else()
            find_package(glfw3 REQUIRED)
            find_package(GLEW REQUIRED)
            set(cis565_preview_gl_libraries
                glfw ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
            set(cis565_preview_gl_includes ${GLEW_INCLUDE_DIRS})
        endif()

        set(cis565_imgui_sources
            src/ImGui/imgui.cpp
            src/ImGui/imgui_demo.cpp
            src/ImGui/imgui_draw.cpp
            src/ImGui/imgui_impl_glfw.cpp
            src/ImGui/imgui_impl_opengl3.cpp
            src/ImGui/imgui_tables.cpp
            src/ImGui/imgui_widgets.cpp)

        add_executable(cis565_path_tracer
            src/main.cpp
            src/glslUtility.cpp
            ${cis565_imgui_sources})
        target_include_directories(cis565_path_tracer SYSTEM PRIVATE
            ${cis565_preview_gl_includes})
        target_compile_definitions(cis565_path_tracer PRIVATE
            ${cis565_preview_definitions})
        target_link_libraries(cis565_path_tracer PRIVATE
            cis565_renderer cis565_core CUDA::cudart
            ${cis565_preview_gl_libraries})
        target_compile_features(cis565_path_tracer PRIVATE cxx_std_17)
        set_target_properties(cis565_path_tracer PROPERTIES
            FOLDER "Applications")
        set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            PROPERTY VS_STARTUP_PROJECT cis565_path_tracer)
    endif()
elseif(BUILD_PREVIEW)
    message(STATUS "BUILD_PREVIEW is ignored because BUILD_RENDERER is OFF")
endif()

if(BUILD_TESTING)
    function(cis565_add_host_test target source)
        add_executable(${target} "${source}")
        target_link_libraries(${target} PRIVATE cis565_core)
        target_compile_features(${target} PRIVATE cxx_std_17)
        set_target_properties(${target} PROPERTIES FOLDER "Tests")
        add_test(NAME ${target} COMMAND ${target})
    endfunction()

    cis565_add_host_test(geometry_tests tests/geometry_tests.cpp)
    cis565_add_host_test(checkpoint_tests tests/checkpoint_tests.cpp)

    if(BUILD_RENDERER)
        add_executable(render_tests tests/render_tests.cpp)
        target_link_libraries(render_tests PRIVATE cis565_renderer)
        target_compile_features(render_tests PRIVATE cxx_std_17)
        set_target_properties(render_tests PROPERTIES FOLDER "Tests")
        add_test(NAME render_tests COMMAND render_tests)
    endif()
endif()
