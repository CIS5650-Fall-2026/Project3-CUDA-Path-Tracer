# Conservative compatibility check: rendering-source or compiler/flag changes
# invalidate saved samples. No timestamps, paths to scene files, or sample target.
set(checkpoint_identity_files
    CMakeLists.txt cmake/CheckpointBuildIdentity.cmake cmake/checkpoint_build_id.h.in
    src/main.cpp src/scene.cpp src/scene.h src/sceneStructs.h
    src/interactions.cu src/interactions.h src/intersections.cu src/intersections.h
    src/pathtrace.cu src/pathtrace.h src/utilities.cpp src/utilities.h
    src/render_checkpoint.cpp src/render_checkpoint.h
    src/generated/metal_presets.h src/generated/transmission_presets.h)
set(checkpoint_source_material "")
foreach(relative_path IN LISTS checkpoint_identity_files)
    set(absolute_path "${CMAKE_CURRENT_SOURCE_DIR}/${relative_path}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${absolute_path}")
    file(SHA256 "${absolute_path}" file_hash)
    string(APPEND checkpoint_source_material "${relative_path}:${file_hash}\n")
endforeach()
string(SHA256 CHECKPOINT_SOURCE_ID "${checkpoint_source_material}")
set(checkpoint_settings_material
    "${CMAKE_CXX_COMPILER_ID}:${CMAKE_CXX_COMPILER_VERSION}:${CMAKE_CUDA_COMPILER_VERSION}:${CMAKE_SYSTEM_PROCESSOR}:${CMAKE_CUDA_ARCHITECTURES}")
foreach(language CXX CUDA)
    foreach(suffix "" _DEBUG _RELEASE _RELWITHDEBINFO _MINSIZEREL)
        string(APPEND checkpoint_settings_material ":${CMAKE_${language}_FLAGS${suffix}}")
    endforeach()
endforeach()
string(SHA256 CHECKPOINT_SETTINGS_ID "${checkpoint_settings_material}")
configure_file(cmake/checkpoint_build_id.h.in generated/checkpoint_build_id.h @ONLY)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE CHECKPOINT_BUILD_CONFIG="$<CONFIG>")
