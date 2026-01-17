# ============================================================================
# YOLOs-CPP - High-Performance C++ Inference for YOLO Models
# ============================================================================
cmake_minimum_required(VERSION 3.16)
project(YOLOs-CPP VERSION 2.0.0 LANGUAGES CXX)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Build Options
# ============================================================================
option(BUILD_EXAMPLES "Build example applications" OFF)
option(ENABLE_WARNINGS "Enable compiler warnings" ON)

# ============================================================================
# ONNX Runtime
# ============================================================================
if(NOT DEFINED ONNXRUNTIME_DIR)
    # Try to find ONNX Runtime in common locations (platform-specific)
    if(WIN32)
        set(ONNXRUNTIME_SEARCH_PATHS
            "${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-1.20.1"
            "${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-gpu-1.20.1"
            "${CMAKE_SOURCE_DIR}/onnxruntime"
            "$ENV{ONNXRUNTIME_DIR}"
            "$ENV{PROGRAMFILES}/onnxruntime"
        )
    elseif(APPLE)
        set(ONNXRUNTIME_SEARCH_PATHS
            "${CMAKE_SOURCE_DIR}/onnxruntime-osx-arm64-1.20.1"
            "${CMAKE_SOURCE_DIR}/onnxruntime-osx-x86_64-1.20.1"
            "${CMAKE_SOURCE_DIR}/onnxruntime"
            "/opt/onnxruntime"
            "$ENV{ONNXRUNTIME_DIR}"
        )
    else()
        set(ONNXRUNTIME_SEARCH_PATHS
            "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.20.1"
            "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-gpu-1.20.1"
            "${CMAKE_SOURCE_DIR}/onnxruntime"
            "/opt/onnxruntime"
            "$ENV{ONNXRUNTIME_DIR}"
        )
    endif()
    
    foreach(path ${ONNXRUNTIME_SEARCH_PATHS})
        if(EXISTS "${path}/include/onnxruntime_cxx_api.h")
            set(ONNXRUNTIME_DIR "${path}")
            break()
        endif()
    endforeach()
endif()

# Convert relative path to absolute (relative paths are resolved from CMAKE_CURRENT_BINARY_DIR)
if(ONNXRUNTIME_DIR AND NOT IS_ABSOLUTE "${ONNXRUNTIME_DIR}")
    get_filename_component(ONNXRUNTIME_DIR "${ONNXRUNTIME_DIR}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()

if(NOT ONNXRUNTIME_DIR OR NOT EXISTS "${ONNXRUNTIME_DIR}/include")
    message(FATAL_ERROR 
        "ONNX Runtime not found!\n"
        "Please set ONNXRUNTIME_DIR to your ONNX Runtime installation:\n"
        "  cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime\n"
        "Or download it from: https://github.com/microsoft/onnxruntime/releases"
    )
endif()

message(STATUS "ONNX Runtime: ${ONNXRUNTIME_DIR}")

# ============================================================================
# OpenCV
# ============================================================================
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV: ${OpenCV_VERSION}")

# ============================================================================
# Include Directories
# ============================================================================
include_directories(
    "${CMAKE_SOURCE_DIR}/include"
    "${ONNXRUNTIME_DIR}/include"
)

# ============================================================================
# Compiler Flags
# ============================================================================
if(ENABLE_WARNINGS)
    if(MSVC)
        add_compile_options(/W4)
    else()
        add_compile_options(-Wall -Wextra -Wpedantic)
    endif()
endif()

# Release optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(NOT MSVC)
        add_compile_options(-O3)
    endif()
endif()

# Threading support
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ============================================================================
# ONNX Runtime Library
# ============================================================================
if(UNIX AND NOT APPLE)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    message(STATUS "Platform: Linux")
elseif(APPLE)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    message(STATUS "Platform: macOS")
elseif(WIN32)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    message(STATUS "Platform: Windows")
endif()

# ============================================================================
# Helper Function
# ============================================================================
function(add_yolo_executable name source)
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE "${ONNXRUNTIME_DIR}/include")
    target_link_libraries(${name} PRIVATE 
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIB}
        Threads::Threads
    )
    
    # Platform-specific runtime library handling
    if(WIN32)
        # Copy ONNX Runtime DLL to output directory
        set(ONNXRUNTIME_DLL "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")
        if(EXISTS "${ONNXRUNTIME_DLL}")
            add_custom_command(TARGET ${name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_DLL}"
                "$<TARGET_FILE_DIR:${name}>"
            )
        endif()
    elseif(UNIX AND NOT APPLE)
        # Set RPATH for runtime library discovery
        set_target_properties(${name} PROPERTIES
            BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
            INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
        )
    elseif(APPLE)
        # Set RPATH for macOS
        set_target_properties(${name} PROPERTIES
            BUILD_RPATH "@executable_path;${ONNXRUNTIME_DIR}/lib"
            INSTALL_RPATH "@executable_path;${ONNXRUNTIME_DIR}/lib"
        )
    endif()
endfunction()

# ============================================================================
# Main Executables
# ============================================================================
add_yolo_executable(image_inference src/image_inference.cpp)
add_yolo_executable(video_inference src/video_inference.cpp)
add_yolo_executable(camera_inference src/camera_inference.cpp)
add_yolo_executable(class_image_inference src/class_image_inference.cpp)
add_yolo_executable(batch_image_inference src/batch_image_inference.cpp)

# ============================================================================
# Examples (optional)
# ============================================================================
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== YOLOs-CPP Configuration ===")
message(STATUS "  Version:       ${PROJECT_VERSION}")
message(STATUS "  Build Type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV:        ${OpenCV_VERSION}")
message(STATUS "  ONNX Runtime:  ${ONNXRUNTIME_DIR}")
message(STATUS "  Examples:      ${BUILD_EXAMPLES}")
message(STATUS "")
