# ============================================================================
# YOLO Unified Benchmark - CMake Configuration
# ============================================================================
cmake_minimum_required(VERSION 3.16)
project(yolo_unified_benchmark 
    VERSION 2.0.0
    DESCRIPTION "Comprehensive benchmarking for YOLO models"
    LANGUAGES CXX
)

# ============================================================================
# C++ Standard and Compiler Settings
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Build Type
# ============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================================
# Compiler Warnings and Optimization
# ============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Common warnings
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -Wno-missing-field-initializers
    )
    
    # Release optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG)
        # Enable architecture-specific optimizations if requested
        option(ENABLE_NATIVE_ARCH "Enable native architecture optimizations" ON)
        if(ENABLE_NATIVE_ARCH)
            add_compile_options(-march=native)
            message(STATUS "Native architecture optimizations enabled")
        endif()
        # LTO (Link Time Optimization)
        option(ENABLE_LTO "Enable Link Time Optimization" OFF)
        if(ENABLE_LTO)
            set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
            message(STATUS "LTO enabled")
        endif()
    endif()
    
    # Debug settings
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
        add_compile_definitions(DEBUG)
    endif()
elseif(MSVC)
    add_compile_options(/W4 /permissive-)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2 /GL)
    endif()
endif()

# ============================================================================
# ONNX Runtime Configuration
# ============================================================================
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime extracted directory")
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_c_api.h")
    message(FATAL_ERROR 
        "Invalid ONNXRUNTIME_DIR: header not found in ${ONNXRUNTIME_DIR}/include\n"
        "Please set -DONNXRUNTIME_DIR=/path/to/onnxruntime"
    )
endif()

# ============================================================================
# OpenCV
# ============================================================================
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs videoio)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# ============================================================================
# Benchmark Executable
# ============================================================================
add_executable(yolo_unified_benchmark yolo_unified_benchmark.cpp)

target_include_directories(yolo_unified_benchmark PRIVATE
    "${CMAKE_SOURCE_DIR}/../include"
    "${ONNXRUNTIME_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolo_unified_benchmark PRIVATE ${OpenCV_LIBS})

# ============================================================================
# Platform-Specific Configuration
# ============================================================================
if(WIN32)
    message(STATUS "Platform: Windows")
    target_link_libraries(yolo_unified_benchmark PRIVATE
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
        psapi
    )
    # Copy DLLs to output directory
    add_custom_command(TARGET yolo_unified_benchmark POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
                "$<TARGET_FILE_DIR:yolo_unified_benchmark>"
        COMMENT "Copying ONNX Runtime DLL"
    )
elseif(APPLE)
    message(STATUS "Platform: macOS")
    target_link_libraries(yolo_unified_benchmark PRIVATE
        "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib"
    )
    set_target_properties(yolo_unified_benchmark PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
    )
elseif(UNIX)
    message(STATUS "Platform: Linux")
    target_link_libraries(yolo_unified_benchmark PRIVATE
        "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
        pthread
        dl
    )
    set_target_properties(yolo_unified_benchmark PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
    )
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS yolo_unified_benchmark
    RUNTIME DESTINATION bin
)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== YOLO Unified Benchmark Configuration ===")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV:         ${OpenCV_VERSION}")
message(STATUS "  ONNX Runtime:   ${ONNXRUNTIME_DIR}")
message(STATUS "")
