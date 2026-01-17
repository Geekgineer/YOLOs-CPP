# ============================================================================
# YOLOs-CPP Test Suite
# ============================================================================
cmake_minimum_required(VERSION 3.16)
project(yolos_cpp_tests VERSION 2.0.0 LANGUAGES CXX)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Base Path Definitions
# ============================================================================
add_compile_definitions("BASE_PATH=${CMAKE_SOURCE_DIR}/")
add_compile_definitions("BASE_PATH_DETECTION=${CMAKE_SOURCE_DIR}/detection/")
add_compile_definitions("BASE_PATH_CLASSIFICATION=${CMAKE_SOURCE_DIR}/classification/")
add_compile_definitions("BASE_PATH_SEGMENTATION=${CMAKE_SOURCE_DIR}/segmentation/")
add_compile_definitions("BASE_PATH_POSE=${CMAKE_SOURCE_DIR}/pose/")
add_compile_definitions("BASE_PATH_OBB=${CMAKE_SOURCE_DIR}/obb/")

message(STATUS "Test Base Path: ${CMAKE_SOURCE_DIR}")

# ============================================================================
# ONNX Runtime
# ============================================================================
if(NOT DEFINED ONNXRUNTIME_DIR)
    # Platform-specific search paths
    if(WIN32)
        set(ONNXRUNTIME_SEARCH_PATHS
            "${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-1.20.1"
            "${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-gpu-1.20.1"
            "${CMAKE_SOURCE_DIR}/../onnxruntime-win-x64-1.20.1"
            "${CMAKE_SOURCE_DIR}/../onnxruntime-win-x64-gpu-1.20.1"
        )
    elseif(APPLE)
        set(ONNXRUNTIME_SEARCH_PATHS
            "${CMAKE_SOURCE_DIR}/onnxruntime-osx-arm64-1.20.1"
            "${CMAKE_SOURCE_DIR}/onnxruntime-osx-x86_64-1.20.1"
            "${CMAKE_SOURCE_DIR}/../onnxruntime-osx-arm64-1.20.1"
            "${CMAKE_SOURCE_DIR}/../onnxruntime-osx-x86_64-1.20.1"
        )
    else()
        set(ONNXRUNTIME_SEARCH_PATHS
            "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.20.1"
            "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-gpu-1.20.1"
            "${CMAKE_SOURCE_DIR}/../onnxruntime-linux-x64-1.20.1"
            "${CMAKE_SOURCE_DIR}/../onnxruntime-linux-x64-gpu-1.20.1"
        )
    endif()
    
    foreach(path ${ONNXRUNTIME_SEARCH_PATHS})
        if(EXISTS "${path}/include/onnxruntime_cxx_api.h")
            set(ONNXRUNTIME_DIR "${path}")
            break()
        endif()
    endforeach()
endif()

if(NOT ONNXRUNTIME_DIR OR NOT EXISTS "${ONNXRUNTIME_DIR}/include")
    message(FATAL_ERROR "ONNX Runtime not found. Set -DONNXRUNTIME_DIR=/path/to/onnxruntime")
endif()

message(STATUS "ONNX Runtime: ${ONNXRUNTIME_DIR}")

# ============================================================================
# Dependencies
# ============================================================================
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV: ${OpenCV_VERSION}")

# nlohmann_json
include(FetchContent)
FetchContent_Declare(json 
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

# GoogleTest
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# Threading
find_package(Threads REQUIRED)

# ============================================================================
# Include Directories
# ============================================================================
include_directories(
    "${CMAKE_SOURCE_DIR}/../include"
    "${ONNXRUNTIME_DIR}/include"
)

# ============================================================================
# ONNX Runtime Library
# ============================================================================
if(UNIX AND NOT APPLE)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
elseif(APPLE)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
elseif(WIN32)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif()

# ============================================================================
# Test Task Selection
# ============================================================================
option(testTask "Specify test task: 0=detection, 1=classification, 2=segmentation, 3=pose, 4=obb, 5=all" STRING)
message(STATUS "Selected test task: ${testTask}")

set(detection_path "${CMAKE_SOURCE_DIR}/detection")
set(classification_path "${CMAKE_SOURCE_DIR}/classification")
set(segmentation_path "${CMAKE_SOURCE_DIR}/segmentation")
set(pose_path "${CMAKE_SOURCE_DIR}/pose")
set(obb_path "${CMAKE_SOURCE_DIR}/obb")

set(INFERENCE_EXES)
set(TEST_EXES)

# ============================================================================
# Detection Tests
# ============================================================================
if(testTask STREQUAL "0" OR testTask STREQUAL "5")
    add_executable(inference_detection_cpp 
        ${detection_path}/inference_detection_cpp.cpp)
    add_executable(compare_detection_results 
        ${detection_path}/compare_results.cpp)
    
    list(APPEND INFERENCE_EXES inference_detection_cpp)
    list(APPEND TEST_EXES compare_detection_results)
endif()

# ============================================================================
# Classification Tests
# ============================================================================
if(testTask STREQUAL "1" OR testTask STREQUAL "5")
    add_executable(inference_classification_cpp 
        ${classification_path}/inference_classification_cpp.cpp)
    add_executable(compare_classification_results 
        ${classification_path}/compare_results.cpp)
    
    list(APPEND INFERENCE_EXES inference_classification_cpp)
    list(APPEND TEST_EXES compare_classification_results)
endif()

# ============================================================================
# Segmentation Tests
# ============================================================================
if(testTask STREQUAL "2" OR testTask STREQUAL "5")
    add_executable(inference_segmentation_cpp 
        ${segmentation_path}/inference_segmentation_cpp.cpp)
    add_executable(compare_segmentation_results 
        ${segmentation_path}/compare_results.cpp)

    list(APPEND INFERENCE_EXES inference_segmentation_cpp)
    list(APPEND TEST_EXES compare_segmentation_results)
endif()

# ============================================================================
# Pose Tests
# ============================================================================
if(testTask STREQUAL "3" OR testTask STREQUAL "5")
    add_executable(inference_pose_cpp 
        ${pose_path}/inference_pose_cpp.cpp)
    add_executable(compare_pose_results 
        ${pose_path}/compare_results.cpp)
    
    list(APPEND INFERENCE_EXES inference_pose_cpp)
    list(APPEND TEST_EXES compare_pose_results)
endif()

# ============================================================================
# OBB Tests
# ============================================================================
if(testTask STREQUAL "4" OR testTask STREQUAL "5")
    add_executable(inference_obb_cpp 
        ${obb_path}/inference_obb_cpp.cpp)
    add_executable(compare_obb_results 
        ${obb_path}/compare_obb_results.cpp)
    
    list(APPEND INFERENCE_EXES inference_obb_cpp)
    list(APPEND TEST_EXES compare_obb_results)
endif()

# ============================================================================
# Configure Inference Executables
# ============================================================================
foreach(target ${INFERENCE_EXES})
    message(STATUS "Configuring inference target: ${target}")
    
    target_include_directories(${target} PRIVATE 
        "${CMAKE_SOURCE_DIR}/../include"
        "${ONNXRUNTIME_DIR}/include"
    )
    
    target_link_libraries(${target} PRIVATE
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIB}
        nlohmann_json::nlohmann_json
        Threads::Threads
    )
    
    # Set RPATH
    if(UNIX AND NOT APPLE)
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
            INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
        )
    endif()
endforeach()

# ============================================================================
# Configure Test Executables
# ============================================================================
foreach(target ${TEST_EXES})
    message(STATUS "Adding test target: ${target}")
    
    target_link_libraries(${target} PRIVATE
        nlohmann_json::nlohmann_json
        GTest::gtest_main
        ${OpenCV_LIBS}
    )
    
    gtest_discover_tests(${target})
endforeach()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== Test Suite Configuration ===")
message(STATUS "  Inference executables: ${INFERENCE_EXES}")
message(STATUS "  Test executables: ${TEST_EXES}")
message(STATUS "")
