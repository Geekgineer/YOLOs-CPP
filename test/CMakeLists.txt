cmake_minimum_required(VERSION 3.0.0)
project(yolo_ort_test)

add_compile_definitions("BASE_PATH=${CMAKE_SOURCE_DIR}/")
add_compile_definitions("BASE_PATH_DETECTION=${CMAKE_SOURCE_DIR}/detection/")
add_compile_definitions("BASE_PATH_CLASSIFICATION=${CMAKE_SOURCE_DIR}/classification/")
add_compile_definitions("BASE_PATH_SEGMENTATION=${CMAKE_SOURCE_DIR}/segmentation/")
add_compile_definitions("BASE_PATH_POSE=${CMAKE_SOURCE_DIR}/pose/")
add_compile_definitions("BASE_PATH_OBB=${CMAKE_SOURCE_DIR}/obb/")

message(STATUS "BASE_PATH: ${CMAKE_SOURCE_DIR}")

option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenCV REQUIRED)


# nlohmann_json
include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

# GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)


include_directories("../include/")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(testTask "Specify which test task to build: 0 for detection, 1 for classification, 2 for segmentation, 3 for pose, 4 for obb, 5 for all." STRING)
message(STATUS "Selected test task: ${testTask}")

set(detection_path "${CMAKE_SOURCE_DIR}/detection")
set(classification_path "${CMAKE_SOURCE_DIR}/classification")
set(segmentation_path "${CMAKE_SOURCE_DIR}/segmentation")
set(pose_path "${CMAKE_SOURCE_DIR}/pose")
set(obb_path "${CMAKE_SOURCE_DIR}/obb")

set(EXES)
set(TESTS)


# Add executables
if (testTask STREQUAL "0" OR testTask STREQUAL "5")
    add_executable(inference_detection_cpp 
                   ${detection_path}/inference_detection_cpp.cpp)
    add_executable(compare_detection_results 
                   ${detection_path}/compare_results.cpp)
    
    list(APPEND EXES inference_detection_cpp)
    list(APPEND TESTS compare_detection_results)
endif()

if (testTask STREQUAL "1" OR testTask STREQUAL "5")
    add_executable(inference_classification_cpp 
                    ${classification_path}/inference_classification_cpp.cpp)
    add_executable(compare_classification_results 
                   ${classification_path}/compare_results.cpp)
    
    list(APPEND EXES inference_classification_cpp)
    list(APPEND TESTS compare_classification_results)
endif()

if (testTask STREQUAL "2" OR testTask STREQUAL "5")
    add_executable(inference_segmentation_cpp 
                   ${segmentation_path}/inference_segmentation_cpp.cpp)
    
    add_executable(compare_segmentation_results 
                   ${segmentation_path}/compare_results.cpp)

    list(APPEND EXES inference_segmentation_cpp)
    list(APPEND TESTS compare_segmentation_results)
endif()

if (testTask STREQUAL "3" OR testTask STREQUAL "5")
    add_executable(inference_pose_cpp 
                   ${pose_path}/inference_pose_cpp.cpp)
    
    add_executable(compare_pose_results 
                   ${pose_path}/compare_results.cpp)
    
    list(APPEND EXES inference_pose_cpp)
    list(APPEND TESTS compare_pose_results)
endif()

if (testTask STREQUAL "4" OR testTask STREQUAL "5")
    add_executable(inference_obb_cpp 
                   ${obb_path}/inference_obb_cpp.cpp)
    
    add_executable(compare_obb_results 
                   ${obb_path}/compare_obb_results.cpp)
    
    list(APPEND EXES inference_obb_cpp)
    list(APPEND TESTS compare_obb_results)
endif()


# Set include directories for all executables
foreach(exec_target ${EXES})

    message(STATUS "Configuring target: ${exec_target}")

    target_include_directories(${exec_target} PRIVATE "${ONNXRUNTIME_DIR}/include")

    # Link libraries for all executables
    target_link_libraries(${exec_target} 
    ${OpenCV_LIBS} 
    nlohmann_json::nlohmann_json
    )

    target_include_directories(${exec_target} PRIVATE ${CMAKE_SOURCE_DIR}/../include/det)


    if(UNIX)
        message(STATUS "We are building on Linux!")
        # Specific Linux build commands or flags
        target_link_libraries(${exec_target} "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    endif(UNIX)

    if(APPLE)
        message(STATUS "We are building on macOS!")
        # Specific macOS build commands or flags
        target_link_libraries(${exec_target} "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    endif(APPLE)

    if(WIN32)
        message(STATUS "We are building on Windows!")
        # Specific Windows build commands or flags
        target_link_libraries(${exec_target} "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    endif(WIN32)

endforeach()

# Add tests
foreach(test_target ${TESTS})
    message(STATUS "Adding test: ${test_target}")

    # Link libraries for all executables
    target_link_libraries(${test_target} 
        nlohmann_json::nlohmann_json
        GTest::gtest_main
    )

    gtest_discover_tests(${test_target})
endforeach()
