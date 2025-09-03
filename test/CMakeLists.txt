cmake_minimum_required(VERSION 3.0.0)
project(yolo_ort_test)

add_compile_definitions("BASE_PATH=${CMAKE_SOURCE_DIR}/")

message(STATUS "BASE_PATH: ${CMAKE_SOURCE_DIR}")

option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenCV REQUIRED)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

include_directories("../include/")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Add executable for image inference
add_executable(test_inference_cpp
               test_inference_cpp.cpp)

# Set include directories for all executables
target_include_directories(test_inference_cpp PRIVATE "${ONNXRUNTIME_DIR}/include")

# Link libraries for all executables
target_link_libraries(test_inference_cpp ${OpenCV_LIBS} nlohmann_json::nlohmann_json)


if(UNIX)
    message(STATUS "We are building on Linux!")
    # Specific Linux build commands or flags
    target_link_libraries(test_inference_cpp "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif(UNIX)

if(APPLE)
    message(STATUS "We are building on macOS!")
    # Specific macOS build commands or flags
    target_link_libraries(test_inference_cpp "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
endif(APPLE)

if(WIN32)
    message(STATUS "We are building on Windows!")
    # Specific Windows build commands or flags
    target_link_libraries(test_inference_cpp "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif(WIN32)
