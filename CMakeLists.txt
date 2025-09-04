cmake_minimum_required(VERSION 3.0.0)
project(yolo_ort)

option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenCV REQUIRED)

include_directories("include/")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Add executable for image inference
add_executable(image_inference
               src/image_inference.cpp)

# Add executable for camera inference
add_executable(camera_inference
               src/camera_inference.cpp)

# Add executable for video inference
add_executable(video_inference
               src/video_inference.cpp)

add_executable(class_image_inference
               src/class_image_inference.cpp)

# Set include directories for all executables
target_include_directories(image_inference PRIVATE "${ONNXRUNTIME_DIR}/include")
target_include_directories(camera_inference PRIVATE "${ONNXRUNTIME_DIR}/include")
target_include_directories(video_inference PRIVATE "${ONNXRUNTIME_DIR}/include")
target_include_directories(class_image_inference PRIVATE "${ONNXRUNTIME_DIR}/include")

# Link libraries for all executables
target_link_libraries(image_inference ${OpenCV_LIBS})
target_link_libraries(camera_inference ${OpenCV_LIBS})
target_link_libraries(video_inference ${OpenCV_LIBS})
target_link_libraries(class_image_inference ${OpenCV_LIBS})

if(UNIX)
    message(STATUS "We are building on Linux!")
    # Specific Linux build commands or flags
    target_link_libraries(image_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    target_link_libraries(camera_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    target_link_libraries(video_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    target_link_libraries(class_image_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif(UNIX)

if(APPLE)
    message(STATUS "We are building on macOS!")
    # Specific macOS build commands or flags
    target_link_libraries(image_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    target_link_libraries(camera_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    target_link_libraries(video_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    target_link_libraries(class_image_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
endif(APPLE)

if(WIN32)
    message(STATUS "We are building on Windows!")
    # Specific Windows build commands or flags
    target_link_libraries(image_inference "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    target_link_libraries(camera_inference "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    target_link_libraries(video_inference "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    target_link_libraries(class_image_inference "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif(WIN32)