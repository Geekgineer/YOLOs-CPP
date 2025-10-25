cmake_minimum_required(VERSION 3.0.0)
project(yolo_ort)

<<<<<<< HEAD
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

=======
>>>>>>> 3a6f35477e72dcabb6e5a27c1d2bb2cd06b64750
option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenCV REQUIRED)

include_directories("include/")

<<<<<<< HEAD
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
=======
>>>>>>> 3a6f35477e72dcabb6e5a27c1d2bb2cd06b64750
# Add executable for image inference
add_executable(image_inference
               src/image_inference.cpp)

# Add executable for camera inference
add_executable(camera_inference
               src/camera_inference.cpp)

# Add executable for video inference
add_executable(video_inference
               src/video_inference.cpp)

<<<<<<< HEAD
add_executable(class_image_inference
               src/class_image_inference.cpp)

# Add executable for batch image inference
add_executable(batch_image_inference
               src/batch_image_inference.cpp)
=======
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
>>>>>>> 3a6f35477e72dcabb6e5a27c1d2bb2cd06b64750

# Set include directories for all executables
target_include_directories(image_inference PRIVATE "${ONNXRUNTIME_DIR}/include")
target_include_directories(camera_inference PRIVATE "${ONNXRUNTIME_DIR}/include")
target_include_directories(video_inference PRIVATE "${ONNXRUNTIME_DIR}/include")
<<<<<<< HEAD
target_include_directories(class_image_inference PRIVATE "${ONNXRUNTIME_DIR}/include")
target_include_directories(batch_image_inference PRIVATE "${ONNXRUNTIME_DIR}/include")
=======

# Set compile features for all executables
target_compile_features(image_inference PRIVATE cxx_std_14)
target_compile_features(camera_inference PRIVATE cxx_std_14)
target_compile_features(video_inference PRIVATE cxx_std_14)
>>>>>>> 3a6f35477e72dcabb6e5a27c1d2bb2cd06b64750

# Link libraries for all executables
target_link_libraries(image_inference ${OpenCV_LIBS})
target_link_libraries(camera_inference ${OpenCV_LIBS})
target_link_libraries(video_inference ${OpenCV_LIBS})
<<<<<<< HEAD
target_link_libraries(class_image_inference ${OpenCV_LIBS})
target_link_libraries(batch_image_inference ${OpenCV_LIBS})
=======
>>>>>>> 3a6f35477e72dcabb6e5a27c1d2bb2cd06b64750

if(UNIX)
    message(STATUS "We are building on Linux!")
    # Specific Linux build commands or flags
    target_link_libraries(image_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    target_link_libraries(camera_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    target_link_libraries(video_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
<<<<<<< HEAD
    target_link_libraries(class_image_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    target_link_libraries(batch_image_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
=======
>>>>>>> 3a6f35477e72dcabb6e5a27c1d2bb2cd06b64750
endif(UNIX)

if(APPLE)
    message(STATUS "We are building on macOS!")
    # Specific macOS build commands or flags
    target_link_libraries(image_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    target_link_libraries(camera_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    target_link_libraries(video_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
<<<<<<< HEAD
    target_link_libraries(class_image_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    target_link_libraries(batch_image_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
=======
>>>>>>> 3a6f35477e72dcabb6e5a27c1d2bb2cd06b64750
endif(APPLE)

if(WIN32)
    message(STATUS "We are building on Windows!")
    # Specific Windows build commands or flags
    target_link_libraries(image_inference "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    target_link_libraries(camera_inference "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    target_link_libraries(video_inference "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
<<<<<<< HEAD
    target_link_libraries(class_image_inference "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    target_link_libraries(batch_image_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.lib")
endif(WIN32)
=======
endif(WIN32)
>>>>>>> 3a6f35477e72dcabb6e5a27c1d2bb2cd06b64750
