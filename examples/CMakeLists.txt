cmake_minimum_required(VERSION 3.10)
project(yolo_ort_examples)

# Find required packages
find_package(OpenCV REQUIRED)
include_directories("../include/")

# ============================================================================
# IMAGE EXAMPLES
# ============================================================================
add_executable(example_image_det example_image_det.cpp)
add_executable(example_image_seg example_image_seg.cpp)
add_executable(example_image_obb example_image_obb.cpp)
add_executable(example_image_class example_image_class.cpp)
add_executable(example_image_pose example_image_pose.cpp)
# ============================================================================
# VIDEO EXAMPLES
# ============================================================================
add_executable(example_video_det example_video_det.cpp)
add_executable(example_video_seg example_video_seg.cpp)
add_executable(example_video_obb example_video_obb.cpp)
add_executable(example_video_class example_video_class.cpp)
add_executable(example_video_pose example_video_pose.cpp)
# ============================================================================
# CAMERA EXAMPLES
# ============================================================================
add_executable(example_camera_det example_camera_det.cpp)
add_executable(example_camera_seg example_camera_seg.cpp)
add_executable(example_camera_obb example_camera_obb.cpp)
add_executable(example_camera_class example_camera_class.cpp)
add_executable(example_camera_pose example_camera_pose.cpp)

# ============================================================================
# ============================================================================
set(ALL_EXAMPLES
    example_image_det example_image_seg example_image_obb example_image_class example_image_pose
    example_video_det example_video_seg example_video_obb example_video_class example_video_pose
    example_camera_det example_camera_seg example_camera_obb example_camera_class example_camera_pose
)

foreach(example ${ALL_EXAMPLES})
    target_include_directories(${example} PRIVATE "${ONNXRUNTIME_DIR}/include")
    target_link_libraries(${example} ${OpenCV_LIBS})
endforeach()

# ============================================================================
# Platform-specific ONNX Runtime linking
# ============================================================================
if(UNIX AND NOT APPLE)
    message(STATUS "Building on Linux")
    foreach(example ${ALL_EXAMPLES})
        target_link_libraries(${example} "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    endforeach()
elseif(APPLE)
    message(STATUS "Building on macOS")
    foreach(example ${ALL_EXAMPLES})
        target_link_libraries(${example} "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    endforeach()
elseif(WIN32)
    message(STATUS "Building on Windows")
    foreach(example ${ALL_EXAMPLES})
        target_link_libraries(${example} "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    endforeach()
endif()



