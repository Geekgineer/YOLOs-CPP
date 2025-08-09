cmake_minimum_required(VERSION 3.16)
project(yolo_ort CXX)

# LTO/IPO policy + enable if available
cmake_policy(SET CMP0069 NEW)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_ok OUTPUT ipo_msg)
if(ipo_ok)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# ---- User-supplied ONNX Runtime root ----
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime extracted directory")
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_c_api.h")
  message(FATAL_ERROR "Invalid ONNXRUNTIME_DIR: header not found in ${ONNXRUNTIME_DIR}/include")
endif()

# ---- C++ standard (filesystem needs 17) ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- OpenCV ----
find_package(OpenCV REQUIRED)

# ---- Targets ----
add_executable(image_inference        src/image_inference.cpp)
add_executable(camera_inference       src/camera_inference.cpp)
add_executable(video_inference        src/video_inference.cpp)
add_executable(yolo_performance_analyzer benchmark/yolo_performance_analyzer.cpp)

# Common include dirs (project + ORT + OpenCV)
set(PROJ_INC "${CMAKE_SOURCE_DIR}/include")
foreach(tgt IN ITEMS image_inference camera_inference video_inference yolo_performance_analyzer)
  target_include_directories(${tgt} PRIVATE
    "${PROJ_INC}"
    "${ONNXRUNTIME_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(${tgt} PRIVATE ${OpenCV_LIBS})
endforeach()

# ---- Link ONNX Runtime & platform deps ----
if(WIN32)
  message(STATUS "Building on Windows")
  # Link import lib
  foreach(tgt IN ITEMS image_inference camera_inference video_inference yolo_performance_analyzer)
    target_link_libraries(${tgt} PRIVATE
      "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
      psapi  # for GetProcessMemoryInfo
    )
    # Copy runtime DLLs next to each exe after build
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
              "${ONNXRUNTIME_DIR}/lib"
              "$<TARGET_FILE_DIR:${tgt}>"
    )
  endforeach()
elseif(APPLE)
  message(STATUS "Building on macOS")
  foreach(tgt IN ITEMS image_inference camera_inference video_inference yolo_performance_analyzer)
    target_link_libraries(${tgt} PRIVATE
      "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib"
    )
  endforeach()
  # Ensure the loader can find ORT at runtime
  set(CMAKE_BUILD_RPATH  "${ONNXRUNTIME_DIR}/lib")
  set(CMAKE_INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")
elseif(UNIX)
  message(STATUS "Building on Linux")
  foreach(tgt IN ITEMS image_inference camera_inference video_inference yolo_performance_analyzer)
    target_link_libraries(${tgt} PRIVATE
      "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
      pthread
      dl
    )
  endforeach()
  # Allow running without setting LD_LIBRARY_PATH
  set(CMAKE_BUILD_RPATH  "${ONNXRUNTIME_DIR}/lib")
  set(CMAKE_INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")
else()
  message(FATAL_ERROR "Unsupported platform")
endif()
