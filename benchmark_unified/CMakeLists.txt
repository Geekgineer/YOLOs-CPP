cmake_minimum_required(VERSION 3.16)
project(yolo_unified_benchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- User-supplied ONNX Runtime root ----
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime extracted directory")
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_c_api.h")
  message(FATAL_ERROR "Invalid ONNXRUNTIME_DIR: header not found in ${ONNXRUNTIME_DIR}/include")
endif()

# ---- OpenCV ----
find_package(OpenCV REQUIRED)

# ---- Target ----
add_executable(yolo_unified_benchmark yolo_unified_benchmark.cpp)

# Include dirs
target_include_directories(yolo_unified_benchmark PRIVATE
  "${CMAKE_SOURCE_DIR}/../include"
  "${ONNXRUNTIME_DIR}/include"
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolo_unified_benchmark PRIVATE ${OpenCV_LIBS})

# ---- Link ONNX Runtime & platform deps ----
if(WIN32)
  message(STATUS "Building on Windows")
  target_link_libraries(yolo_unified_benchmark PRIVATE
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
    psapi  # for GetProcessMemoryInfo
  )
  # Copy runtime DLLs next to each exe after build
  add_custom_command(TARGET yolo_unified_benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${ONNXRUNTIME_DIR}/lib"
            "$<TARGET_FILE_DIR:yolo_unified_benchmark>"
  )
elseif(APPLE)
  message(STATUS "Building on macOS")
  target_link_libraries(yolo_unified_benchmark PRIVATE
    "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib"
  )
  # Ensure the loader can find ORT at runtime
  set(CMAKE_BUILD_RPATH  "${ONNXRUNTIME_DIR}/lib")
  set(CMAKE_INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")
elseif(UNIX)
  message(STATUS "Building on Linux")
  target_link_libraries(yolo_unified_benchmark PRIVATE
    "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
    pthread
    dl
  )
  # Allow running without setting LD_LIBRARY_PATH
  set(CMAKE_BUILD_RPATH  "${ONNXRUNTIME_DIR}/lib")
  set(CMAKE_INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")
else()
  message(FATAL_ERROR "Unsupported platform")
endif()

