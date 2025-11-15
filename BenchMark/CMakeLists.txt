cmake_minimum_required(VERSION 3.10)
project(YOLO_Evaluator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------
# Find OpenCV
# ----------------------------------------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ----------------------------------------
# Add source directory
# ----------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories("${ONNXRUNTIME_DIR}/include")

# ----------------------------------------
# Executable
# ----------------------------------------
add_executable(evaluate
    src/evaluation_main.cpp
)

# ----------------------------------------
# Link libraries
# ----------------------------------------
target_link_libraries(evaluate
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
)

# ----------------------------------------
# Optimization flags (optional but recommended)
# ----------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Using Release optimizations")
    add_compile_options(-O3 -march=native -ffast-math)
endif()

