#pragma once

// ============================================================================
// YOLO Core Utilities
// ============================================================================
// Common utility functions used across all YOLO tasks.
// All functions are marked inline to prevent ODR violations.
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <type_traits>
#include <cstdint>

namespace yolos {
namespace utils {

// ============================================================================
// Math Utilities
// ============================================================================

/// @brief Clamp a value to a specified range [low, high]
/// @tparam T Arithmetic type (int, float, etc.)
/// @param value The value to clamp
/// @param low Lower bound
/// @param high Upper bound
/// @return Clamped value
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
clamp(const T& value, const T& low, const T& high) {
    // Ensure range is valid; swap if necessary
    T validLow = low < high ? low : high;
    T validHigh = low < high ? high : low;
    
    if (value < validLow) return validLow;
    if (value > validHigh) return validHigh;
    return value;
}

/// @brief Compute the product of elements in a vector
/// @param shape Vector of dimensions
/// @return Product of all elements
inline size_t vectorProduct(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
}

// ============================================================================
// File I/O Utilities
// ============================================================================

/// @brief Load class names from a file (one class name per line)
/// @param path Path to the class names file
/// @return Vector of class names
inline std::vector<std::string> getClassNames(const std::string& path) {
    std::vector<std::string> classNames;
    std::ifstream infile(path);
    
    if (!infile) {
        std::cerr << "[ERROR] Failed to open class names file: " << path << std::endl;
        return classNames;
    }
    
    std::string line;
    while (std::getline(infile, line)) {
        // Remove carriage return if present (Windows compatibility)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            classNames.emplace_back(line);
        }
    }
    
    return classNames;
}

// ============================================================================
// Sigmoid Activation
// ============================================================================

/// @brief Apply sigmoid activation: 1 / (1 + exp(-x))
/// @param x Input value
/// @return Sigmoid of x
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/// @brief Apply sigmoid activation to a vector in-place
/// @param values Vector of values to transform
inline void sigmoidInplace(std::vector<float>& values) {
    for (auto& v : values) {
        v = sigmoid(v);
    }
}

} // namespace utils
} // namespace yolos
