

# File utils.hpp

[**File List**](files.md) **>** [**core**](dir_a763ec46eda5c5a329ecdb0f0bec1eed.md) **>** [**utils.hpp**](utils_8hpp.md)

[Go to the documentation of this file](utils_8hpp.md)


```C++
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

inline size_t vectorProduct(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
}

// ============================================================================
// File I/O Utilities
// ============================================================================

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

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline void sigmoidInplace(std::vector<float>& values) {
    for (auto& v : values) {
        v = sigmoid(v);
    }
}

} // namespace utils
} // namespace yolos
```


