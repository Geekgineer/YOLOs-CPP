

# File version.hpp

[**File List**](files.md) **>** [**core**](dir_a763ec46eda5c5a329ecdb0f0bec1eed.md) **>** [**version.hpp**](version_8hpp.md)

[Go to the documentation of this file](version_8hpp.md)


```C++
#pragma once

// ============================================================================
// YOLO Version Detection
// ============================================================================
// Defines YOLO model version enum and utilities for runtime version detection
// based on output tensor shapes.
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <vector>
#include <cstdint>
#include <string>

namespace yolos {

// ============================================================================
// YOLOVersion Enum
// ============================================================================
enum class YOLOVersion {
    Auto,   
    V7,     
    V8,     
    V10,    
    V11,    
    V12,    
    V26,    
    NAS     
};

// ============================================================================
// Version Detection Utilities
// ============================================================================
namespace version {

inline std::string toString(YOLOVersion version) {
    switch (version) {
        case YOLOVersion::Auto: return "Auto";
        case YOLOVersion::V7:   return "YOLOv7";
        case YOLOVersion::V8:   return "YOLOv8";
        case YOLOVersion::V10:  return "YOLOv10";
        case YOLOVersion::V11:  return "YOLOv11";
        case YOLOVersion::V12:  return "YOLOv12";
        case YOLOVersion::V26:  return "YOLOv26";
        case YOLOVersion::NAS:  return "YOLO-NAS";
        default: return "Unknown";
    }
}

inline YOLOVersion detectFromOutputShape(const std::vector<int64_t>& outputShape, size_t numOutputs = 1) {
    // YOLO-NAS has 2 outputs: boxes and scores
    if (numOutputs == 2) {
        return YOLOVersion::NAS;
    }
    
    // Must have at least 3 dimensions for detection models
    if (outputShape.size() < 3) {
        return YOLOVersion::V11; // Default fallback
    }
    
    const int64_t dim1 = outputShape[1];
    const int64_t dim2 = outputShape[2];
    
    // YOLOv10: [batch, num_boxes, 6] - end-to-end format
    if (dim2 == 6) {
        return YOLOVersion::V10;
    }
    
    // YOLO-NAS single output: [batch, num_boxes, 4] for boxes only
    if (dim2 == 4 && numOutputs == 1) {
        return YOLOVersion::NAS;
    }
    
    // YOLOv7: [batch, num_boxes, num_features] where num_boxes > num_features
    // V7 uses format [batch, 25200, 85] typically
    if (dim1 > dim2) {
        return YOLOVersion::V7;
    }
    
    // YOLOv8/v11: [batch, num_features, num_boxes] where num_features < num_boxes
    // Typical: [1, 84, 8400] for 80 classes
    return YOLOVersion::V11;
}

inline YOLOVersion detectClassificationVersion(const std::vector<int64_t>& outputShape) {
    // Classification models typically have [batch, num_classes] output
    // V11 and V12 have same format, default to V11
    return YOLOVersion::V11;
}

inline bool requiresNMS(YOLOVersion version) {
    // YOLOv10 and YOLOv26 have end-to-end NMS built into the model
    return version != YOLOVersion::V10 && version != YOLOVersion::V26;
}

} // namespace version

} // namespace yolos
```


