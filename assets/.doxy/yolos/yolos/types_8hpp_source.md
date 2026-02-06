

# File types.hpp

[**File List**](files.md) **>** [**core**](dir_a763ec46eda5c5a329ecdb0f0bec1eed.md) **>** [**types.hpp**](types_8hpp.md)

[Go to the documentation of this file](types_8hpp.md)


```C++
#pragma once

// ============================================================================
// YOLO Core Types
// ============================================================================
// Single source of truth for shared data structures used across all YOLO tasks.
// 
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <vector>
#include <algorithm>
#include <cmath>

namespace yolos {

// ============================================================================
// BoundingBox - Axis-aligned bounding box for detection, segmentation, pose
// ============================================================================
struct BoundingBox {
    int x{0};       
    int y{0};       
    int width{0};   
    int height{0};  

    BoundingBox() = default;
    
    BoundingBox(int x_, int y_, int width_, int height_)
        : x(x_), y(y_), width(width_), height(height_) {}

    [[nodiscard]] float area() const noexcept {
        return static_cast<float>(width * height);
    }

    [[nodiscard]] BoundingBox intersect(const BoundingBox& other) const noexcept {
        int xStart = std::max(x, other.x);
        int yStart = std::max(y, other.y);
        int xEnd = std::min(x + width, other.x + other.width);
        int yEnd = std::min(y + height, other.y + other.height);
        int iw = std::max(0, xEnd - xStart);
        int ih = std::max(0, yEnd - yStart);
        return BoundingBox(xStart, yStart, iw, ih);
    }

    [[nodiscard]] float iou(const BoundingBox& other) const noexcept {
        BoundingBox inter = intersect(other);
        float interArea = inter.area();
        float unionArea = area() + other.area() - interArea;
        return (unionArea > 0.0f) ? (interArea / unionArea) : 0.0f;
    }
};

// ============================================================================
// OrientedBoundingBox - Rotated bounding box for OBB detection
// ============================================================================
struct OrientedBoundingBox {
    float x{0.0f};       
    float y{0.0f};       
    float width{0.0f};   
    float height{0.0f};  
    float angle{0.0f};   

    OrientedBoundingBox() = default;

    OrientedBoundingBox(float x_, float y_, float width_, float height_, float angle_)
        : x(x_), y(y_), width(width_), height(height_), angle(angle_) {}

    [[nodiscard]] float area() const noexcept {
        return width * height;
    }
};

// ============================================================================
// KeyPoint - Single keypoint for pose estimation
// ============================================================================
struct KeyPoint {
    float x{0.0f};          
    float y{0.0f};          
    float confidence{0.0f}; 

    KeyPoint() = default;

    KeyPoint(float x_, float y_, float conf_ = 0.0f)
        : x(x_), y(y_), confidence(conf_) {}
};

// ============================================================================
// Skeleton connections for COCO pose format (17 keypoints)
// ============================================================================
inline const std::vector<std::pair<int, int>>& getPoseSkeleton() {
    static const std::vector<std::pair<int, int>> POSE_SKELETON = {
        // Face connections
        {0, 1}, {0, 2}, {1, 3}, {2, 4},
        // Head-to-shoulder connections
        {3, 5}, {4, 6},
        // Arms
        {5, 7}, {7, 9}, {6, 8}, {8, 10},
        // Body
        {5, 6}, {5, 11}, {6, 12}, {11, 12},
        // Legs
        {11, 13}, {13, 15}, {12, 14}, {14, 16}
    };
    return POSE_SKELETON;
}

} // namespace yolos
```


