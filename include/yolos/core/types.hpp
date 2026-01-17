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
    int x{0};       ///< X-coordinate of top-left corner
    int y{0};       ///< Y-coordinate of top-left corner
    int width{0};   ///< Width of the bounding box
    int height{0};  ///< Height of the bounding box

    BoundingBox() = default;
    
    BoundingBox(int x_, int y_, int width_, int height_)
        : x(x_), y(y_), width(width_), height(height_) {}

    /// @brief Compute area of the bounding box
    [[nodiscard]] float area() const noexcept {
        return static_cast<float>(width * height);
    }

    /// @brief Compute intersection with another bounding box
    [[nodiscard]] BoundingBox intersect(const BoundingBox& other) const noexcept {
        int xStart = std::max(x, other.x);
        int yStart = std::max(y, other.y);
        int xEnd = std::min(x + width, other.x + other.width);
        int yEnd = std::min(y + height, other.y + other.height);
        int iw = std::max(0, xEnd - xStart);
        int ih = std::max(0, yEnd - yStart);
        return BoundingBox(xStart, yStart, iw, ih);
    }

    /// @brief Compute IoU (Intersection over Union) with another bounding box
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
    float x{0.0f};       ///< X-coordinate of center
    float y{0.0f};       ///< Y-coordinate of center
    float width{0.0f};   ///< Width of the box
    float height{0.0f};  ///< Height of the box
    float angle{0.0f};   ///< Rotation angle in radians

    OrientedBoundingBox() = default;

    OrientedBoundingBox(float x_, float y_, float width_, float height_, float angle_)
        : x(x_), y(y_), width(width_), height(height_), angle(angle_) {}

    /// @brief Compute area of the oriented bounding box
    [[nodiscard]] float area() const noexcept {
        return width * height;
    }
};

// ============================================================================
// KeyPoint - Single keypoint for pose estimation
// ============================================================================
struct KeyPoint {
    float x{0.0f};          ///< X-coordinate
    float y{0.0f};          ///< Y-coordinate
    float confidence{0.0f}; ///< Confidence score

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
