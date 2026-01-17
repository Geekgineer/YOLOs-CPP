#pragma once

// ============================================================================
// YOLO Non-Maximum Suppression
// ============================================================================
// NMS implementations for axis-aligned and oriented bounding boxes.
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>

#include <opencv2/opencv.hpp>

#include "yolos/core/types.hpp"

namespace yolos {
namespace nms {

// ============================================================================
// Standard NMS for Axis-Aligned Bounding Boxes
// ============================================================================

/// @brief Perform Non-Maximum Suppression on bounding boxes
/// @param boxes Vector of bounding boxes
/// @param scores Vector of confidence scores
/// @param scoreThreshold Minimum score to consider
/// @param nmsThreshold IoU threshold for suppression
/// @param[out] indices Output indices of boxes that survived NMS
inline void NMSBoxes(const std::vector<BoundingBox>& boxes,
                     const std::vector<float>& scores,
                     float scoreThreshold,
                     float nmsThreshold,
                     std::vector<int>& indices) {
    indices.clear();

    const size_t numBoxes = boxes.size();
    if (numBoxes == 0) {
        return;
    }

    // Step 1: Filter boxes by score threshold and create sorted indices
    std::vector<int> sortedIndices;
    sortedIndices.reserve(numBoxes);
    for (size_t i = 0; i < numBoxes; ++i) {
        if (scores[i] >= scoreThreshold) {
            sortedIndices.push_back(static_cast<int>(i));
        }
    }

    if (sortedIndices.empty()) {
        return;
    }

    // Sort by score in descending order
    std::sort(sortedIndices.begin(), sortedIndices.end(),
              [&scores](int a, int b) { return scores[a] > scores[b]; });

    // Step 2: Precompute areas
    std::vector<float> areas(numBoxes, 0.0f);
    for (size_t i = 0; i < numBoxes; ++i) {
        areas[i] = static_cast<float>(boxes[i].width * boxes[i].height);
    }

    // Step 3: Suppression
    std::vector<bool> suppressed(numBoxes, false);

    for (size_t i = 0; i < sortedIndices.size(); ++i) {
        int currentIdx = sortedIndices[i];
        if (suppressed[currentIdx]) {
            continue;
        }

        indices.push_back(currentIdx);

        const BoundingBox& currentBox = boxes[currentIdx];
        const float x1_max = static_cast<float>(currentBox.x);
        const float y1_max = static_cast<float>(currentBox.y);
        const float x2_max = static_cast<float>(currentBox.x + currentBox.width);
        const float y2_max = static_cast<float>(currentBox.y + currentBox.height);
        const float areaCurrent = areas[currentIdx];

        for (size_t j = i + 1; j < sortedIndices.size(); ++j) {
            int compareIdx = sortedIndices[j];
            if (suppressed[compareIdx]) {
                continue;
            }

            const BoundingBox& compareBox = boxes[compareIdx];
            const float x1 = std::max(x1_max, static_cast<float>(compareBox.x));
            const float y1 = std::max(y1_max, static_cast<float>(compareBox.y));
            const float x2 = std::min(x2_max, static_cast<float>(compareBox.x + compareBox.width));
            const float y2 = std::min(y2_max, static_cast<float>(compareBox.y + compareBox.height));

            const float interWidth = x2 - x1;
            const float interHeight = y2 - y1;

            if (interWidth <= 0 || interHeight <= 0) {
                continue;
            }

            const float intersection = interWidth * interHeight;
            const float unionArea = areaCurrent + areas[compareIdx] - intersection;
            const float iou = (unionArea > 0.0f) ? (intersection / unionArea) : 0.0f;

            if (iou > nmsThreshold) {
                suppressed[compareIdx] = true;
            }
        }
    }
}

// ============================================================================
// Float-Precision NMS for Letterbox Coordinates
// ============================================================================

/// @brief Perform NMS on float-precision bounding boxes (for letterbox space)
/// @param boxes Vector of cv::Rect2f boxes
/// @param scores Vector of confidence scores
/// @param scoreThreshold Minimum score to consider
/// @param nmsThreshold IoU threshold for suppression
/// @param[out] indices Output indices of boxes that survived NMS
inline void NMSBoxesF(const std::vector<cv::Rect2f>& boxes,
                      const std::vector<float>& scores,
                      float scoreThreshold,
                      float nmsThreshold,
                      std::vector<int>& indices) {
    indices.clear();

    const size_t numBoxes = boxes.size();
    if (numBoxes == 0) {
        return;
    }

    // Step 1: Filter boxes by score threshold and create sorted indices
    std::vector<int> sortedIndices;
    sortedIndices.reserve(numBoxes);
    for (size_t i = 0; i < numBoxes; ++i) {
        if (scores[i] >= scoreThreshold) {
            sortedIndices.push_back(static_cast<int>(i));
        }
    }

    if (sortedIndices.empty()) {
        return;
    }

    // Sort by score in descending order
    std::sort(sortedIndices.begin(), sortedIndices.end(),
              [&scores](int a, int b) { return scores[a] > scores[b]; });

    // Step 2: Precompute areas
    std::vector<float> areas(numBoxes, 0.0f);
    for (size_t i = 0; i < numBoxes; ++i) {
        areas[i] = boxes[i].width * boxes[i].height;
    }

    // Step 3: Suppression
    std::vector<bool> suppressed(numBoxes, false);

    for (size_t i = 0; i < sortedIndices.size(); ++i) {
        int currentIdx = sortedIndices[i];
        if (suppressed[currentIdx]) {
            continue;
        }

        indices.push_back(currentIdx);

        const cv::Rect2f& currentBox = boxes[currentIdx];
        const float x1_max = currentBox.x;
        const float y1_max = currentBox.y;
        const float x2_max = currentBox.x + currentBox.width;
        const float y2_max = currentBox.y + currentBox.height;
        const float areaCurrent = areas[currentIdx];

        for (size_t j = i + 1; j < sortedIndices.size(); ++j) {
            int compareIdx = sortedIndices[j];
            if (suppressed[compareIdx]) {
                continue;
            }

            const cv::Rect2f& compareBox = boxes[compareIdx];
            const float x1 = std::max(x1_max, compareBox.x);
            const float y1 = std::max(y1_max, compareBox.y);
            const float x2 = std::min(x2_max, compareBox.x + compareBox.width);
            const float y2 = std::min(y2_max, compareBox.y + compareBox.height);

            const float interWidth = x2 - x1;
            const float interHeight = y2 - y1;

            if (interWidth <= 0 || interHeight <= 0) {
                continue;
            }

            const float intersection = interWidth * interHeight;
            const float unionArea = areaCurrent + areas[compareIdx] - intersection;
            const float iou = (unionArea > 0.0f) ? (intersection / unionArea) : 0.0f;

            if (iou > nmsThreshold) {
                suppressed[compareIdx] = true;
            }
        }
    }
}

/// @brief Perform class-aware NMS on float-precision boxes
inline void NMSBoxesFBatched(const std::vector<cv::Rect2f>& boxes,
                             const std::vector<float>& scores,
                             const std::vector<int>& classIds,
                             float scoreThreshold,
                             float nmsThreshold,
                             std::vector<int>& indices) {
    // Create offset boxes to separate classes
    std::vector<cv::Rect2f> offsetBoxes = boxes;
    const float offset = 7680.0f; // Large offset to prevent cross-class overlap

    for (size_t i = 0; i < offsetBoxes.size(); ++i) {
        offsetBoxes[i].x += classIds[i] * offset;
        offsetBoxes[i].y += classIds[i] * offset;
    }

    // Apply standard NMS on offset boxes
    NMSBoxesF(offsetBoxes, scores, scoreThreshold, nmsThreshold, indices);
}

// ============================================================================
// Rotated NMS for Oriented Bounding Boxes
// ============================================================================

/// @brief Compute IoU between two oriented bounding boxes using OpenCV
/// @param box1 First oriented bounding box
/// @param box2 Second oriented bounding box
/// @return IoU value between 0 and 1
inline float computeRotatedIoU(const OrientedBoundingBox& box1, const OrientedBoundingBox& box2) {
    // Convert to OpenCV RotatedRect (angle in degrees)
    cv::RotatedRect rect1(
        cv::Point2f(box1.x, box1.y),
        cv::Size2f(box1.width, box1.height),
        box1.angle * 180.0f / static_cast<float>(CV_PI)
    );

    cv::RotatedRect rect2(
        cv::Point2f(box2.x, box2.y),
        cv::Size2f(box2.width, box2.height),
        box2.angle * 180.0f / static_cast<float>(CV_PI)
    );

    // Compute intersection
    std::vector<cv::Point2f> intersectionPoints;
    int result = cv::rotatedRectangleIntersection(rect1, rect2, intersectionPoints);

    if (result == cv::INTERSECT_NONE) {
        return 0.0f;
    }

    float intersectionArea = 0.0f;
    if (intersectionPoints.size() > 2) {
        intersectionArea = static_cast<float>(cv::contourArea(intersectionPoints));
    }

    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    float unionArea = area1 + area2 - intersectionArea;

    if (unionArea < 1e-7f) {
        return 0.0f;
    }

    return intersectionArea / unionArea;
}

/// @brief Perform NMS on oriented bounding boxes using rotated IoU
/// @param boxes Vector of oriented bounding boxes
/// @param scores Vector of confidence scores
/// @param nmsThreshold IoU threshold for suppression
/// @param maxDet Maximum number of detections to keep
/// @return Indices of boxes that survived NMS
inline std::vector<int> NMSRotated(const std::vector<OrientedBoundingBox>& boxes,
                                   const std::vector<float>& scores,
                                   float nmsThreshold = 0.45f,
                                   int maxDet = 300) {
    if (boxes.empty()) {
        return {};
    }

    // Create indices sorted by score (descending)
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&scores](int a, int b) { return scores[a] > scores[b]; });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<int> keep;

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];

        if (suppressed[idx]) {
            continue;
        }

        keep.push_back(idx);

        if (static_cast<int>(keep.size()) >= maxDet) {
            break;
        }

        // Suppress boxes with high IoU
        for (size_t j = i + 1; j < indices.size(); ++j) {
            int idx2 = indices[j];

            if (suppressed[idx2]) {
                continue;
            }

            float iou = computeRotatedIoU(boxes[idx], boxes[idx2]);

            if (iou >= nmsThreshold) {
                suppressed[idx2] = true;
            }
        }
    }

    return keep;
}

// ============================================================================
// Batched NMS (per-class NMS)
// ============================================================================

/// @brief Perform class-aware NMS by offsetting boxes by class ID
/// @param boxes Vector of bounding boxes
/// @param scores Vector of confidence scores
/// @param classIds Vector of class IDs
/// @param scoreThreshold Minimum score to consider
/// @param nmsThreshold IoU threshold for suppression
/// @param[out] indices Output indices of boxes that survived NMS
inline void NMSBoxesBatched(const std::vector<BoundingBox>& boxes,
                            const std::vector<float>& scores,
                            const std::vector<int>& classIds,
                            float scoreThreshold,
                            float nmsThreshold,
                            std::vector<int>& indices) {
    // Create offset boxes to separate classes
    std::vector<BoundingBox> offsetBoxes = boxes;
    const int offset = 7680; // Large offset to prevent cross-class overlap

    for (size_t i = 0; i < offsetBoxes.size(); ++i) {
        offsetBoxes[i].x += classIds[i] * offset;
        offsetBoxes[i].y += classIds[i] * offset;
    }

    // Apply standard NMS on offset boxes
    NMSBoxes(offsetBoxes, scores, scoreThreshold, nmsThreshold, indices);
}

/// @brief Perform class-aware NMS on oriented bounding boxes
/// @param boxes Vector of oriented bounding boxes
/// @param scores Vector of confidence scores
/// @param classIds Vector of class IDs
/// @param nmsThreshold IoU threshold for suppression
/// @param maxDet Maximum number of detections to keep
/// @return Indices of boxes that survived NMS
inline std::vector<int> NMSRotatedBatched(const std::vector<OrientedBoundingBox>& boxes,
                                          const std::vector<float>& scores,
                                          const std::vector<int>& classIds,
                                          float nmsThreshold = 0.45f,
                                          int maxDet = 300) {
    if (boxes.empty()) {
        return {};
    }

    // Create offset boxes to separate classes
    std::vector<OrientedBoundingBox> offsetBoxes = boxes;
    const float offset = 7680.0f; // Large offset to prevent cross-class overlap

    for (size_t i = 0; i < offsetBoxes.size(); ++i) {
        offsetBoxes[i].x += classIds[i] * offset;
        offsetBoxes[i].y += classIds[i] * offset;
    }

    // Apply rotated NMS on offset boxes
    return NMSRotated(offsetBoxes, scores, nmsThreshold, maxDet);
}

} // namespace nms
} // namespace yolos
