#pragma once

// ============================================================================
// YOLO Drawing Utilities
// ============================================================================
// Visualization functions for detection results including bounding boxes,
// labels, masks, and pose skeletons.
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <random>
#include <unordered_map>
#include <cmath>

#include "yolos/core/types.hpp"

namespace yolos {
namespace drawing {

// ============================================================================
// Color Generation
// ============================================================================

/// @brief Generate consistent random colors for each class
/// @param classNames Vector of class names
/// @param seed Random seed for reproducibility
/// @return Vector of BGR colors
inline std::vector<cv::Scalar> generateColors(const std::vector<std::string>& classNames, int seed = 42) {
    // Static cache to avoid regenerating colors
    static std::unordered_map<size_t, std::vector<cv::Scalar>> colorCache;

    // Compute hash key from class names
    size_t hashKey = 0;
    for (const auto& name : classNames) {
        hashKey ^= std::hash<std::string>{}(name) + 0x9e3779b9 + (hashKey << 6) + (hashKey >> 2);
    }

    // Check cache
    auto it = colorCache.find(hashKey);
    if (it != colorCache.end()) {
        return it->second;
    }

    // Generate colors
    std::vector<cv::Scalar> colors;
    colors.reserve(classNames.size());

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);

    for (size_t i = 0; i < classNames.size(); ++i) {
        colors.emplace_back(cv::Scalar(dist(rng), dist(rng), dist(rng)));
    }

    colorCache[hashKey] = colors;
    return colors;
}

/// @brief Get the Ultralytics pose palette colors
/// @return Vector of BGR colors for pose visualization
inline const std::vector<cv::Scalar>& getPosePalette() {
    static const std::vector<cv::Scalar> palette = {
        cv::Scalar(0, 128, 255),    // 0
        cv::Scalar(51, 153, 255),   // 1
        cv::Scalar(102, 178, 255),  // 2
        cv::Scalar(0, 230, 230),    // 3
        cv::Scalar(255, 153, 255),  // 4
        cv::Scalar(255, 204, 153),  // 5
        cv::Scalar(255, 102, 255),  // 6
        cv::Scalar(255, 51, 255),   // 7
        cv::Scalar(255, 178, 102),  // 8
        cv::Scalar(255, 153, 51),   // 9
        cv::Scalar(153, 153, 255),  // 10
        cv::Scalar(102, 102, 255),  // 11
        cv::Scalar(51, 51, 255),    // 12
        cv::Scalar(153, 255, 153),  // 13
        cv::Scalar(102, 255, 102),  // 14
        cv::Scalar(51, 255, 51),    // 15
        cv::Scalar(0, 255, 0),      // 16
        cv::Scalar(255, 0, 0),      // 17
        cv::Scalar(0, 0, 255),      // 18
        cv::Scalar(255, 255, 255)   // 19
    };
    return palette;
}

// ============================================================================
// Bounding Box Drawing
// ============================================================================

/// @brief Draw a single bounding box with label on an image
/// @param image Image to draw on
/// @param box Bounding box
/// @param label Text label
/// @param color Box color
/// @param thickness Line thickness
inline void drawBoundingBox(cv::Mat& image,
                           const BoundingBox& box,
                           const std::string& label,
                           const cv::Scalar& color,
                           int thickness = 2) {
    // Draw rectangle
    cv::rectangle(image,
                  cv::Point(box.x, box.y),
                  cv::Point(box.x + box.width, box.y + box.height),
                  color, thickness, cv::LINE_AA);

    // Draw label background and text
    if (!label.empty()) {
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = std::min(image.rows, image.cols) * 0.0008;
        fontScale = std::max(fontScale, 0.4);
        int textThickness = std::max(1, static_cast<int>(std::min(image.rows, image.cols) * 0.002));
        int baseline = 0;

        cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, textThickness, &baseline);

        int labelY = std::max(box.y, textSize.height + 5);
        cv::Point labelTopLeft(box.x, labelY - textSize.height - 5);
        cv::Point labelBottomRight(box.x + textSize.width + 5, labelY + baseline - 5);

        cv::rectangle(image, labelTopLeft, labelBottomRight, color, cv::FILLED);
        cv::putText(image, label, cv::Point(box.x + 2, labelY - 2),
                    fontFace, fontScale, cv::Scalar(255, 255, 255), textThickness, cv::LINE_AA);
    }
}

/// @brief Draw a bounding box with semi-transparent mask fill
/// @param image Image to draw on
/// @param box Bounding box
/// @param label Text label
/// @param color Box color
/// @param maskAlpha Transparency of the mask fill (0-1)
inline void drawBoundingBoxWithMask(cv::Mat& image,
                                    const BoundingBox& box,
                                    const std::string& label,
                                    const cv::Scalar& color,
                                    float maskAlpha = 0.4f) {
    // Draw semi-transparent fill
    cv::Mat overlay = image.clone();
    cv::rectangle(overlay,
                  cv::Rect(box.x, box.y, box.width, box.height),
                  color, cv::FILLED);
    cv::addWeighted(overlay, maskAlpha, image, 1.0f - maskAlpha, 0, image);

    // Draw box border and label
    drawBoundingBox(image, box, label, color, 2);
}

// ============================================================================
// Oriented Bounding Box Drawing
// ============================================================================

/// @brief Draw an oriented bounding box on an image
/// @param image Image to draw on
/// @param obb Oriented bounding box
/// @param label Text label
/// @param color Box color
/// @param thickness Line thickness
inline void drawOrientedBoundingBox(cv::Mat& image,
                                    const OrientedBoundingBox& obb,
                                    const std::string& label,
                                    const cv::Scalar& color,
                                    int thickness = 2) {
    // Create rotated rectangle
    cv::RotatedRect rotatedRect(
        cv::Point2f(obb.x, obb.y),
        cv::Size2f(obb.width, obb.height),
        obb.angle * 180.0f / static_cast<float>(CV_PI)
    );

    // Get vertices and draw
    cv::Point2f vertices[4];
    rotatedRect.points(vertices);

    for (int i = 0; i < 4; ++i) {
        cv::line(image, vertices[i], vertices[(i + 1) % 4], color, thickness, cv::LINE_AA);
    }

    // Draw label
    if (!label.empty()) {
        int fontFace = cv::FONT_HERSHEY_DUPLEX;
        double fontScale = 0.5;
        int textThickness = 1;
        int baseline;

        cv::Size labelSize = cv::getTextSize(label, fontFace, fontScale, textThickness, &baseline);

        int x = static_cast<int>(obb.x - obb.width / 2);
        int y = static_cast<int>(obb.y - obb.height / 2) - 5;

        x = std::max(0, std::min(x, image.cols - labelSize.width));
        y = std::max(labelSize.height, std::min(y, image.rows - baseline));

        cv::Scalar labelBgColor = color * 0.6;
        cv::rectangle(image,
                      cv::Rect(x, y - labelSize.height, labelSize.width, labelSize.height + baseline),
                      labelBgColor, cv::FILLED);
        cv::putText(image, label, cv::Point(x, y),
                    fontFace, fontScale, cv::Scalar::all(255), textThickness, cv::LINE_AA);
    }
}

// ============================================================================
// Pose Drawing
// ============================================================================

/// @brief Draw pose keypoints and skeleton on an image
/// @param image Image to draw on
/// @param keypoints Vector of keypoints
/// @param skeleton Skeleton connections
/// @param kptRadius Keypoint circle radius
/// @param kptThreshold Minimum confidence to draw keypoint
/// @param lineThickness Skeleton line thickness
inline void drawPoseSkeleton(cv::Mat& image,
                             const std::vector<KeyPoint>& keypoints,
                             const std::vector<std::pair<int, int>>& skeleton,
                             int kptRadius = 4,
                             float kptThreshold = 0.5f,
                             int lineThickness = 2) {
    const auto& palette = getPosePalette();

    // Keypoint color indices (for 17 COCO keypoints)
    static const std::vector<int> kptColorIndices = {16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9};
    // Limb color indices
    static const std::vector<int> limbColorIndices = {9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16};

    // Prepare keypoint positions
    std::vector<cv::Point> kptPoints(keypoints.size(), cv::Point(-1, -1));
    std::vector<bool> valid(keypoints.size(), false);

    // Draw keypoints
    for (size_t i = 0; i < keypoints.size(); ++i) {
        if (keypoints[i].confidence >= kptThreshold) {
            int x = static_cast<int>(std::round(keypoints[i].x));
            int y = static_cast<int>(std::round(keypoints[i].y));
            kptPoints[i] = cv::Point(x, y);
            valid[i] = true;

            int colorIdx = (i < kptColorIndices.size()) ? kptColorIndices[i] : 0;
            cv::circle(image, cv::Point(x, y), kptRadius, palette[colorIdx], -1, cv::LINE_AA);
        }
    }

    // Draw skeleton
    for (size_t j = 0; j < skeleton.size(); ++j) {
        int src = skeleton[j].first;
        int dst = skeleton[j].second;

        if (src < static_cast<int>(keypoints.size()) &&
            dst < static_cast<int>(keypoints.size()) &&
            valid[src] && valid[dst]) {
            int limbColorIdx = (j < limbColorIndices.size()) ? limbColorIndices[j] : 0;
            cv::line(image, kptPoints[src], kptPoints[dst],
                     palette[limbColorIdx], lineThickness, cv::LINE_AA);
        }
    }
}

// ============================================================================
// Segmentation Mask Drawing
// ============================================================================

/// @brief Draw a segmentation mask on an image
/// @param image Image to draw on
/// @param mask Binary mask (CV_8UC1)
/// @param color Mask color
/// @param alpha Mask transparency (0-1)
inline void drawSegmentationMask(cv::Mat& image,
                                 const cv::Mat& mask,
                                 const cv::Scalar& color,
                                 float alpha = 0.5f) {
    if (mask.empty()) {
        return;
    }

    cv::Mat maskGray;
    if (mask.channels() == 3) {
        cv::cvtColor(mask, maskGray, cv::COLOR_BGR2GRAY);
    } else {
        maskGray = mask;
    }

    cv::Mat maskBinary;
    cv::threshold(maskGray, maskBinary, 127, 255, cv::THRESH_BINARY);

    cv::Mat coloredMask;
    cv::cvtColor(maskBinary, coloredMask, cv::COLOR_GRAY2BGR);
    coloredMask.setTo(color, maskBinary);

    cv::addWeighted(image, 1.0, coloredMask, alpha, 0, image);
}

} // namespace drawing
} // namespace yolos
