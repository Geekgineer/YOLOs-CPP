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
#include <algorithm>
#include <limits>

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

// ============================================================================
// Depth Visualization
// ============================================================================

/// @brief Colormap for depth visualization
/// @note Ultralytics also offers "spectral" (matplotlib Spectral_r). It has no OpenCV
///       equivalent and would need a hand-embedded 256x3 LUT, so it is not supported.
enum class DepthColormap {
    Jet,      ///< Ultralytics default
    Inferno
};

/// @brief How depth values are mapped onto the colormap range
enum class DepthNorm {
    Disparity,  ///< Normalize 1/depth between the 2nd and 98th percentile (Ultralytics default)
    Metric      ///< Normalize depth linearly between its min and max
};

namespace detail {

/// @brief numpy.percentile with linear interpolation, over a mutable pool
/// @param pool Values to summarize; partially reordered in place
/// @param q Percentile in [0, 100]
inline float percentileInPlace(std::vector<float>& pool, double q) {
    if (pool.empty()) return 0.0f;
    if (pool.size() == 1) return pool[0];

    const double pos = (static_cast<double>(pool.size()) - 1.0) * q / 100.0;
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);

    std::nth_element(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(lo), pool.end());
    const double loVal = pool[lo];
    if (hi == lo) return static_cast<float>(loVal);

    // The hi-th element is now somewhere in [lo, end); the smallest of that tail.
    const double hiVal = *std::min_element(pool.begin() + static_cast<std::ptrdiff_t>(lo) + 1, pool.end());
    return static_cast<float>(loVal * (1.0 - frac) + hiVal * frac);
}

} // namespace detail

/// @brief Colorize a metric depth map
/// @param depth Depth in meters (CV_32FC1); values <= 0 are treated as invalid
/// @param cmap Colormap to apply
/// @param mode Disparity or metric normalization
/// @param vmin Lower bound of the colour range; NaN derives it as Ultralytics does
/// @param vmax Upper bound of the colour range; NaN derives it as Ultralytics does
/// @return BGR CV_8UC3 image at @p depth's size, invalid pixels black
/// @note Port of ultralytics.utils.plotting.colorize_depth(). Pass explicit bounds for
///       video: per-frame percentiles make the overlay flicker between frames.
inline cv::Mat colorizeDepth(const cv::Mat& depth,
                             DepthColormap cmap = DepthColormap::Jet,
                             DepthNorm mode = DepthNorm::Disparity,
                             float vmin = std::numeric_limits<float>::quiet_NaN(),
                             float vmax = std::numeric_limits<float>::quiet_NaN()) {
    CV_Assert(!depth.empty() && depth.type() == CV_32FC1);

    const int rows = depth.rows;
    const int cols = depth.cols;

    // Build the normalization source: disparity (1/d) or metric depth, 0 where invalid
    cv::Mat values(rows, cols, CV_32FC1, cv::Scalar(0.0f));
    std::vector<float> pool;
    pool.reserve(static_cast<size_t>(rows) * cols);

    for (int y = 0; y < rows; ++y) {
        const float* dRow = depth.ptr<float>(y);
        float* vRow = values.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            if (dRow[x] > 0.0f) {
                vRow[x] = (mode == DepthNorm::Disparity) ? (1.0f / dRow[x]) : dRow[x];
                pool.push_back(vRow[x]);
            }
        }
    }

    float lo = vmin;
    float hi = vmax;
    if (std::isnan(lo) || std::isnan(hi)) {
        float derivedLo = 0.0f;
        float derivedHi = 1.0f;
        if (!pool.empty()) {
            if (mode == DepthNorm::Disparity) {
                derivedLo = detail::percentileInPlace(pool, 2.0);
                derivedHi = detail::percentileInPlace(pool, 98.0);
            } else {
                const auto minMax = std::minmax_element(pool.begin(), pool.end());
                derivedLo = *minMax.first;
                derivedHi = *minMax.second;
            }
        }
        if (std::isnan(lo)) lo = derivedLo;
        if (std::isnan(hi)) hi = derivedHi;
    }
    if (hi <= lo) hi = lo + 1e-6f;

    // numpy does (dn * 255).astype(uint8), which truncates rather than rounds
    cv::Mat indices(rows, cols, CV_8UC1);
    const float span = hi - lo;
    for (int y = 0; y < rows; ++y) {
        const float* vRow = values.ptr<float>(y);
        uchar* iRow = indices.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x) {
            float dn = (vRow[x] - lo) / span;
            dn = std::min(1.0f, std::max(0.0f, dn));
            iRow[x] = static_cast<uchar>(dn * 255.0f);
        }
    }

    cv::Mat colored;
    cv::applyColorMap(indices,
                      colored,
                      (cmap == DepthColormap::Inferno) ? cv::COLORMAP_INFERNO : cv::COLORMAP_JET);

    // Invalid pixels are black, not whatever the colormap maps index 0 to
    for (int y = 0; y < rows; ++y) {
        const float* dRow = depth.ptr<float>(y);
        cv::Vec3b* cRow = colored.ptr<cv::Vec3b>(y);
        for (int x = 0; x < cols; ++x) {
            if (!(dRow[x] > 0.0f)) cRow[x] = cv::Vec3b(0, 0, 0);
        }
    }

    return colored;
}

/// @brief Blend a colorized depth map over an image
/// @param image Image to draw on, modified in place (CV_8UC3)
/// @param depth Depth in meters (CV_32FC1); resized to @p image if sizes differ
/// @param alpha Blend factor for the heatmap
/// @param cmap Colormap to apply
/// @param mode Disparity or metric normalization
/// @param vmin Lower bound of the colour range; NaN derives it
/// @param vmax Upper bound of the colour range; NaN derives it
/// @note Port of ultralytics.utils.plotting.Annotator.depth_map().
inline void drawDepthMap(cv::Mat& image,
                         const cv::Mat& depth,
                         float alpha = 0.6f,
                         DepthColormap cmap = DepthColormap::Jet,
                         DepthNorm mode = DepthNorm::Disparity,
                         float vmin = std::numeric_limits<float>::quiet_NaN(),
                         float vmax = std::numeric_limits<float>::quiet_NaN()) {
    if (image.empty() || depth.empty()) {
        return;
    }

    cv::Mat heat = colorizeDepth(depth, cmap, mode, vmin, vmax);
    if (heat.size() != image.size()) {
        cv::resize(heat, heat, image.size(), 0, 0, cv::INTER_LINEAR);
    }

    cv::addWeighted(image, 1.0 - alpha, heat, alpha, 0.0, image);
}

} // namespace drawing
} // namespace yolos
