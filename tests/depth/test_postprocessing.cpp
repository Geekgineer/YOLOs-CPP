/**
 * @file test_postprocessing.cpp
 * @brief Self-contained tests for depth postprocessing and visualization
 *
 * These need no model and no Python: they pin the letterbox crop geometry and
 * the depth colorizer, both of which are ports of Ultralytics behaviour
 * (ops.scale_masks and plotting.colorize_depth).
 */

#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

#include <cmath>
#include <vector>

#include "yolos/core/preprocessing.hpp"

namespace {

/// @brief Dense map with a horizontal ramp, so a crop is detectable by value
cv::Mat rampMap(int rows, int cols) {
    cv::Mat m(rows, cols, CV_32FC1);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            m.at<float>(y, x) = static_cast<float>(x) + 1000.0f * static_cast<float>(y);
        }
    }
    return m;
}

} // namespace

// ============================================================================
// cropLetterboxAndResize
// ============================================================================

TEST(CropLetterboxAndResize, ReturnsRequestedSize) {
    // 640x640 letterbox holding a 640x480 image: gain = 1.0, padH = 80, padW = 0
    cv::Mat out;
    yolos::preprocessing::cropLetterboxAndResize(rampMap(640, 640), out, cv::Size(640, 480));

    EXPECT_EQ(480, out.rows);
    EXPECT_EQ(640, out.cols);
    EXPECT_EQ(CV_32FC1, out.type());
}

TEST(CropLetterboxAndResize, CropsVerticalPaddingForLandscapeSource) {
    // gain = min(640/480, 640/640) = 1.0; round(480*1.0)=480 -> padH = (640-480)/2 = 80
    // top = nearbyint(79.9) = 80, bottom = 640 - nearbyint(80.1) = 640 - 80 = 560
    // The crop is exactly rows [80, 560), which is 480 rows -> no rescale in y.
    cv::Mat out;
    yolos::preprocessing::cropLetterboxAndResize(rampMap(640, 640), out, cv::Size(640, 480));

    // Row 0 of the output must come from row 80 of the input (value 1000*80 + x)
    EXPECT_NEAR(80000.0f, out.at<float>(0, 0), 1e-3f);
    EXPECT_NEAR(80000.0f + 100.0f, out.at<float>(0, 100), 1e-3f);
}

TEST(CropLetterboxAndResize, CropsHorizontalPaddingForPortraitSource) {
    // 480x640 original in a 640x640 letterbox: gain = 1.0, padW = 80, padH = 0
    cv::Mat out;
    yolos::preprocessing::cropLetterboxAndResize(rampMap(640, 640), out, cv::Size(480, 640));

    EXPECT_EQ(640, out.rows);
    EXPECT_EQ(480, out.cols);
    // Column 0 of the output must come from column 80 of the input
    EXPECT_NEAR(80.0f, out.at<float>(0, 0), 1e-3f);
}

TEST(CropLetterboxAndResize, IdenticalSizeReturnsIndependentCopy) {
    const cv::Mat src = rampMap(64, 64);

    cv::Mat out;
    yolos::preprocessing::cropLetterboxAndResize(src, out, cv::Size(64, 64));

    cv::Mat diff;
    cv::absdiff(src, out, diff);
    EXPECT_EQ(0, cv::countNonZero(diff));
    EXPECT_NE(src.data, out.data) << "must not alias the source";
}

TEST(CropLetterboxAndResize, UpscalesReducedResolutionMap) {
    // A map emitted at a quarter of the input resolution still rescales correctly,
    // because gain and padding come from the map's own dimensions.
    cv::Mat out;
    yolos::preprocessing::cropLetterboxAndResize(rampMap(160, 160), out, cv::Size(640, 480));

    EXPECT_EQ(480, out.rows);
    EXPECT_EQ(640, out.cols);
}

TEST(CropLetterboxAndResize, RoundsHalfToEvenWhenScalingTheTargetSize) {
    // 320x320 map, 640x241 target: gain = 0.5 so dstH * gain is exactly 120.5.
    // Python round() gives 120 (half-to-even), so padH = 100 and the crop is rows
    // [100, 220) -> 120 rows. Half-away-from-zero rounding would give 121 rows and
    // shift every value in the rescaled map.
    cv::Mat out;
    yolos::preprocessing::cropLetterboxAndResize(rampMap(320, 320), out, cv::Size(640, 241));

    ASSERT_EQ(241, out.rows);
    ASSERT_EQ(640, out.cols);

    // Row 0 must come from map row 100 (value 1000*100), not row 99.
    // cv::resize maps output row 0 to a source row < 0.5, which clamps to source row 0.
    EXPECT_NEAR(100000.0f, out.at<float>(0, 0), 1.0f);
}

TEST(CropLetterboxAndResize, DegenerateGeometryDoesNotThrowOrEmpty) {
    // A 1x1 map is pathological; the helper must clamp rather than build an invalid ROI.
    cv::Mat out;
    ASSERT_NO_THROW(
        yolos::preprocessing::cropLetterboxAndResize(rampMap(1, 1), out, cv::Size(32, 32)));
    EXPECT_FALSE(out.empty());
    EXPECT_EQ(32, out.rows);
    EXPECT_EQ(32, out.cols);
}
