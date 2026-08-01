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
    // Pins that gain/padding came from the map's own 160x160 dims: crop is rows
    // [20, 140), so output row 0 must come from map row 20 (ramp value 1000*20).
    EXPECT_NEAR(20000.0f, out.at<float>(0, 0), 1.0f);
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

#include "yolos/core/drawing.hpp"

namespace {

/// @brief Depth map ramping 1..10 m left to right, with one invalid column
cv::Mat depthRamp(int rows = 16, int cols = 32) {
    cv::Mat d(rows, cols, CV_32FC1);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            d.at<float>(y, x) = 1.0f + 9.0f * static_cast<float>(x) / (cols - 1);
        }
    }
    d.col(0).setTo(0.0f);  // invalid pixels
    return d;
}

} // namespace

// ============================================================================
// colorizeDepth
// ============================================================================

TEST(ColorizeDepth, ReturnsBgrImageAtInputSize) {
    const cv::Mat colored = yolos::drawing::colorizeDepth(depthRamp());

    EXPECT_EQ(16, colored.rows);
    EXPECT_EQ(32, colored.cols);
    EXPECT_EQ(CV_8UC3, colored.type());
}

TEST(ColorizeDepth, NonPositivePixelsAreBlack) {
    const cv::Mat colored = yolos::drawing::colorizeDepth(depthRamp());

    for (int y = 0; y < colored.rows; ++y) {
        EXPECT_EQ(cv::Vec3b(0, 0, 0), colored.at<cv::Vec3b>(y, 0))
            << "invalid pixel at row " << y << " must be black";
    }
}

TEST(ColorizeDepth, ValidPixelsAreNotAllBlack) {
    const cv::Mat colored = yolos::drawing::colorizeDepth(depthRamp());

    int nonBlack = 0;
    for (int y = 0; y < colored.rows; ++y) {
        for (int x = 1; x < colored.cols; ++x) {
            if (colored.at<cv::Vec3b>(y, x) != cv::Vec3b(0, 0, 0)) ++nonBlack;
        }
    }
    EXPECT_GT(nonBlack, 0);
}

TEST(ColorizeDepth, ConstantDepthProducesUniformColour) {
    // vmax == vmin is guarded with a 1e-6 bump, exactly as Ultralytics does. The guard
    // itself is not directly observable (the [0,1] clamp launders the 0/0 NaN to 0), so
    // what this pins is the observable consequence: a constant-depth map must colorize
    // to a single uniform colour. A NaN leaking through per-pixel would break that.
    cv::Mat flat(8, 8, CV_32FC1, cv::Scalar(3.0f));

    cv::Mat colored;
    ASSERT_NO_THROW(colored = yolos::drawing::colorizeDepth(flat));
    ASSERT_EQ(CV_8UC3, colored.type());

    const cv::Vec3b first = colored.at<cv::Vec3b>(0, 0);
    for (int y = 0; y < colored.rows; ++y) {
        for (int x = 0; x < colored.cols; ++x) {
            ASSERT_EQ(first, colored.at<cv::Vec3b>(y, x))
                << "non-uniform colour at (" << y << "," << x << ") for constant depth";
        }
    }
}

TEST(ColorizeDepth, AllInvalidDepthIsAllBlack) {
    cv::Mat zeros(8, 8, CV_32FC1, cv::Scalar(0.0f));

    cv::Mat colored;
    ASSERT_NO_THROW(colored = yolos::drawing::colorizeDepth(zeros));
    EXPECT_EQ(0, cv::countNonZero(colored.reshape(1)));
}

TEST(ColorizeDepth, ExplicitRangeIsHonoured) {
    using yolos::drawing::DepthColormap;
    using yolos::drawing::DepthNorm;
    const cv::Mat d = depthRamp();

    // Two different explicit ranges must colorize the same map differently. If vmin/vmax
    // were ignored and the percentiles recomputed, both calls would return the same image.
    const cv::Mat narrow = yolos::drawing::colorizeDepth(
        d, DepthColormap::Jet, DepthNorm::Disparity, 0.10f, 0.20f);
    const cv::Mat wide = yolos::drawing::colorizeDepth(
        d, DepthColormap::Jet, DepthNorm::Disparity, 0.10f, 1.00f);

    cv::Mat diff;
    cv::absdiff(narrow, wide, diff);
    EXPECT_GT(cv::sum(diff)[0] + cv::sum(diff)[1] + cv::sum(diff)[2], 0.0)
        << "vmin/vmax appear to be ignored";

    // And a narrow explicit range must differ from the derived (percentile) range.
    const cv::Mat derived = yolos::drawing::colorizeDepth(d);
    cv::absdiff(narrow, derived, diff);
    EXPECT_GT(cv::sum(diff)[0] + cv::sum(diff)[1] + cv::sum(diff)[2], 0.0)
        << "explicit range produced the same result as the derived range";

    // Secondary property, and the reason the parameters exist: with a pinned range, a
    // far outlier in one frame cannot shift the colours of the rest of the image.
    cv::Mat outlier = d.clone();
    outlier.at<float>(0, 5) = 500.0f;
    const cv::Mat pinned = yolos::drawing::colorizeDepth(
        outlier, DepthColormap::Jet, DepthNorm::Disparity, 0.10f, 1.00f);
    EXPECT_EQ(wide.at<cv::Vec3b>(8, 20), pinned.at<cv::Vec3b>(8, 20));
}

TEST(ColorizeDepth, MetricModeDiffersFromDisparityMode) {
    const cv::Mat disp = yolos::drawing::colorizeDepth(
        depthRamp(), yolos::drawing::DepthColormap::Jet, yolos::drawing::DepthNorm::Disparity);
    const cv::Mat metric = yolos::drawing::colorizeDepth(
        depthRamp(), yolos::drawing::DepthColormap::Jet, yolos::drawing::DepthNorm::Metric);

    cv::Mat diff;
    cv::absdiff(disp, metric, diff);
    EXPECT_GT(cv::sum(diff)[0] + cv::sum(diff)[1] + cv::sum(diff)[2], 0.0);
}

// ============================================================================
// drawDepthMap
// ============================================================================

TEST(DrawDepthMap, BlendsInPlaceKeepingSizeAndType) {
    cv::Mat image(16, 32, CV_8UC3, cv::Scalar(10, 20, 30));
    const cv::Mat before = image.clone();

    yolos::drawing::drawDepthMap(image, depthRamp());

    EXPECT_EQ(16, image.rows);
    EXPECT_EQ(32, image.cols);
    EXPECT_EQ(CV_8UC3, image.type());

    cv::Mat diff;
    cv::absdiff(before, image, diff);
    EXPECT_GT(cv::sum(diff)[0] + cv::sum(diff)[1] + cv::sum(diff)[2], 0.0)
        << "overlay must change the image";
}

TEST(DrawDepthMap, ResizesDepthToImageWhenSizesDiffer) {
    cv::Mat image(32, 64, CV_8UC3, cv::Scalar(10, 20, 30));

    ASSERT_NO_THROW(yolos::drawing::drawDepthMap(image, depthRamp(16, 32)));
    EXPECT_EQ(32, image.rows);
    EXPECT_EQ(64, image.cols);
}

TEST(DrawDepthMap, EmptyDepthLeavesImageUnchanged) {
    cv::Mat image(16, 32, CV_8UC3, cv::Scalar(10, 20, 30));
    const cv::Mat before = image.clone();

    yolos::drawing::drawDepthMap(image, cv::Mat());

    cv::Mat diff;
    cv::absdiff(before, image, diff);
    EXPECT_EQ(0, cv::countNonZero(diff.reshape(1)));
}
