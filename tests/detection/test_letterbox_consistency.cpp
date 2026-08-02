/**
 * @file test_letterbox_consistency.cpp
 * @brief Pins the letterbox forward/inverse contract in yolos::preprocessing
 *
 * Detection, segmentation, pose and OBB all letterbox with letterBoxToBlob() and then
 * descale their outputs with getScalePad(). Those two must agree about where the image
 * was placed. They did not: letterBoxToBlob padded by an integer nearbyint(dw - 0.1)
 * while getScalePad returned the unrounded dw, so every box was descaled against
 * padding that had never been applied. The error is up to half a letterbox pixel, which
 * the gain then magnifies - on a 1600x2128 image at 320 it reached 3.3 original pixels.
 *
 * The parity suite could not see it: its tolerance was +-50 px.
 *
 * These tests need no model and no Python.
 */

#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

#include <cmath>
#include <vector>

#include "yolos/core/preprocessing.hpp"

namespace {

/// @brief Column at which real image content starts in a letterboxed blob
/// @param blob CHW blob produced by letterBoxToBlob
/// @param size Letterbox size
/// @return First column whose value differs from the 114/255 padding fill
int firstContentColumn(const std::vector<float>& blob, const cv::Size& size) {
    constexpr float padNorm = 114.0f / 255.0f;
    const int row = size.height / 2;
    for (int x = 0; x < size.width; ++x) {
        if (std::abs(blob[static_cast<size_t>(row) * size.width + x] - padNorm) > 0.05f) {
            return x;
        }
    }
    return -1;
}

/// @brief Row at which real image content starts in a letterboxed blob
int firstContentRow(const std::vector<float>& blob, const cv::Size& size) {
    constexpr float padNorm = 114.0f / 255.0f;
    const int col = size.width / 2;
    for (int y = 0; y < size.height; ++y) {
        if (std::abs(blob[static_cast<size_t>(y) * size.width + col] - padNorm) > 0.05f) {
            return y;
        }
    }
    return -1;
}

} // namespace

// ============================================================================
// getScalePad must report the padding letterBoxToBlob actually applied
// ============================================================================

TEST(LetterboxConsistency, ScalePadMatchesAppliedPadding) {
    // Each pair is (source size, letterbox size). The odd cases are the ones that used
    // to break: when (target - resized) is odd, dw lands on .5 and the applied integer
    // pad and the returned float pad diverged.
    const std::vector<std::pair<cv::Size, cv::Size>> cases = {
        {cv::Size(640, 480), cv::Size(640, 640)},    // even vertical pad
        {cv::Size(480, 640), cv::Size(640, 640)},    // even horizontal pad
        {cv::Size(481, 640), cv::Size(640, 640)},    // ODD horizontal pad (dw = 79.5)
        {cv::Size(640, 481), cv::Size(640, 640)},    // ODD vertical pad
        {cv::Size(1600, 2128), cv::Size(320, 320)},  // ODD, small gain: the 3.3 px case
        {cv::Size(302, 329), cv::Size(320, 320)},    // a real test image's shape
        {cv::Size(1280, 720), cv::Size(640, 640)},
        {cv::Size(500, 500), cv::Size(640, 640)},    // square, no padding at all
    };

    for (const auto& [srcSize, boxSize] : cases) {
        // Black source on the 114-grey pad makes the content boundary unambiguous.
        const cv::Mat src(srcSize, CV_8UC3, cv::Scalar(0, 0, 0));

        yolos::preprocessing::InferenceBuffer buffer;
        cv::Size actual;
        yolos::preprocessing::letterBoxToBlob(src, buffer, 3, boxSize, actual, false);
        ASSERT_EQ(boxSize, actual);

        float scale = 0.0f;
        float padX = 0.0f;
        float padY = 0.0f;
        yolos::preprocessing::getScalePad(srcSize, boxSize, scale, padX, padY);

        const std::string what = std::to_string(srcSize.width) + "x" +
                                 std::to_string(srcSize.height) + " -> " +
                                 std::to_string(boxSize.width) + "x" +
                                 std::to_string(boxSize.height);

        // Compare as float, not via static_cast<int>: truncating would silently accept
        // a fractional pad such as 79.5 as "equal to" the applied 79, which is exactly
        // the bug this test exists to catch.
        EXPECT_FLOAT_EQ(static_cast<float>(firstContentColumn(buffer.blob, boxSize)), padX)
            << what << ": getScalePad padX disagrees with the padding actually applied";
        EXPECT_FLOAT_EQ(static_cast<float>(firstContentRow(buffer.blob, boxSize)), padY)
            << what << ": getScalePad padY disagrees with the padding actually applied";
    }
}

TEST(LetterboxConsistency, ScalePadReturnsWholePixels) {
    // The letterbox can only pad by whole pixels, so a fractional pad is by definition
    // a pad that was never applied. This is the invariant the old code violated.
    float scale = 0.0f;
    float padX = 0.0f;
    float padY = 0.0f;
    yolos::preprocessing::getScalePad(cv::Size(481, 640), cv::Size(640, 640), scale, padX, padY);

    EXPECT_FLOAT_EQ(std::floor(padX), padX) << "padX = " << padX << " is not a whole pixel";
    EXPECT_FLOAT_EQ(std::floor(padY), padY) << "padY = " << padY << " is not a whole pixel";
}

// ============================================================================
// Tie rounding: Python round() breaks ties to even, std::round does not
// ============================================================================

TEST(LetterboxConsistency, ResizedDimensionBreaksTiesToEven) {
    // 193x256 letterboxed to 640: the scale is exactly 2.5, so the resized width is
    // exactly 482.5. Ultralytics' round() gives 482 (ties to even); std::round would
    // give 483 and place one extra column of image content.
    const cv::Size srcSize(193, 256);
    const cv::Size boxSize(640, 640);

    const cv::Mat src(srcSize, CV_8UC3, cv::Scalar(0, 0, 0));
    yolos::preprocessing::InferenceBuffer buffer;
    cv::Size actual;
    yolos::preprocessing::letterBoxToBlob(src, buffer, 3, boxSize, actual, false);

    // 482 wide content leaves (640 - 482) / 2 = 79 columns of padding on each side.
    EXPECT_EQ(79, firstContentColumn(buffer.blob, boxSize))
        << "resized width should be 482 (ties to even), not 483";

    float scale = 0.0f;
    float padX = 0.0f;
    float padY = 0.0f;
    yolos::preprocessing::getScalePad(srcSize, boxSize, scale, padX, padY);
    EXPECT_FLOAT_EQ(79.0f, padX) << "getScalePad must break the same tie the same way";
}
