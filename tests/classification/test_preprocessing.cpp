/**
 * @file test_preprocessing.cpp
 * @brief Self-contained tests for the PIL-compatible antialiased resize
 *
 * Ultralytics preprocesses classification images with torchvision's
 * T.Resize(size, BILINEAR) on a PIL image, whose bilinear filter is
 * antialiased. cv::resize(INTER_LINEAR) is not, which made YOLOs-CPP
 * confidences disagree with Ultralytics and could flip the top-1 class
 * (issue #137).
 *
 * The expected values below were produced by Pillow itself:
 *
 *   PIL.Image.fromarray(src).resize((w, h), PIL.Image.BILINEAR)
 *
 * on the deterministic source built by makeSource(). These tests need no model
 * and no Python, so they guard the resize even when the Ultralytics parity run
 * is unavailable.
 */

#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "yolos/core/preprocessing.hpp"

namespace {

constexpr int kSrcW = 13;
constexpr int kSrcH = 9;

/// @brief Deterministic RGB source, identical to the Python generator
cv::Mat makeSource() {
    cv::Mat src(kSrcH, kSrcW, CV_8UC3);
    for (int y = 0; y < kSrcH; ++y) {
        for (int x = 0; x < kSrcW; ++x) {
            src.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>((x * 37 + y * 11) % 256),
                static_cast<uchar>((x * 5 + y * 61) % 256),
                static_cast<uchar>((x * y * 17 + 3) % 256));
        }
    }
    return src;
}

// ---- Pillow reference output -------------------------------------------------

// kGolden5x4: 13x9 -> 5x4
const std::vector<uchar> kGolden5x4 = {
    48, 56, 18, 133, 68, 50, 151, 81, 83, 73, 94, 81, 158, 106, 106, 70, 169, 54, 155, 140, 
    120, 150, 142, 120, 95, 155, 115, 180, 167, 112, 96, 122, 84, 178, 93, 105, 101, 94, 103, 
    121, 107, 112, 200, 119, 158, 118, 186, 94, 172, 184, 127, 78, 125, 124, 143, 129, 130, 
    166, 141, 133
};

// kGolden13x4: 13x9 -> 13x4
const std::vector<uchar> kGolden13x4 = {
    9, 51, 3, 46, 56, 17, 83, 61, 31, 120, 66, 45, 157, 71, 59, 194, 76, 73, 231, 81, 88, 12, 
    86, 102, 49, 91, 64, 86, 96, 79, 123, 101, 93, 160, 106, 107, 197, 111, 121, 31, 168, 3, 
    68, 173, 52, 105, 178, 100, 142, 127, 142, 179, 132, 135, 216, 137, 77, 191, 142, 119, 34, 
    147, 168, 71, 152, 91, 108, 157, 134, 145, 162, 76, 182, 167, 125, 219, 172, 117, 57, 120, 
    3, 94, 125, 90, 131, 130, 178, 168, 79, 72, 205, 84, 103, 223, 89, 97, 29, 94, 78, 60, 99, 
    147, 97, 104, 110, 134, 109, 72, 171, 114, 154, 208, 119, 154, 226, 124, 185, 79, 181, 3, 
    116, 186, 125, 153, 191, 152, 190, 196, 113, 227, 201, 140, 59, 111, 101, 45, 116, 127, 82, 
    121, 140, 119, 126, 115, 156, 131, 127, 193, 136, 154, 230, 141, 115, 62, 146, 142
};

// kGolden5x9: 13x9 -> 5x9
const std::vector<uchar> kGolden5x9 = {
    39, 5, 3, 124, 17, 3, 142, 30, 3, 64, 43, 3, 149, 55, 3, 50, 66, 21, 135, 78, 60, 153, 91, 
    105, 75, 104, 150, 160, 116, 189, 61, 127, 39, 146, 139, 117, 164, 152, 185, 86, 165, 79, 
    171, 177, 119, 72, 188, 57, 157, 200, 136, 175, 213, 75, 97, 226, 133, 182, 238, 66, 83, 
    232, 76, 168, 59, 115, 90, 18, 133, 108, 31, 117, 193, 43, 158, 94, 54, 77, 179, 66, 87, 
    101, 79, 81, 119, 92, 85, 204, 104, 164, 105, 115, 95, 190, 127, 106, 112, 140, 103, 130, 
    153, 147, 215, 165, 171, 116, 176, 113, 163, 188, 163, 65, 201, 146, 141, 214, 115, 149, 
    226, 101, 127, 237, 71, 174, 210, 96, 76, 28, 110, 152, 19, 138, 160, 31, 150
};

void expectMatchesPillow(const cv::Size& size, const std::vector<uchar>& expected) {
    cv::Mat out;
    yolos::preprocessing::resizeAntialiasBilinear(makeSource(), out, size);

    ASSERT_EQ(size.height, out.rows);
    ASSERT_EQ(size.width, out.cols);
    ASSERT_EQ(3, out.channels());
    ASSERT_EQ(expected.size(), out.total() * out.channels());

    size_t i = 0;
    for (int y = 0; y < out.rows; ++y) {
        const uchar* row = out.ptr<uchar>(y);
        for (int x = 0; x < out.cols * 3; ++x, ++i) {
            ASSERT_EQ(static_cast<int>(expected[i]), static_cast<int>(row[x]))
                << "mismatch at row " << y << " byte " << x;
        }
    }
}

} // namespace

// ============================================================================
// Downscaling: the antialiased path
// ============================================================================

TEST(AntialiasResize, MatchesPillowWhenDownscalingBothAxes) {
    expectMatchesPillow(cv::Size(5, 4), kGolden5x4);
}

TEST(AntialiasResize, MatchesPillowWhenDownscalingHeightOnly) {
    expectMatchesPillow(cv::Size(13, 4), kGolden13x4);
}

TEST(AntialiasResize, MatchesPillowWhenDownscalingWidthOnly) {
    expectMatchesPillow(cv::Size(5, 9), kGolden5x9);
}

TEST(AntialiasResize, DiffersFromPlainBilinearWhenDownscaling) {
    // The whole point of the fix: a non-antialiased resize gives a different
    // answer, so this guards against silently reverting to cv::resize.
    const cv::Mat src = makeSource();

    cv::Mat antialiased;
    yolos::preprocessing::resizeAntialiasBilinear(src, antialiased, cv::Size(5, 4));

    cv::Mat plain;
    cv::resize(src, plain, cv::Size(5, 4), 0, 0, cv::INTER_LINEAR);

    cv::Mat diff;
    cv::absdiff(antialiased, plain, diff);
    EXPECT_GT(cv::sum(diff)[0] + cv::sum(diff)[1] + cv::sum(diff)[2], 0.0);
}

// ============================================================================
// Upscaling and identity
// ============================================================================

TEST(AntialiasResize, DegeneratesToBilinearWhenUpscaling) {
    // Pillow clamps the filter support to 1.0 when upscaling, so the result is
    // ordinary bilinear interpolation and must track cv::resize within rounding.
    const cv::Mat src = makeSource();

    for (const cv::Size size : {cv::Size(39, 27), cv::Size(27, 21), cv::Size(13, 36)}) {
        cv::Mat antialiased;
        yolos::preprocessing::resizeAntialiasBilinear(src, antialiased, size);

        cv::Mat plain;
        cv::resize(src, plain, size, 0, 0, cv::INTER_LINEAR);

        cv::Mat diff;
        cv::absdiff(antialiased, plain, diff);
        double maxDiff = 0.0;
        cv::minMaxLoc(diff.reshape(1), nullptr, &maxDiff);
        EXPECT_LE(maxDiff, 1.0) << "upscale to " << size.width << "x" << size.height;
    }
}

TEST(AntialiasResize, SameSizeReturnsAnIndependentCopy) {
    const cv::Mat src = makeSource();

    cv::Mat out;
    yolos::preprocessing::resizeAntialiasBilinear(src, out, cv::Size(kSrcW, kSrcH));

    cv::Mat diff;
    cv::absdiff(src, out, diff);
    EXPECT_EQ(0, cv::countNonZero(diff.reshape(1)));
    EXPECT_NE(src.data, out.data) << "must not alias the source";
}

TEST(AntialiasResize, HandlesSingleChannelInput) {
    cv::Mat gray;
    cv::cvtColor(makeSource(), gray, cv::COLOR_BGR2GRAY);

    cv::Mat out;
    yolos::preprocessing::resizeAntialiasBilinear(gray, out, cv::Size(5, 4));

    EXPECT_EQ(4, out.rows);
    EXPECT_EQ(5, out.cols);
    EXPECT_EQ(1, out.channels());
}
