/**
 * @file example_video_depth.cpp
 * @brief Monocular depth estimation on video using YOLO26-depth models
 * @details Pins the colour range so the overlay does not flicker between frames
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <limits>
#include "yolos/tasks/depth.hpp"
#include "utils.hpp"

using namespace yolos::depth;

int main(int argc, char* argv[]) {
    std::string modelPath = "../../models/yolo26n-depth.onnx";
    std::string inputPath = "../../data/video.mp4";
    std::string outputDir = "../../outputs/depth/";

    if (argc > 1) modelPath = argv[1];
    if (argc > 2) inputPath = argv[2];

    utils::printUsage(argv[0], "Depth (video)", modelPath, inputPath, "(none - depth has no classes)");

    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "❌ Could not open video: " << inputPath << std::endl;
        return -1;
    }

    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 30.0;

    std::string outputPath = utils::getVideoOutputPath(inputPath, outputDir);
    cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, cv::Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "❌ Could not open video writer for: " << outputPath << std::endl;
        cap.release();
        return -1;
    }

    bool useGPU = false;
    std::cout << "🔄 Loading depth model: " << modelPath << std::endl;

    try {
        YOLODepthEstimator estimator(modelPath, useGPU);
        std::cout << "✅ Model loaded successfully!" << std::endl;

        // Per-frame percentile normalization flickers. Derive the disparity range from
        // the first frame and reuse it, so the colours stay stable across the video.
        float vmin = std::numeric_limits<float>::quiet_NaN();
        float vmax = std::numeric_limits<float>::quiet_NaN();
        bool pinFailed = false;   // first frame was degenerate; use per-frame normalization

        cv::Mat frame;
        long frameIndex = 0;
        while (cap.read(frame)) {
            auto start = std::chrono::high_resolution_clock::now();
            cv::Mat depth = estimator.estimate(frame);
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);

            if (std::isnan(vmin) && !pinFailed) {
                double lo = 0.0, hi = 0.0;
                cv::minMaxLoc(depth, &lo, &hi);
                // Disparity: near objects have the largest 1/d, so the range is [1/hi, 1/lo].
                // Pin only a sane first frame. A non-positive or degenerate minimum would
                // pin a range of ~1e6 and flatten everything into one end of the colormap,
                // so fall back to per-frame percentiles instead: it flickers, but it stays
                // readable and it is honest about what happened.
                if (lo > 0.0 && hi > lo) {
                    vmin = static_cast<float>(1.0 / hi);
                    vmax = static_cast<float>(1.0 / lo);
                    std::cout << "📊 Pinned disparity range: " << vmin << " - " << vmax << std::endl;
                } else {
                    std::cout << "⚠️  First frame depth range (" << lo << " - " << hi
                              << " m) is degenerate; using per-frame normalization." << std::endl;
                    pinFailed = true;
                }
            }

            yolos::drawing::drawDepthMap(frame, depth, 0.6f,
                                         yolos::drawing::DepthColormap::Jet,
                                         yolos::drawing::DepthNorm::Disparity,
                                         vmin, vmax);

            writer.write(frame);
            cv::imshow("YOLO Depth (video)", frame);

            if (frameIndex % 30 == 0) {
                utils::printMetrics("Depth", duration.count());
            }
            ++frameIndex;

            if (cv::waitKey(1) == 27) break;  // Esc
        }

        cap.release();
        writer.release();
        cv::destroyAllWindows();
        std::cout << "\n💾 Saved video to: " << outputPath << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
