/**
 * @file example_camera_depth.cpp
 * @brief Live monocular depth estimation from a camera using YOLO26-depth models
 * @details Pins the colour range after warm-up so the overlay does not flicker
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include "yolos/tasks/depth.hpp"
#include "utils.hpp"

using namespace yolos::depth;

int main(int argc, char* argv[]) {
    std::string modelPath = "../../models/yolo26n-depth.onnx";
    int cameraIndex = 0;

    if (argc > 1) modelPath = argv[1];
    if (argc > 2) cameraIndex = std::stoi(argv[2]);

    utils::printUsage(argv[0], "Depth (camera)", modelPath,
                      "camera " + std::to_string(cameraIndex),
                      "(none - depth has no classes)");

    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "❌ Could not open camera " << cameraIndex << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    bool useGPU = false;
    std::cout << "🔄 Loading depth model: " << modelPath << std::endl;

    try {
        YOLODepthEstimator estimator(modelPath, useGPU);
        std::cout << "✅ Model loaded. Press Esc to quit, 'r' to re-pin the colour range."
                  << std::endl;

        float vmin = std::numeric_limits<float>::quiet_NaN();
        float vmax = std::numeric_limits<float>::quiet_NaN();
        bool pinFailed = false;   // first frame was degenerate; use per-frame normalization

        cv::Mat frame;
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

            const float centre = depth.at<float>(depth.rows / 2, depth.cols / 2);
            std::ostringstream hud;
            hud << std::fixed << std::setprecision(2) << centre << " m  |  "
                << duration.count() << " ms";
            cv::putText(frame, hud.str(), cv::Point(12, 32), cv::FONT_HERSHEY_SIMPLEX,
                        0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

            cv::imshow("YOLO Depth (camera)", frame);

            const int key = cv::waitKey(1);
            if (key == 27) break;                                        // Esc
            if (key == 'r') {                                            // re-pin the colour range
                vmin = std::numeric_limits<float>::quiet_NaN();
                pinFailed = false;
            }
        }

        cap.release();
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
