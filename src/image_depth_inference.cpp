/**
 * @file image_depth_inference.cpp
 * @brief Monocular metric depth estimation on images using YOLO26-depth models.
 *
 * Prints the depth range in meters and writes a colorized overlay next to a
 * side-by-side view. Depth values are metric: depth.at<float>(y, x) is meters.
 *
 * Usage:
 *   ./image_depth_inference [model.onnx] [image_or_folder] [--save-raw] [--no-display]
 *
 * Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
 */

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "yolos/tasks/depth.hpp"

namespace fs = std::filesystem;

namespace {

bool isImageFile(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
           ext == ".bmp" || ext == ".tiff" || ext == ".tif";
}

std::vector<std::string> collectImageFiles(const std::string& inputPath) {
    std::vector<std::string> files;
    if (fs::is_directory(inputPath)) {
        for (const auto& entry : fs::directory_iterator(inputPath)) {
            if (entry.is_regular_file() && isImageFile(entry.path())) {
                files.push_back(fs::absolute(entry.path()).string());
            }
        }
        std::sort(files.begin(), files.end());
    } else if (fs::is_regular_file(inputPath)) {
        files.push_back(inputPath);
    }
    return files;
}

/// @brief Write a depth map as raw float32, for downstream numeric use
void saveRawDepth(const cv::Mat& depth, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Could not open " << path << " for writing" << std::endl;
        return;
    }
    for (int y = 0; y < depth.rows; ++y) {
        out.write(reinterpret_cast<const char*>(depth.ptr<float>(y)),
                  static_cast<std::streamsize>(depth.cols * sizeof(float)));
    }
    out.close();
    if (!out) {
        std::cerr << "Failed writing raw depth to " << path << std::endl;
        return;
    }
    std::cout << "Wrote raw float32 depth (" << depth.cols << "x" << depth.rows
              << ") to: " << path << std::endl;
}

} // namespace

int main(int argc, char* argv[]) {
    std::string modelPath = "../models/yolo26n-depth.onnx";
    std::string inputPath = "../data/dog.jpg";
    bool saveRaw = false;
    bool showWindows = true;

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--save-raw") {
            saveRaw = true;
        } else if (arg == "--no-display") {
            showWindows = false;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0]
                      << " [model.onnx] [image_or_folder] [--save-raw] [--no-display]\n";
            return 0;
        } else {
            positional.push_back(arg);
        }
    }
    if (positional.size() > 0) modelPath = positional[0];
    if (positional.size() > 1) inputPath = positional[1];

    const std::vector<std::string> imageFiles = collectImageFiles(inputPath);
    if (imageFiles.empty()) {
        std::cerr << "No image files found at: " << inputPath << std::endl;
        return -1;
    }

    const bool useGPU = false;

    try {
        yolos::depth::YOLODepthEstimator estimator(modelPath, useGPU);

        for (const auto& imgPath : imageFiles) {
            cv::Mat image = cv::imread(imgPath);
            if (image.empty()) {
                std::cerr << "Could not read image: " << imgPath << std::endl;
                continue;
            }

            const auto start = std::chrono::high_resolution_clock::now();
            const cv::Mat depth = estimator.estimate(image);
            const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);

            double lo = 0.0;
            double hi = 0.0;
            cv::minMaxLoc(depth, &lo, &hi);

            std::cout << "\nImage: " << imgPath << " size: " << image.size() << std::endl;
            std::cout << "Depth range: " << lo << " - " << hi << " m" << std::endl;
            std::cout << "Center pixel: " << depth.at<float>(depth.rows / 2, depth.cols / 2)
                      << " m" << std::endl;
            std::cout << "Inference: " << duration.count() << " ms" << std::endl;

            cv::Mat overlay = image.clone();
            estimator.drawDepth(overlay, depth);

            const cv::Mat colorized = yolos::drawing::colorizeDepth(depth);

            if (saveRaw) {
                saveRawDepth(depth, fs::path(imgPath).stem().string() + "_depth.bin");
            }

            if (showWindows) {
                cv::Mat sideBySide;
                cv::hconcat(overlay, colorized, sideBySide);
                cv::imshow("YOLO Depth - overlay | colorized", sideBySide);
                std::cout << "Press any key to continue..." << std::endl;
                cv::waitKey(0);
            }
        }

        if (showWindows) {
            cv::destroyAllWindows();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
