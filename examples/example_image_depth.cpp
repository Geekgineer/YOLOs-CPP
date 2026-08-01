/**
 * @file example_image_depth.cpp
 * @brief Monocular depth estimation on images using YOLO26-depth models
 * @details Produces a per-pixel metric depth map and a colorized overlay
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <vector>
#include "yolos/tasks/depth.hpp"
#include "utils.hpp"

using namespace yolos::depth;

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;

    // Default configuration (depth models need no labels file)
    std::string modelPath = "../../models/yolo26n-depth.onnx";
    std::string inputPath = "../../data/dog.jpg";
    std::string outputDir = "../../outputs/depth/";

    if (argc > 1) modelPath = argv[1];
    if (argc > 2) inputPath = argv[2];

    utils::printUsage(argv[0], "Depth", modelPath, inputPath, "(none - depth has no classes)");

    // Collect image files
    std::vector<std::string> imageFiles;
    if (fs::is_directory(inputPath)) {
        for (const auto& entry : fs::directory_iterator(inputPath)) {
            if (entry.is_regular_file() && utils::isImageFile(entry.path().string())) {
                imageFiles.push_back(fs::absolute(entry.path()).string());
            }
        }
        if (imageFiles.empty()) {
            std::cerr << "❌ No image files found in: " << inputPath << std::endl;
            return -1;
        }
    } else if (fs::is_regular_file(inputPath)) {
        imageFiles.push_back(inputPath);
    } else {
        std::cerr << "❌ Invalid path: " << inputPath << std::endl;
        return -1;
    }

    bool useGPU = false;
    std::cout << "🔄 Loading depth model: " << modelPath << std::endl;

    try {
        YOLODepthEstimator estimator(modelPath, useGPU);
        std::cout << "✅ Model loaded successfully!" << std::endl;

        for (const auto& imgPath : imageFiles) {
            std::cout << "\n📷 Processing: " << imgPath << std::endl;

            cv::Mat image = cv::imread(imgPath);
            if (image.empty()) {
                std::cerr << "❌ Could not load image: " << imgPath << std::endl;
                continue;
            }

            auto start = std::chrono::high_resolution_clock::now();
            cv::Mat depth = estimator.estimate(image);
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);

            double lo = 0.0, hi = 0.0;
            cv::minMaxLoc(depth, &lo, &hi);

            std::cout << "✅ Depth estimation completed!" << std::endl;
            std::cout << "📊 Range: " << std::fixed << std::setprecision(2)
                      << lo << " - " << hi << " m" << std::endl;
            std::cout << "   Center pixel: "
                      << depth.at<float>(depth.rows / 2, depth.cols / 2) << " m" << std::endl;

            cv::Mat resultImage = image.clone();
            estimator.drawDepth(resultImage, depth);

            std::string outputPath = utils::saveImage(resultImage, imgPath, outputDir);
            std::cout << "💾 Saved result to: " << outputPath << std::endl;

            utils::printMetrics("Depth", duration.count());

            cv::imshow("YOLO Depth", resultImage);
            std::cout << "Press any key to continue..." << std::endl;
            cv::waitKey(0);
        }

        cv::destroyAllWindows();
        std::cout << "\n✅ All images processed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
