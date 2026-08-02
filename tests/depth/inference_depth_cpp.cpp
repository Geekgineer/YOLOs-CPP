/**
 * @file inference_depth_cpp.cpp
 * @brief Depth inference for the YOLOs-CPP parity suite
 * @details Writes one raw float32 map per image plus a JSON index, matching the
 *          layout inference_depth_ultralytics.py produces.
 */

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "yolos/tasks/depth.hpp"

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

const std::string kDepthSubdir = "results/depth/cpp";

std::vector<std::string> listImages(const std::string& dir) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            files.push_back(fs::absolute(entry.path()).string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

void writeRaw(const cv::Mat& depth, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    for (int y = 0; y < depth.rows; ++y) {
        out.write(reinterpret_cast<const char*>(depth.ptr<float>(y)),
                  static_cast<std::streamsize>(depth.cols * sizeof(float)));
    }
}

} // namespace

int main(int argc, char* argv[]) {
    std::cout << "=== YOLOs-CPP Depth Test ===" << std::endl;
    const bool isGPU = argc > 1 && std::string(argv[1]) == "gpu";

    const std::string basePath = XSTRING(BASE_PATH_DEPTH);
    const std::string imagesPath = basePath + "data/images/";
    const std::string weightsPath = basePath + "models/";
    const std::string resultsPath = basePath + "results/";
    const std::string depthDir = basePath + kDepthSubdir + "/";

    if (!fs::exists(imagesPath) || !fs::exists(weightsPath)) {
        std::cerr << "Missing images or models directory under " << basePath << std::endl;
        return -1;
    }

    const std::vector<std::string> imageFiles = listImages(imagesPath);
    if (imageFiles.empty()) {
        std::cerr << "No images found in " << imagesPath << std::endl;
        return -1;
    }

    fs::create_directories(depthDir);

    // Explicit list, like the detection and segmentation harnesses. Iterating
    // models/*.onnx would also pick up the synthetic fixtures that
    // make_synthetic_models.py writes there, including not_depth.onnx, which the
    // estimator is designed to reject at construction.
    // Keep in sync with PARITY_MODELS in inference_depth_ultralytics.py.
    const std::vector<std::string> parityModels = {"yolo26n-depth"};

    json out;
    for (const std::string& modelStem : parityModels) {
        const std::string modelPath = weightsPath + modelStem + ".onnx";
        if (!fs::exists(modelPath)) {
            std::cout << "Skipping " << modelStem << ": " << modelPath << " not found." << std::endl;
            continue;
        }
        std::cout << "\n======== Running: " << modelStem << " ========" << std::endl;

        yolos::depth::YOLODepthEstimator estimator(modelPath, isGPU);

        json entries = json::array();
        for (const auto& imgPath : imageFiles) {
            cv::Mat image = cv::imread(imgPath);
            if (image.empty()) {
                std::cerr << "Could not read " << imgPath << std::endl;
                continue;
            }

            const cv::Mat depth = estimator.estimate(image);

            double lo = 0.0;
            double hi = 0.0;
            cv::minMaxLoc(depth, &lo, &hi);

            const std::string stem = fs::path(imgPath).stem().string();
            const std::string rel = kDepthSubdir + "/" + modelStem + "__" + stem + ".bin";
            writeRaw(depth, basePath + rel);

            entries.push_back({
                {"image_path", imgPath},
                {"depth_file", rel},
                {"height", depth.rows},
                {"width", depth.cols},
                {"min", lo},
                {"max", hi},
            });

            std::cout << "  " << stem << ": " << depth.cols << "x" << depth.rows
                      << " range " << lo << "-" << hi << " m" << std::endl;
        }

        out[modelStem] = {
            {"weights_path", fs::absolute(modelPath).string()},
            {"task", "depth"},
            {"results", entries},
        };
    }

    const std::string resultsFile = resultsPath + "results_cpp.json";
    std::ofstream file(resultsFile);
    file << std::setw(2) << out << std::endl;

    std::cout << "\nResults saved to: " << resultsFile << std::endl;
    return 0;
}
