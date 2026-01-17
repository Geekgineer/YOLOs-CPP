/**
 * @file inference_obb_cpp.cpp
 * @brief OBB (Oriented Bounding Box) inference test for YOLOs-CPP
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <unordered_map>
#include <nlohmann/json.hpp>

// New architecture header
#include "yolos/tasks/obb.hpp"

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace yolos::obb;

struct SingleOBBInferenceResult {
    int classId;
    float conf;
    float cx, cy, width, height, angle;
    float normalized_cx, normalized_cy, normalized_width, normalized_height;
};

struct Results {
    std::string weightsPath;
    std::string task;
    std::unordered_map<std::string, std::vector<SingleOBBInferenceResult>> inferenceResults;
};

bool validatePaths(const std::unordered_map<std::string, std::string>& paths) {
    for (const auto& [key, path] : paths) {
        if (!fs::exists(path)) {
            std::cerr << "Error: " << key << " path does not exist: " << path << std::endl;
            return false;
        }
    }
    std::cout << "All paths are valid." << std::endl;
    return true;
}

void loadInferenceConfig(const std::string& configFilePath, std::unordered_map<std::string, std::string>& config) {
    if (!fs::exists(configFilePath)) return;
    std::ifstream file(configFilePath);
    if (!file.is_open()) return;
    json jsonConfig;
    file >> jsonConfig;
    if (jsonConfig.contains("conf")) config["conf"] = std::to_string(jsonConfig["conf"].get<double>());
    if (jsonConfig.contains("iou")) config["iou"] = std::to_string(jsonConfig["iou"].get<double>());
}

bool loadImages(const std::string& imagesPath, std::vector<std::string>& imageFiles) {
    if (!fs::exists(imagesPath) || !fs::is_directory(imagesPath)) {
        std::cerr << "Error: Images path does not exist: " << imagesPath << std::endl;
        return false;
    }
    std::vector<std::string> validExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
    for (const auto& entry : fs::directory_iterator(imagesPath)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(validExtensions.begin(), validExtensions.end(), ext) != validExtensions.end()) {
                imageFiles.push_back(entry.path().string());
            }
        }
    }
    return !imageFiles.empty();
}

void runInference(const std::string& modelPath, const std::string& labelsPath, 
                  const std::vector<std::string>& imageFiles, 
                  const std::unordered_map<std::string, std::string>& inferenceConfig, bool isGPU,
                  std::unordered_map<std::string, std::vector<SingleOBBInferenceResult>>& inferenceResults) {
    
    std::cout << "Model: " << modelPath << std::endl;
    float confThreshold = std::stof(inferenceConfig.at("conf"));
    float iouThreshold = std::stof(inferenceConfig.at("iou"));

    YOLOOBBDetector detector(modelPath, labelsPath, isGPU);

    for (const auto& imagePath : imageFiles) {
        std::cout << "\nProcessing: " << imagePath << std::endl;

        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) continue;

        int image_width = image.cols;
        int image_height = image.rows;
        std::cout << "Image size: " << image_width << "x" << image_height << std::endl;

        inferenceResults[imagePath] = std::vector<SingleOBBInferenceResult>();

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<OBBResult> detResults = detector.detect(image, confThreshold, iouThreshold);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);

        std::cout << "OBB time: " << duration.count() << " ms, Found: " << detResults.size() << std::endl;

        for (size_t i = 0; i < detResults.size(); ++i) {
            SingleOBBInferenceResult singleResult;
            singleResult.classId = detResults[i].classId;
            singleResult.conf = detResults[i].conf;
            singleResult.cx = detResults[i].box.x;  // center x
            singleResult.cy = detResults[i].box.y;  // center y
            singleResult.width = detResults[i].box.width;
            singleResult.height = detResults[i].box.height;
            singleResult.angle = detResults[i].box.angle;
            singleResult.normalized_cx = detResults[i].box.x / static_cast<float>(image_width);
            singleResult.normalized_cy = detResults[i].box.y / static_cast<float>(image_height);
            singleResult.normalized_width = detResults[i].box.width / static_cast<float>(image_width);
            singleResult.normalized_height = detResults[i].box.height / static_cast<float>(image_height);

            inferenceResults[imagePath].push_back(singleResult);
        }
    }
}

void toJson(const std::unordered_map<std::string, Results>& results, 
            const std::string& basePath, json& outputJson) {
    for (const auto& [modelName, result] : results) {
        outputJson[modelName] = json();
        outputJson[modelName]["weights_path"] = result.weightsPath.substr(basePath.length());
        outputJson[modelName]["task"] = result.task;
        outputJson[modelName]["results"] = json::array();

        for (const auto& [imagePath, inferenceResults] : result.inferenceResults) {
            json imageResults;
            imageResults["image_path"] = imagePath.substr(basePath.length());
            imageResults["inference_results"] = json::array();

            for (const auto& res : inferenceResults) {
                json singleResult;
                singleResult["class_id"] = res.classId;
                singleResult["confidence"] = res.conf;
                singleResult["obb"] = {
                    {"cx", res.cx}, {"cy", res.cy}, 
                    {"width", res.width}, {"height", res.height}, 
                    {"angle", res.angle}
                };
                imageResults["inference_results"].push_back(singleResult);
            }
            outputJson[modelName]["results"].push_back(imageResults);
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== YOLOs-CPP OBB Test ===" << std::endl;
    bool isGPU = argc > 1 && std::string(argv[1]) == "gpu";

    std::string basePath = XSTRING(BASE_PATH_OBB);
    std::string imagesPath = basePath + "data/images/";
    std::string weightsPath = basePath + "models/";
    std::string labelsPath = weightsPath + "Dota.names";
    std::string resultsPath = basePath + "results/";

    std::unordered_map<std::string, std::string> paths_map = {
        {"images", imagesPath}, {"weights", weightsPath}, {"labels", labelsPath}, {"results", resultsPath}
    };

    if (!validatePaths(paths_map)) return -1;

    std::vector<std::string> imageFiles;
    if (!loadImages(imagesPath, imageFiles)) return -1;

    std::vector<std::string> models = {"yolov8n-obb", "yolo11n-obb", "yolo26n-obb"};
    std::unordered_map<std::string, std::string> inferenceConfig = {{"conf", "0.50"}, {"iou", "0.45"}};
    loadInferenceConfig(basePath + "inference_config_obb.json", inferenceConfig);

    std::string resultsFilePath = resultsPath + "results_cpp.json";
    if (fs::exists(resultsFilePath)) fs::remove(resultsFilePath);

    std::unordered_map<std::string, Results> allResults;

    for (const auto& model : models) {
        std::string modelPath = weightsPath + model + ".onnx";
        if (!fs::exists(modelPath)) continue;

        allResults[model] = Results{modelPath, "obb", {}};
        std::cout << "\n======== Running: " << model << " ========" << std::endl;
        runInference(modelPath, labelsPath, imageFiles, inferenceConfig, isGPU, allResults[model].inferenceResults);
    }

    json outputJson;
    toJson(allResults, basePath, outputJson);
    std::ofstream file(resultsFilePath);
    file << std::setw(2) << outputJson << std::endl;

    std::cout << "Results saved to: " << resultsFilePath << std::endl;
    return 0;
}
