/**
 * @file inference_pose_cpp.cpp
 * @brief Pose estimation inference test for YOLOs-CPP
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
#include "yolos/tasks/pose.hpp"

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace yolos::pose;

struct SingleInferenceResult {
    int classId;
    float conf;
    int left, top, width, height;
    std::vector<yolos::KeyPoint> keypoints;
};

struct Results {
    std::string weightsPath;
    std::string task;
    std::unordered_map<std::string, std::vector<SingleInferenceResult>> inferenceResults;
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
                  std::unordered_map<std::string, std::vector<SingleInferenceResult>>& inferenceResults) {
    
    std::cout << "Model: " << modelPath << std::endl;
    float confThreshold = std::stof(inferenceConfig.at("conf"));
    float iouThreshold = std::stof(inferenceConfig.at("iou"));

    YOLOPoseDetector detector(modelPath, labelsPath, isGPU);

    for (const auto& imagePath : imageFiles) {
        std::cout << "\nProcessing: " << imagePath << std::endl;

        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) continue;

        std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
        inferenceResults[imagePath] = std::vector<SingleInferenceResult>();

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<PoseResult> detResults = detector.detect(image, confThreshold, iouThreshold);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);

        std::cout << "Pose time: " << duration.count() << " ms, Found: " << detResults.size() << std::endl;

        for (const auto& res : detResults) {
            SingleInferenceResult singleResult;
            singleResult.classId = res.classId;
            singleResult.conf = res.conf;
            singleResult.left = static_cast<int>(res.box.x);
            singleResult.top = static_cast<int>(res.box.y);
            singleResult.width = static_cast<int>(res.box.width);
            singleResult.height = static_cast<int>(res.box.height);
            singleResult.keypoints = res.keypoints;
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
                singleResult["bbox"] = {{"left", res.left}, {"top", res.top}, {"width", res.width}, {"height", res.height}};
                singleResult["keypoints"] = json::array();
                for (const auto& kp : res.keypoints) {
                    singleResult["keypoints"].push_back({{"x", kp.x}, {"y", kp.y}, {"confidence", kp.confidence}});
                }
                imageResults["inference_results"].push_back(singleResult);
            }
            outputJson[modelName]["results"].push_back(imageResults);
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== YOLOs-CPP Pose Test ===" << std::endl;
    bool isGPU = argc > 1 && std::string(argv[1]) == "gpu";

    std::string basePath = XSTRING(BASE_PATH_POSE);
    std::string imagesPath = basePath + "data/images/";
    std::string weightsPath = basePath + "models/";
    std::string labelsPath = weightsPath + "coco.names";
    std::string resultsPath = basePath + "results/";

    std::unordered_map<std::string, std::string> paths_map = {
        {"images", imagesPath}, {"weights", weightsPath}, {"labels", labelsPath}, {"results", resultsPath}
    };

    if (!validatePaths(paths_map)) return -1;

    std::vector<std::string> imageFiles;
    if (!loadImages(imagesPath, imageFiles)) return -1;

    std::vector<std::string> models = {"yolov8n-pose", "yolo11n-pose", "yolo26n-pose"};
    std::unordered_map<std::string, std::string> inferenceConfig = {{"conf", "0.50"}, {"iou", "0.50"}};
    loadInferenceConfig(basePath + "inference_config.json", inferenceConfig);

    std::string resultsFilePath = resultsPath + "results_cpp.json";
    if (fs::exists(resultsFilePath)) fs::remove(resultsFilePath);

    std::unordered_map<std::string, Results> allResults;

    for (const auto& model : models) {
        std::string modelPath = weightsPath + model + ".onnx";
        if (!fs::exists(modelPath)) continue;

        allResults[model] = Results{modelPath, "pose", {}};
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
