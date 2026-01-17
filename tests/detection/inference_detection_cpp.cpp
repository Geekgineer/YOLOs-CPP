/**
 * @file inference_detection_cpp.cpp
 * @brief Detection inference test for YOLOs-CPP
 * @details Runs detection inference using the new yolos architecture
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
#include "yolos/tasks/detection.hpp"

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace yolos::det;

struct SingleInferenceResult {
    int classId;
    float conf;
    int left;
    int top;
    int width;
    int height;
    float normalized_left;
    float normalized_top;
    float normalized_width;
    float normalized_height;
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
    if (!fs::exists(configFilePath)) {
        std::cerr << "Warning: Inference config file does not exist. Using default values." << std::endl;
        return;
    }
    std::ifstream file(configFilePath);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open inference config file. Using default values." << std::endl;
        return;
    }
    json jsonConfig;
    file >> jsonConfig;
    if (jsonConfig.contains("conf")) {
        config["conf"] = std::to_string(jsonConfig["conf"].get<double>());
        std::cout << "Loaded confidence threshold: " << config["conf"] << std::endl;
    }
    if (jsonConfig.contains("iou")) {
        config["iou"] = std::to_string(jsonConfig["iou"].get<double>());
        std::cout << "Loaded IoU threshold: " << config["iou"] << std::endl;
    }
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
                std::cout << "Found image: " << entry.path().string() << std::endl;
            }
        }
    }
    if (imageFiles.empty()) {
        std::cerr << "Error: No valid image files found in: " << imagesPath << std::endl;
        return false;
    }
    std::cout << "Found " << imageFiles.size() << " image(s)" << std::endl;
    return true;
}

void runInference(const std::string& modelPath, const std::string& labelsPath, 
                  const std::vector<std::string>& imageFiles, 
                  const std::unordered_map<std::string, std::string>& inferenceConfig, 
                  bool isGPU,
                  std::unordered_map<std::string, std::vector<SingleInferenceResult>>& inferenceResults) {
    
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Labels: " << labelsPath << std::endl;
    std::cout << "Device: " << (isGPU ? "GPU" : "CPU") << std::endl;

    float confThreshold = std::stof(inferenceConfig.at("conf"));
    float iouThreshold = std::stof(inferenceConfig.at("iou"));

    // Create detector using new architecture
    YOLODetector detector(modelPath, labelsPath, isGPU);

    for (const auto& imagePath : imageFiles) {
        std::cout << "\nProcessing: " << imagePath << std::endl;

        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Error: Could not load image: " << imagePath << std::endl;
            continue;
        }

        int image_width = image.cols;
        int image_height = image.rows;
        std::cout << "Image size: " << image_width << "x" << image_height << std::endl;

        inferenceResults[imagePath] = std::vector<SingleInferenceResult>();

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Detection> results = detector.detect(image, confThreshold, iouThreshold);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);

        std::cout << "Detection time: " << duration.count() << " ms" << std::endl;
        std::cout << "Detections found: " << results.size() << std::endl;

        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "  [" << i << "] Class=" << results[i].classId 
                      << ", Conf=" << results[i].conf 
                      << ", Box=(" << results[i].box.x << "," << results[i].box.y 
                      << "," << results[i].box.width << "," << results[i].box.height << ")" << std::endl;

            SingleInferenceResult singleResult;
            singleResult.classId = results[i].classId;
            singleResult.conf = results[i].conf;
            singleResult.left = static_cast<int>(results[i].box.x);
            singleResult.top = static_cast<int>(results[i].box.y);
            singleResult.width = static_cast<int>(results[i].box.width);
            singleResult.height = static_cast<int>(results[i].box.height);
            singleResult.normalized_left = results[i].box.x / static_cast<float>(image_width);
            singleResult.normalized_top = results[i].box.y / static_cast<float>(image_height);
            singleResult.normalized_width = results[i].box.width / static_cast<float>(image_width);
            singleResult.normalized_height = results[i].box.height / static_cast<float>(image_height);

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
                singleResult["bbox"] = {
                    {"left", res.left},
                    {"top", res.top},
                    {"width", res.width},
                    {"height", res.height}
                };
                imageResults["inference_results"].push_back(singleResult);
            }
            outputJson[modelName]["results"].push_back(imageResults);
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== YOLOs-CPP Detection Test ===" << std::endl;
    std::cout << "Usage: ./inference_detection_cpp [cpu|gpu]" << std::endl;

    bool isGPU = argc > 1 && std::string(argv[1]) == "gpu";

    std::string basePath = XSTRING(BASE_PATH_DETECTION);
    std::string dataPath = basePath + "data/";
    std::string imagesPath = dataPath + "images/";
    std::string weightsPath = basePath + "models/";
    std::string labelsPath = weightsPath + "voc.names";
    std::string resultsPath = basePath + "results/";

    std::unordered_map<std::string, std::string> paths_map = {
        {"data", dataPath},
        {"images", imagesPath},
        {"weights", weightsPath},
        {"labels", labelsPath},
        {"results", resultsPath}
    };

    if (!validatePaths(paths_map)) {
        return -1;
    }

    std::vector<std::string> imageFiles;
    if (!loadImages(imagesPath, imageFiles)) {
        return -1;
    }

    std::vector<std::string> models = {
        "YOLOv5nu_voc",
        "YOLOv6n_voc",
        "YOLOv8n_voc",
        "YOLOv9t_voc",
        "YOLOv10n_voc",
        "YOLOv11n_voc",
        "YOLOv12n_voc",
        "YOLO26n_voc"
    };

    std::unordered_map<std::string, std::string> inferenceConfig = {
        {"conf", "0.50"},
        {"iou", "0.50"}
    };

    loadInferenceConfig(basePath + "inference_config.json", inferenceConfig);

    std::string resultsFilePath = resultsPath + "results_cpp.json";
    if (fs::exists(resultsFilePath)) {
        fs::remove(resultsFilePath);
    }

    std::unordered_map<std::string, Results> allResults;

    for (const auto& model : models) {
        std::string modelPath = weightsPath + model + ".onnx";
        if (!fs::exists(modelPath)) {
            std::cerr << "Warning: Model not found, skipping: " << modelPath << std::endl;
            continue;
        }

        allResults[model] = Results();
        allResults[model].weightsPath = modelPath;
        allResults[model].task = "detect";

        std::cout << "\n======== Running: " << model << " ========" << std::endl;
        runInference(modelPath, labelsPath, imageFiles, inferenceConfig, isGPU, allResults[model].inferenceResults);
        std::cout << "======== Completed: " << model << " ========\n" << std::endl;
    }

    json outputJson;
    toJson(allResults, basePath, outputJson);
    std::ofstream file(resultsFilePath);
    file << std::setw(2) << outputJson << std::endl;

    std::cout << "Results saved to: " << resultsFilePath << std::endl;
    return 0;
}
