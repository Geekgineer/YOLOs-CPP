/**
 * @file inference_classification_cpp.cpp
 * @brief Classification inference test for YOLOs-CPP
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
#include "yolos/tasks/classification.hpp"

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace yolos::cls;

struct SingleInferenceResultCls {
    int classId;
    float conf;
    std::string className;
};

struct ResultsCls {
    std::string weightsPath;
    std::string task;
    std::unordered_map<std::string, std::vector<SingleInferenceResultCls>> inferenceResults;
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

void findModels(const std::string& modelsDir, std::vector<std::string>& modelFiles) {
    if (!fs::exists(modelsDir) || !fs::is_directory(modelsDir)) return;
    for (const auto& entry : fs::directory_iterator(modelsDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".onnx") {
            modelFiles.push_back(entry.path().string());
        }
    }
}

void runInference(const std::string& modelPath, const std::string& labelsPath,
                  const std::vector<std::string>& imageFiles, bool isGPU,
                  std::unordered_map<std::string, std::vector<SingleInferenceResultCls>>& inferenceResults) {
    
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Labels: " << labelsPath << std::endl;
    std::cout << "Device: " << (isGPU ? "GPU" : "CPU") << std::endl;

    // Use the new YOLOClassifier with auto version detection
    YOLOClassifier classifier(modelPath, labelsPath, isGPU);

    for (const auto& imagePath : imageFiles) {
        std::cout << "\nProcessing: " << imagePath << std::endl;

        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) continue;

        auto start = std::chrono::high_resolution_clock::now();
        ClassificationResult res = classifier.classify(image);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);

        std::cout << "Classification time: " << duration.count() << " ms" << std::endl;
        std::cout << "Top-1: classId=" << res.classId << ", conf=" << res.confidence
                  << ", className=\"" << res.className << "\"" << std::endl;

        SingleInferenceResultCls single;
        single.classId = res.classId;
        single.conf = res.confidence;
        single.className = res.className;

        inferenceResults[imagePath].push_back(single);
    }
}

void toJson(const std::unordered_map<std::string, ResultsCls>& results,
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
                json single;
                single["class_id"] = res.classId;
                single["confidence"] = res.conf;
                single["class_name"] = res.className;
                imageResults["inference_results"].push_back(single);
            }
            outputJson[modelName]["results"].push_back(imageResults);
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== YOLOs-CPP Classification Test ===" << std::endl;
    bool isGPU = argc > 1 && std::string(argv[1]) == "gpu";

    std::string basePath = XSTRING(BASE_PATH_CLASSIFICATION);
    std::string imagesPath = basePath + "data/images/";
    std::string modelsPath = basePath + "models/";
    std::string labelsPath = modelsPath + "labels.names";
    std::string resultsPath = basePath + "results/";

    std::unordered_map<std::string, std::string> paths_map = {
        {"images", imagesPath}, {"models", modelsPath}, {"labels", labelsPath}, {"results", resultsPath}
    };

    if (!validatePaths(paths_map)) return -1;

    std::vector<std::string> imageFiles;
    if (!loadImages(imagesPath, imageFiles)) return -1;

    std::vector<std::string> modelFiles;
    findModels(modelsPath, modelFiles);
    if (modelFiles.empty()) {
        std::cerr << "No models found in: " << modelsPath << std::endl;
        return 0;
    }

    if (!fs::exists(resultsPath)) fs::create_directories(resultsPath);
    std::string resultsFilePath = resultsPath + "results_cpp.json";
    if (fs::exists(resultsFilePath)) fs::remove(resultsFilePath);

    std::unordered_map<std::string, ResultsCls> allResults;

    for (const auto& modelPath : modelFiles) {
        std::string modelName = fs::path(modelPath).stem().string();
        allResults[modelName] = ResultsCls{modelPath, "classify", {}};
        
        std::cout << "\n======== Running: " << modelName << " ========" << std::endl;
        runInference(modelPath, labelsPath, imageFiles, isGPU, allResults[modelName].inferenceResults);
    }

    json outputJson;
    toJson(allResults, basePath, outputJson);
    std::ofstream file(resultsFilePath);
    file << std::setw(2) << outputJson << std::endl;

    std::cout << "Results saved to: " << resultsFilePath << std::endl;
    return 0;
}
