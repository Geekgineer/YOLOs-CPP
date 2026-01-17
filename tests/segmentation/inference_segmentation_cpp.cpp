/**
 * @file inference_segmentation_cpp.cpp
 * @brief Segmentation inference test for YOLOs-CPP
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
#include "yolos/tasks/segmentation.hpp"

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace yolos::seg;

struct SingleInferenceResult {
    int classId;
    float conf;
    int left, top, width, height;
    float normalized_left, normalized_top, normalized_width, normalized_height;
};

struct Results {
    std::string weightsPath;
    std::string task;
    std::unordered_map<std::string, std::vector<SingleInferenceResult>> inferenceResults;
    std::unordered_map<std::string, std::string> inferenceMasks;
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

void runInference(const std::string& modelPath, const std::string& labelsPath, const std::string& masksPath,
                  const std::vector<std::string>& imageFiles, 
                  const std::unordered_map<std::string, std::string>& inferenceConfig, bool isGPU,
                  std::unordered_map<std::string, std::vector<SingleInferenceResult>>& inferenceResults,
                  std::unordered_map<std::string, std::string>& inferenceMasks) {
    
    std::cout << "Model: " << modelPath << std::endl;
    float confThreshold = std::stof(inferenceConfig.at("conf"));
    float iouThreshold = std::stof(inferenceConfig.at("iou"));

    std::string model_name = fs::path(modelPath).stem().string();
    YOLOSegDetector detector(modelPath, labelsPath, isGPU);

    for (const auto& imagePath : imageFiles) {
        std::cout << "\nProcessing: " << imagePath << std::endl;

        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) continue;

        std::string image_name = fs::path(imagePath).stem().string();
        int image_width = image.cols;
        int image_height = image.rows;

        inferenceResults[imagePath] = std::vector<SingleInferenceResult>();

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Segmentation> results = detector.segment(image, confThreshold, iouThreshold);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);

        std::cout << "Segmentation time: " << duration.count() << " ms, Found: " << results.size() << std::endl;

        cv::Mat mask_full = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);

        for (size_t i = 0; i < results.size(); ++i) {
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

            // Merge mask
            cv::Mat mask = results[i].mask;
            for (int row = 0; row < mask.rows; ++row) {
                for (int col = 0; col < mask.cols; ++col) {
                    if (mask.at<uchar>(row, col) == 255) {
                        mask_full.at<uchar>(row, col) = static_cast<uchar>(results[i].classId);
                    }
                }
            }
            inferenceResults[imagePath].push_back(singleResult);
        }

        std::string mask_image_path = masksPath + "/" + model_name + "_" + image_name + "_mask.png";
        cv::imwrite(mask_image_path, mask_full);
        inferenceMasks[imagePath] = mask_image_path;
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
            imageResults["mask_path"] = result.inferenceMasks.at(imagePath).substr(basePath.length());
            imageResults["inference_results"] = json::array();

            for (const auto& res : inferenceResults) {
                json singleResult;
                singleResult["class_id"] = res.classId;
                singleResult["confidence"] = res.conf;
                singleResult["bbox"] = {{"left", res.left}, {"top", res.top}, {"width", res.width}, {"height", res.height}};
                imageResults["inference_results"].push_back(singleResult);
            }
            outputJson[modelName]["results"].push_back(imageResults);
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== YOLOs-CPP Segmentation Test ===" << std::endl;
    bool isGPU = argc > 1 && std::string(argv[1]) == "gpu";

    std::string basePath = XSTRING(BASE_PATH_SEGMENTATION);
    std::string imagesPath = basePath + "data/images/";
    std::string weightsPath = basePath + "models/";
    std::string labelsPath = weightsPath + "coco.names";
    std::string resultsPath = basePath + "results/";
    std::string masksPath = resultsPath + "masks/cpp/";

    std::unordered_map<std::string, std::string> paths_map = {
        {"images", imagesPath}, {"weights", weightsPath}, {"labels", labelsPath}, {"results", resultsPath}
    };

    if (!validatePaths(paths_map)) return -1;

    std::vector<std::string> imageFiles;
    if (!loadImages(imagesPath, imageFiles)) return -1;

    std::vector<std::string> models = {"yolov8s-seg", "yolov9c-seg", "yolo11s-seg", "yolo26n-seg"};
    std::unordered_map<std::string, std::string> inferenceConfig = {{"conf", "0.50"}, {"iou", "0.50"}};
    loadInferenceConfig(basePath + "inference_config.json", inferenceConfig);

    if (fs::exists(masksPath)) fs::remove_all(masksPath);
    fs::create_directories(masksPath);

    std::unordered_map<std::string, Results> allResults;

    for (const auto& model : models) {
        std::string modelPath = weightsPath + model + ".onnx";
        if (!fs::exists(modelPath)) continue;

        allResults[model] = Results{modelPath, "segment", {}, {}};
        std::cout << "\n======== Running: " << model << " ========" << std::endl;
        runInference(modelPath, labelsPath, masksPath, imageFiles, inferenceConfig, isGPU,
                     allResults[model].inferenceResults, allResults[model].inferenceMasks);
    }

    json outputJson;
    toJson(allResults, basePath, outputJson);
    std::string resultsFilePath = resultsPath + "results_cpp.json";
    std::ofstream file(resultsFilePath);
    file << std::setw(2) << outputJson << std::endl;

    std::cout << "Results saved to: " << resultsFilePath << std::endl;
    return 0;
}
