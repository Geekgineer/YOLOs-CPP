/**
 * @file inference_yoloe_cpp.cpp
 * @brief YOLOE segmentation parity test (Ultralytics reference vs YOLOs-CPP)
 */

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "yolos/tasks/yoloe.hpp"

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace yolos::yoloe;
using yolos::seg::Segmentation;

struct SingleInferenceResult {
    int classId;
    float conf;
    int left, top, width, height;
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

void loadYoloeInferenceConfig(const std::string& configFilePath,
                              std::unordered_map<std::string, std::string>& config,
                              std::vector<std::string>& classNames) {
    if (!fs::exists(configFilePath)) return;
    std::ifstream file(configFilePath);
    if (!file.is_open()) return;
    json j;
    file >> j;
    if (j.contains("conf")) config["conf"] = std::to_string(j["conf"].get<double>());
    if (j.contains("iou")) config["iou"] = std::to_string(j["iou"].get<double>());
    if (j.contains("classes") && j["classes"].is_array()) {
        for (const auto& c : j["classes"]) {
            classNames.push_back(c.get<std::string>());
        }
    }
}

bool loadImages(const std::string& imagesPath, std::vector<std::string>& imageFiles) {
    if (!fs::exists(imagesPath) || !fs::is_directory(imagesPath)) {
        std::cerr << "Error: Images path does not exist: " << imagesPath << std::endl;
        return false;
    }
    const std::vector<std::string> validExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
    for (const auto& entry : fs::directory_iterator(imagesPath)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(validExtensions.begin(), validExtensions.end(), ext) != validExtensions.end()) {
                imageFiles.push_back(entry.path().string());
            }
        }
    }
    std::sort(imageFiles.begin(), imageFiles.end());
    return !imageFiles.empty();
}

void runInference(const std::string& modelPath,
                  const std::vector<std::string>& classNames,
                  const std::string& masksPath,
                  const std::vector<std::string>& imageFiles,
                  const std::unordered_map<std::string, std::string>& inferenceConfig,
                  bool isGPU,
                  std::unordered_map<std::string, std::vector<SingleInferenceResult>>& inferenceResults,
                  std::unordered_map<std::string, std::string>& inferenceMasks) {
    std::cout << "Model: " << modelPath << std::endl;
    float confThreshold = std::stof(inferenceConfig.at("conf"));
    float iouThreshold = std::stof(inferenceConfig.at("iou"));

    const std::string model_name = fs::path(modelPath).stem().string();
    YOLOESegDetector detector(modelPath, classNames, isGPU);

    for (const auto& imagePath : imageFiles) {
        std::cout << "\nProcessing: " << imagePath << std::endl;

        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) continue;

        const std::string image_name = fs::path(imagePath).stem().string();

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

        const std::string mask_image_path = masksPath + "/" + model_name + "_" + image_name + "_mask.png";
        cv::imwrite(mask_image_path, mask_full);
        inferenceMasks[imagePath] = mask_image_path;
    }
}

void toJson(const std::unordered_map<std::string, Results>& results, const std::string& basePath, json& outputJson) {
    for (const auto& [modelName, result] : results) {
        outputJson[modelName] = json();
        outputJson[modelName]["weights_path"] = result.weightsPath.substr(basePath.length());
        outputJson[modelName]["task"] = result.task;
        outputJson[modelName]["results"] = json::array();

        for (const auto& [imagePath, inferenceRes] : result.inferenceResults) {
            json imageResults;
            imageResults["image_path"] = imagePath.substr(basePath.length());
            imageResults["mask_path"] = result.inferenceMasks.at(imagePath).substr(basePath.length());
            imageResults["inference_results"] = json::array();

            for (const auto& res : inferenceRes) {
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
    std::cout << "=== YOLOs-CPP YOLOE Segmentation Parity Test ===" << std::endl;
    bool isGPU = argc > 1 && std::string(argv[1]) == "gpu";

    std::string basePath = XSTRING(BASE_PATH_YOLOE);
    std::string imagesPath = basePath + "data/images/";
    std::string weightsPath = basePath + "models/";
    std::string resultsPath = basePath + "results/";
    std::string masksPath = resultsPath + "masks/cpp/";

    std::unordered_map<std::string, std::string> inferenceConfig = {{"conf", "0.50"}, {"iou", "0.50"}};
    std::vector<std::string> classNames;
    loadYoloeInferenceConfig(basePath + "inference_config.json", inferenceConfig, classNames);
    if (classNames.empty()) {
        std::cerr << "Error: inference_config.json must define non-empty 'classes' for YOLOE." << std::endl;
        return -1;
    }

    std::unordered_map<std::string, std::string> paths_map = {
        {"images", imagesPath}, {"weights", weightsPath}, {"results", resultsPath}};

    if (!validatePaths(paths_map)) return -1;

    std::vector<std::string> imageFiles;
    if (!loadImages(imagesPath, imageFiles)) return -1;

    if (fs::exists(masksPath)) fs::remove_all(masksPath);
    fs::create_directories(masksPath);

    std::unordered_map<std::string, Results> allResults;
    const std::vector<std::string> models = {"yoloe-26n-seg"};

    for (const auto& model : models) {
        std::string modelPath = weightsPath + model + ".onnx";
        if (!fs::exists(modelPath)) {
            std::cout << "Skipping missing model: " << modelPath << std::endl;
            continue;
        }

        allResults[model] = Results{modelPath, "segment", {}, {}};
        std::cout << "\n======== Running: " << model << " ========" << std::endl;
        runInference(modelPath, classNames, masksPath, imageFiles, inferenceConfig, isGPU,
                     allResults[model].inferenceResults, allResults[model].inferenceMasks);
    }

    if (allResults.empty()) {
        std::cerr << "Error: no YOLOE ONNX models found under " << weightsPath << std::endl;
        return -1;
    }

    json outputJson;
    toJson(allResults, basePath, outputJson);
    std::string resultsFilePath = resultsPath + "results_cpp.json";
    std::ofstream file(resultsFilePath);
    file << std::setw(2) << outputJson << std::endl;

    std::cout << "Results saved to: " << resultsFilePath << std::endl;
    return 0;
}
