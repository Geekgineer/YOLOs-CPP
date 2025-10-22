// Include necessary headers
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <unordered_map>
#include <nlohmann/json.hpp>

#include "class/YOLOCLASS.hpp"

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace fs = std::filesystem;
using json = nlohmann::json;

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

void loadImages(const std::string& imagesPath, std::vector<std::string>& imageFiles) {
    if (!fs::exists(imagesPath) || !fs::is_directory(imagesPath)) {
        std::cerr << "Error: Images path does not exist or is not a directory: " << imagesPath << std::endl;
        return;
    }
    std::vector<std::string> validExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
    for (const auto& entry : fs::directory_iterator(imagesPath)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(validExtensions.begin(), validExtensions.end(), ext) != validExtensions.end()) {
            imageFiles.push_back(entry.path().string());
            std::cout << "Found image file: " << entry.path().string() << std::endl;
        }
    }
    if (imageFiles.empty()) {
        std::cerr << "Error: No valid image files found in directory: " << imagesPath << std::endl;
    } else {
        std::cout << "Found " << imageFiles.size() << " image(s) in directory: " << imagesPath << std::endl;
    }
}

void findModels(const std::string& modelsDir, std::vector<std::string>& modelFiles) {
    if (!fs::exists(modelsDir) || !fs::is_directory(modelsDir)) {
        std::cerr << "Error: Models path does not exist or is not a directory: " << modelsDir << std::endl;
        return;
    }
    for (const auto& entry : fs::directory_iterator(modelsDir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".onnx") {
            modelFiles.push_back(entry.path().string());
            std::cout << "Found model: " << entry.path().string() << std::endl;
        }
    }
    if (modelFiles.empty()) {
        std::cerr << "Warning: No .onnx models found in directory: " << modelsDir << std::endl;
    }
}

void runInference(const std::string& modelPath,
                  const std::string& labelsPath,
                  const std::vector<std::string>& imageFiles,
                  bool isGPU,
                  std::unordered_map<std::string, std::vector<SingleInferenceResultCls>>& inferenceResults) {

    std::cout << "Using model: " << modelPath << std::endl;
    std::cout << "Using labels: " << labelsPath << std::endl;
    std::cout << "Using device: " << (isGPU ? "GPU" : "CPU") << std::endl;

    YOLOClassifier classifier(modelPath, labelsPath, isGPU, YOLOClassVersion::V11);

    for (const auto& imagePath : imageFiles) {
        std::cout << "\nProcessing: " << imagePath << std::endl;
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Error: Could not open or find the image!\n";
            continue;
        }

        auto start = std::chrono::high_resolution_clock::now();
        ClassificationResult res = classifier.classify(image);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
        std::cout << "Classification completed in: " << duration.count() << " ms" << std::endl;

        SingleInferenceResultCls single;
        single.classId = res.classId;
        single.conf = res.confidence;
        single.className = res.className;

        inferenceResults[imagePath].push_back(single);
        std::cout << "Top-1: classId=" << single.classId << ", conf=" << single.conf
                  << ", className=\"" << single.className << "\"" << std::endl;
    }
}

void fromMapToJson(const std::unordered_map<std::string, ResultsCls>& results,
                   const std::string& basePath,
                   nlohmann::json& outputJson) {
    for (const auto& [modelName, resultsForModel] : results) {
        outputJson[modelName] = nlohmann::json();
        outputJson[modelName]["weights_path"] = resultsForModel.weightsPath.substr(basePath.length());
        outputJson[modelName]["task"] = resultsForModel.task;
        outputJson[modelName]["results"] = nlohmann::json::array();

        for (const auto& [imagePath, inferenceResVec] : resultsForModel.inferenceResults) {
            nlohmann::json imageResults;
            imageResults["image_path"] = imagePath.substr(basePath.length());
            imageResults["inference_results"] = nlohmann::json::array();

            for (const auto& res : inferenceResVec) {
                nlohmann::json single;
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
    std::cout << "Usage: ./inference_classification_cpp [cpu|gpu]" << std::endl;
    bool isGPU = argc > 1 ? std::string(argv[1]) == "gpu" : false;

    std::string basePath = XSTRING(BASE_PATH_CLASSIFICATION);
    std::string dataPath = basePath + "data/";
    std::string imagesPath = dataPath + "images/";
    std::string modelsPath = basePath + "models/";
    std::string labelsPath = modelsPath + "labels.names"; // expected labels file
    std::string resultsPath = basePath + "results/";

    std::unordered_map<std::string, std::string> paths_map = {
        {"data", dataPath},
        {"images", imagesPath},
        {"models", modelsPath},
        {"labels", labelsPath},
        {"results", resultsPath}
    };

    if (!validatePaths(paths_map)) return -1;

    std::vector<std::string> imageFiles;
    loadImages(imagesPath, imageFiles);
    if (imageFiles.empty()) return -1;

    std::vector<std::string> modelFiles;
    findModels(modelsPath, modelFiles);
    if (modelFiles.empty()) return 0; // nothing to do

    if (!fs::exists(resultsPath)) fs::create_directories(resultsPath);
    std::string resultsFilePath = resultsPath + "results_cpp.json";
    if (fs::exists(resultsFilePath)) fs::remove(resultsFilePath);

    std::unordered_map<std::string, ResultsCls> allResults;
    for (const auto& modelFullPath : modelFiles) {
        const std::string modelName = fs::path(modelFullPath).stem().string();
        allResults[modelName] = ResultsCls();
        allResults[modelName].weightsPath = modelFullPath;
        allResults[modelName].task = "classify";

        std::cout << "\n ######## Running inference for model: " << modelFullPath << " ########" << std::endl;
        runInference(modelFullPath, labelsPath, imageFiles, isGPU, allResults[modelName].inferenceResults);
        std::cout << " ######## Finished inference for model: " << modelFullPath << " ########\n" << std::endl;
    }

    nlohmann::json outputJson;
    fromMapToJson(allResults, basePath, outputJson);
    std::ofstream file(resultsFilePath);
    file << std::setw(2) << outputJson << std::endl;
    return 0;
}


