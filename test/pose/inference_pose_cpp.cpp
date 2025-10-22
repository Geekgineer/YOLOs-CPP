// Include necessary headers
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <unordered_map>
#include <nlohmann/json.hpp>

#include "pose/YOLO-POSE.hpp"  // Pose header

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace fs = std::filesystem;
using json = nlohmann::json;

struct Keypoint {
    float x, y, confidence;
};

struct SingleInferenceResult {
    int classId;
    float conf;
    std::vector<Keypoint> keypoints;
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
    nlohmann::json jsonConfig;
    file >> jsonConfig;
    if (jsonConfig.contains("conf")) {
        config["conf"] = std::to_string(jsonConfig["conf"].get<double>());
        std::cout << "Loaded confidence threshold from " << configFilePath << ": " << config["conf"] << std::endl;
    }
    if (jsonConfig.contains("iou")) {
        config["iou"] = std::to_string(jsonConfig["iou"].get<double>());
        std::cout << "Loaded IoU threshold from " << configFilePath << ": " << config["iou"] << std::endl;
    }
}

bool loadImages(const std::string& imagesPath, std::vector<std::string>& imageFiles) {
    if (!fs::exists(imagesPath) || !fs::is_directory(imagesPath)) {
        std::cerr << "Error: Images path does not exist or is not a directory: " << imagesPath << std::endl;
        return false;
    }
    std::vector<std::string> validExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
    for (const auto& entry : fs::directory_iterator(imagesPath)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(validExtensions.begin(), validExtensions.end(), ext) != validExtensions.end()) {
                imageFiles.push_back(entry.path().string());
                std::cout << "Found image file: " << entry.path().string() << std::endl;
            }
        }
    }
    if (imageFiles.empty()) {
        std::cerr << "Error: No valid image files found in directory: " << imagesPath << std::endl;
        return false;
    }
    std::cout << "Found " << imageFiles.size() << " image(s) in directory: " << imagesPath << std::endl;
    return true;
}

bool validateModelFiles(const std::vector<std::string>& modelFiles) {
    for (const auto& modelPath : modelFiles) {
        if (!fs::exists(modelPath)) {
            std::cerr << "Error: Model file does not exist: " << modelPath << std::endl;
            return false;
        }
    }
    std::cout << "All model files are valid." << std::endl;
    return true;
}

void runInference(const std::string& modelPath, const std::string& labelsPath, const std::vector<std::string>& imageFiles, 
                  const std::unordered_map<std::string, std::string>& inferenceConfig, bool isGPU,
                  std::unordered_map<std::string, std::vector<SingleInferenceResult>>& inferenceResults) {
    std::cout << "Using model: " << modelPath << std::endl;
    std::cout << "Using labels: " << labelsPath << std::endl;
    std::cout << "Using inference config: " << std::endl;
    for (const auto& [key, value] : inferenceConfig) {
        std::cout << "  " << key << ": " << value << std::endl;
    }
    std::cout << "Using device: " << (isGPU ? "GPU" : "CPU") << std::endl;

    float confThreshold = std::stof(inferenceConfig.at("conf"));
    float iouThreshold = std::stof(inferenceConfig.at("iou"));

    YOLOPOSEDetector detector(modelPath, labelsPath, isGPU);

    for (const auto& imagePath : imageFiles) {
        std::cout << "\nProcessing: " << imagePath << std::endl;
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Error: Could not open or find the image!\n";
            continue;
        }
        std::cout << "Image loaded: " << imagePath << " (Size: " << image.cols << "x" << image.rows << ")" << std::endl;
        inferenceResults[imagePath] = std::vector<SingleInferenceResult>();

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Pose> results = detector.detect(image, confThreshold, iouThreshold);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - start);
        std::cout << "Detection completed in: " << duration.count() << " ms" << std::endl;
        std::cout << "Number of poses found: " << results.size() << std::endl;

        std::vector<SingleInferenceResult> poses;
        for (const auto& result : results) {
            std::cout << "Pose: Class=" << result.classId << ", Confidence=" << result.conf << ", Keypoints=" << result.keypoints.size() << std::endl;
            SingleInferenceResult singleResult;
            singleResult.classId = result.classId;
            singleResult.conf = result.conf;
            singleResult.keypoints = result.keypoints;
            inferenceResults[imagePath].push_back(singleResult);
        }
    }
}

void fromMapToJson(const std::unordered_map<std::string, Results>& results, std::string basePath, nlohmann::json& outputJson) {
    for (const auto& [modelName, result] : results) {
        outputJson[modelName] = nlohmann::json();
        outputJson[modelName]["weights_path"] = result.weightsPath.substr(basePath.length());
        outputJson[modelName]["task"] = result.task;
        outputJson[modelName]["results"] = nlohmann::json::array();

        for (const auto& [imagePath, inferenceResults] : result.inferenceResults) {
            nlohmann::json imageResults;
            imageResults["image_path"] = imagePath.substr(basePath.length());
            imageResults["inference_results"] = nlohmann::json::array();
            for (const auto& res : inferenceResults) {
                nlohmann::json singleResult;
                singleResult["class_id"] = res.classId;
                singleResult["confidence"] = res.conf;
                singleResult["keypoints"] = nlohmann::json::array();
                for (const auto& kp : res.keypoints) {
                    nlohmann::json keypoint;
                    keypoint["x"] = kp.x;
                    keypoint["y"] = kp.y;
                    keypoint["confidence"] = kp.confidence;
                    singleResult["keypoints"].push_back(keypoint);
                }
                imageResults["inference_results"].push_back(singleResult);
            }
            outputJson[modelName]["results"].push_back(imageResults);
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Usage: ./inference_pose_cpp [cpu|gpu]" << std::endl;
    bool isGPU = argc > 1 ? std::string(argv[1]) == "gpu" : false;
    std::string basePath = XSTRING(BASE_PATH_POSE);
    std::string dataPath = basePath + "data/";
    std::string imagesPath = dataPath + "images/";
    std::string weightsPath = basePath + "models/";
    std::string labelsPath = weightsPath + "coco.names";
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
    std::vector<std::string> models = {"yolov8n-pose"};
    std::unordered_map<std::string, std::string> inferenceConfig = {
        {"conf", "0.50"},
        {"iou", "0.50"}
    };
    std::string inferenceConfigFilePath = basePath + "inference_config.json";
    loadInferenceConfig(inferenceConfigFilePath, inferenceConfig);
    std::string resultsFilePath = resultsPath + "results_cpp.json";
    if (fs::exists(resultsFilePath)) {
        fs::remove(resultsFilePath);
    }
    std::unordered_map<std::string, Results> allResults;
    for (const auto& model : models) {
        std::string modelPath = weightsPath + model + ".onnx";
        if (!fs::exists(modelPath)) {
            std::cerr << "Warning: Model file does not exist, skipping: " << modelPath << std::endl;
            continue;
        }
        allResults[model] = Results();
        allResults[model].weightsPath = modelPath;
        allResults[model].task = "pose";
        std::cout << "\n ######## Running inference for model: " << modelPath << " ########" << std::endl;
        runInference(modelPath, labelsPath, imageFiles, inferenceConfig, isGPU, allResults[model].inferenceResults);
        std::cout << " ######## Finished inference for model: " << modelPath << " ########\n" << std::endl;
    }
    nlohmann::json outputJson;
    fromMapToJson(allResults, basePath, outputJson);
    std::ofstream file(resultsFilePath);
    file << std::setw(2) << outputJson << std::endl;
    return 0;
}