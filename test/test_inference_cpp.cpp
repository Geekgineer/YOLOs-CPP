
// Include necessary headers
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <algorithm> // Required for std::transform
#include <unordered_map>
#include <nlohmann/json.hpp>


#include "det/YOLO.hpp"

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace fs = std::filesystem;
using json = nlohmann::json;

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

            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower); // Convert to lowercase

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

    int image_width, image_height;

    YOLODetector detector(modelPath, labelsPath, isGPU);

    for (const auto& imagePath : imageFiles) {

        std::cout << "\nProcessing: " << imagePath << std::endl;

        // Load an image
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Error: Could not open or find the image!\n";
            continue;
        }

        image_width = image.cols;
        image_height = image.rows;

        std::cout << "Image loaded: " << imagePath << " (Size: " << image.cols << "x" << image.rows << ")" << std::endl;

        inferenceResults[imagePath] = std::vector<SingleInferenceResult>();

        // Detect objects in the image and measure execution time
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Detection> results = detector.detect(image, confThreshold, iouThreshold);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - start);

        std::cout << "Detection completed in: " << duration.count() << " ms" << std::endl;
        std::cout << "Number of detections found: " << results.size() << std::endl;

        std::vector<SingleInferenceResult> detections;

        float normalized_left, normalized_top, normalized_width, normalized_height;

        // Print details of each detection
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "Detection " << i << ": Class=" << results[i].classId 
                      << ", Confidence=" << results[i].conf 
                      << ", Box=(" << results[i].box.x << "," << results[i].box.y 
                      << "," << results[i].box.width << "," << results[i].box.height << ")" << std::endl;

            SingleInferenceResult singleResult;

            singleResult.classId = results[i].classId;

            singleResult.conf = results[i].conf;

            singleResult.normalized_left = results[i].box.x / static_cast<float>(image_width);
            singleResult.normalized_top = results[i].box.y / static_cast<float>(image_height);
            singleResult.normalized_width = results[i].box.width / static_cast<float>(image_width);
            singleResult.normalized_height = results[i].box.height / static_cast<float>(image_height);

            singleResult.left = results[i].box.x;
            singleResult.top = results[i].box.y;
            singleResult.width = results[i].box.width;
            singleResult.height = results[i].box.height;

            inferenceResults[imagePath].push_back(singleResult);
        }
    }
}


void fromMapToJson(const std::unordered_map<std::string, Results>& results, std::string basePath, nlohmann::json& outputJson) {

    for (const auto& [modelName, results] : results) {

        outputJson[modelName] = nlohmann::json();
        outputJson[modelName]["weights_path"] = results.weightsPath.substr(basePath.length());
        outputJson[modelName]["task"] = results.task;
        outputJson[modelName]["results"] = nlohmann::json::array();

        for (const auto& [imagePath, inferenceResults] : results.inferenceResults) {

            nlohmann::json imageResults;
            imageResults["image_path"] = imagePath.substr(basePath.length());
            imageResults["inference_results"] = nlohmann::json::array();

            for (const auto& res : inferenceResults) {
                nlohmann::json singleResult;
                singleResult["class_id"] = res.classId;
                singleResult["confidence"] = res.conf;

                singleResult["bbox"] = nlohmann::json();
                singleResult["bbox"]["left"] = res.left;
                singleResult["bbox"]["top"] = res.top;
                singleResult["bbox"]["width"] = res.width;
                singleResult["bbox"]["height"] = res.height;

                imageResults["inference_results"].push_back(singleResult);
            }
            outputJson[modelName]["results"].push_back(imageResults);
        }
    }
}

int main(int argc, char* argv[]){

    std::cout << "Usage: ./test_inference_cpp [cpu|gpu]" << std::endl;

    bool isGPU = argc > 1 ? std::string(argv[1]) == "gpu" : false; // Default to CPU if no argument is provided

    std::string basePath = XSTRING(BASE_PATH); // Base path defined in CMakeLists.txt

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
        return -1; // Exit if any path is invalid
    }

    std::vector<std::string> imageFiles;
    if(!loadImages(imagesPath, imageFiles)){
        return -1; // Exit if no images found
    }
    
    std::vector<std::string> models = {
        "YOLOv5nu_voc",
        "YOLOv6n_voc",
        "YOLOv8n_voc",
        "YOLOv9t_voc",
        "YOLOv10n_voc",
        "YOLOv11n_voc",
        "YOLOv12n_voc"
    };

    std::unordered_map<std::string, std::string> inferenceConfig = {
        {"conf", "0.50"},
        {"iou", "0.50"}
    };

    std::string inferenceConfigFilePath = basePath + "inference_config.json";

    loadInferenceConfig(inferenceConfigFilePath, inferenceConfig);

    std::string resultsFilePath = resultsPath + "results_cpp.json";

    if(fs::exists(resultsFilePath)){
        fs::remove(resultsFilePath);
    }

    std::unordered_map<std::string, Results> allResults;

    for(const auto& model : models){

        std::string modelPath = weightsPath + model + ".onnx";
        if (!fs::exists(modelPath)) {
            std::cerr << "Warning: Model file does not exist, skipping: " << modelPath << std::endl;
            continue;
        }

        allResults[model] = Results();
        allResults[model].weightsPath = modelPath;
        allResults[model].task = "detect";

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