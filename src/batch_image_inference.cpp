/**
 * @file batch_image_inference.cpp
 * @brief Batch object detection on multiple images using YOLO models (v5, v7, v8, v9, v10, v11, v12).
 *
 * Packs every image into a single ONNX Runtime call via YOLODetector::batchDetect(),
 * which is where the GPU throughput win comes from. Models exported with a fixed
 * batch dimension fall back to a per-image loop automatically, so this works with
 * any export; use `model.export(format="onnx", dynamic=True)` for the batched path.
 *
 * Also demonstrates in-memory model loading (`--in-memory`): the ONNX bytes are
 * read into a buffer and handed to the detector directly, for encrypted stores,
 * network streams or resources embedded in the binary.
 *
 * Usage Instructions:
 * 1. Compile the application with the necessary OpenCV and YOLO dependencies.
 * 2. Provide the model path and image folder or list of images as arguments.
 * 3. Run the executable to initiate batch object detection.
 *
 * Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
 * Date: 29.09.2024
 */

#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "yolos/tasks/detection.hpp"

using namespace yolos::det;

namespace {

namespace fs = std::filesystem;

bool isImageFile(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
           ext == ".bmp" || ext == ".tiff" || ext == ".tif";
}

std::vector<std::string> collectImageFiles(const std::string& imagePath) {
    std::vector<std::string> imageFiles;

    if (fs::is_directory(imagePath)) {
        for (const auto& entry : fs::directory_iterator(imagePath)) {
            if (entry.is_regular_file() && isImageFile(entry.path())) {
                imageFiles.push_back(fs::absolute(entry.path()).string());
            }
        }
        std::sort(imageFiles.begin(), imageFiles.end());
    } else if (fs::is_regular_file(imagePath)) {
        imageFiles.push_back(imagePath);
    }

    return imageFiles;
}

} // namespace

int main(int argc, char* argv[]) {
    std::string labelsPath = "../models/coco.names";
    std::string imagePath = "../data/";
    std::string modelPath = "../models/yolo11n.onnx";
    bool loadFromMemory = false;
    bool showWindows = true;

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--in-memory") {
            loadFromMemory = true;
        } else if (arg == "--no-display") {
            showWindows = false;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0]
                      << " [model_path] [image_path_or_folder] [labels_path] [--in-memory] [--no-display]\n";
            return 0;
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() > 0) modelPath = positional[0];
    if (positional.size() > 1) imagePath = positional[1];
    if (positional.size() > 2) labelsPath = positional[2];

    if (positional.empty()) {
        std::cout << "Usage: " << argv[0]
                  << " [model_path] [image_path_or_folder] [labels_path] [--in-memory] [--no-display]\n";
        std::cout << "No model path provided. Using defaults: " << modelPath << ", " << imagePath << std::endl;
    }

    const std::vector<std::string> imageFiles = collectImageFiles(imagePath);
    if (imageFiles.empty()) {
        std::cerr << "No image files found at: " << imagePath << std::endl;
        return -1;
    }

    // Load all images
    std::vector<cv::Mat> images;
    std::vector<std::string> loadedFiles;
    images.reserve(imageFiles.size());
    for (const auto& imgPath : imageFiles) {
        cv::Mat img = cv::imread(imgPath);
        if (img.empty()) {
            std::cerr << "Warning: Could not open or find image: " << imgPath << std::endl;
            continue;
        }
        images.push_back(img);
        loadedFiles.push_back(imgPath);
    }
    if (images.empty()) {
        std::cerr << "No valid images to process." << std::endl;
        return -1;
    }

    const bool isGPU = true; // Set to false for CPU processing

    std::unique_ptr<YOLODetector> detector;
    if (loadFromMemory) {
        // Stand-in for an encrypted store, a network stream or an embedded resource:
        // whatever produces the bytes, only the buffer reaches the detector.
        const std::vector<uint8_t> modelBytes = yolos::utils::readFileBytes(modelPath);
        if (modelBytes.empty()) {
            std::cerr << "Failed to read model into memory: " << modelPath << std::endl;
            return -1;
        }
        std::cout << "[INFO] Loading model from memory (" << modelBytes.size() << " bytes)" << std::endl;

        // Class names come in as a vector, so no labels file is needed either.
        const std::vector<std::string> classNames = yolos::utils::getClassNames(labelsPath);
        detector = std::make_unique<YOLODetector>(modelBytes.data(), modelBytes.size(), classNames, isGPU);
        // modelBytes may go out of scope here: ONNX Runtime copied it during session creation.
    } else {
        detector = std::make_unique<YOLODetector>(modelPath, labelsPath, isGPU);
    }

    if (detector->supportsBatchSize(images.size())) {
        std::cout << "[INFO] Running a single batched inference over " << images.size()
                  << " image(s)" << std::endl;
    } else {
        std::cout << "[INFO] Model batch size is fixed at " << detector->getModelBatchSize()
                  << "; falling back to a per-image loop for " << images.size()
                  << " image(s). Re-export with dynamic=True for true batching." << std::endl;
    }

    const auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<Detection>> allResults = detector->batchDetect(images, 0.45f);
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::high_resolution_clock::now() - start);

    std::cout << "Detection completed in: " << duration.count() << " ms ("
              << (duration.count() / static_cast<double>(images.size())) << " ms/image)" << std::endl;

    for (size_t i = 0; i < allResults.size(); ++i) {
        std::cout << "\nImage: " << loadedFiles[i] << " size: " << images[i].size() << std::endl;
        std::cout << "Number of detections: " << allResults[i].size() << std::endl;
        for (size_t j = 0; j < allResults[i].size(); ++j) {
            const Detection& det = allResults[i][j];
            std::cout << "Detection " << j << ": Class=" << det.classId
                      << ", Confidence=" << det.conf
                      << ", Box=(" << det.box.x << "," << det.box.y
                      << "," << det.box.width << "," << det.box.height << ")" << std::endl;
        }
        // Draw bounding boxes on the image
        detector->drawDetections(images[i], allResults[i]);
        if (showWindows) {
            cv::imshow("Detections - " + std::to_string(i), images[i]);
        }
    }

    if (showWindows) {
        cv::waitKey(0);
    }
    return 0;
}
