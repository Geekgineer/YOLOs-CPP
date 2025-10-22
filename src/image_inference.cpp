/**
 * @file image_inference.cpp
 * @brief Object detection in a static image using YOLO models (v5, v7, v8, v9, v10, v11, v12).
 * 
 * This file implements an object detection application that utilizes YOLO 
 * (You Only Look Once) models, specifically versions 5, 7, 8, 9, 10, 11 and 12. 
 * The application loads a specified image, processes it to detect objects, 
 * and displays the results with bounding boxes around detected objects.
 *
 * The application supports the following functionality:
 * - Loading a specified image from disk.
 * - Initializing the YOLO detector with the desired model and labels.
 * - Detecting objects within the image.
 * - Drawing bounding boxes around detected objects and displaying the result.
 *
 * Configuration parameters can be adjusted to suit specific requirements:
 * - `isGPU`: Set to true to enable GPU processing for improved performance; 
 *   set to false for CPU processing.
 * - `labelsPath`: Path to the class labels file (e.g., COCO dataset).
 * - `imagePath`: Path to the image file to be processed (e.g., dogs.jpg).
 * - `modelPath`: Path to the desired YOLO model file (e.g., ONNX format).
 *
 * The application can be extended to use different YOLO versions by modifying 
 * the model path and the corresponding detector class.
 *
 * Usage Instructions:
 * 1. Compile the application with the necessary OpenCV and YOLO dependencies.
 * 2. Ensure that the specified image and model files are present in the 
 *    provided paths.
 * 3. Run the executable to initiate the object detection process.
 *
 * @note The code includes commented-out sections to demonstrate how to switch 
 * between different YOLO models and image inputs.
 *
 * Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
 * Date: 29.09.2024
 */

// Include necessary headers
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <algorithm> // Required for std::transform


// #ifndef DEBUG_MODE
// #define DEBUG_MODE
// #endif
// #ifndef TIMING_MODE
// #define TIMING_MODE
// #endif
#include "det/YOLO.hpp"


int main(int argc, char* argv[]){
    namespace fs = std::filesystem;
    // Paths to the model, labels, and test image
    std::string labelsPath = "../models/coco.names";
    std::string imagePath = "../data/dog.jpg";           // Default image path
    std::string modelPath = "../models/yolo11n.onnx";
    std::vector<std::string> imageFiles;

    if(argc > 1){
        modelPath = argv[1];            
    }
    // If an argument is provided, use it as the image path or directory
    if (argc > 2) {
        imagePath = argv[2];
        if (fs::is_directory(imagePath)) {
            // Collect all image files in the directory
            for (const auto& entry : fs::directory_iterator(imagePath)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tiff" || ext == ".tif") {
                        imageFiles.push_back(fs::absolute(entry.path()).string());
                    }
                }
            }
            if (imageFiles.empty()) {
                std::cerr << "No image files found in directory: " << imagePath << std::endl;
                return -1;
            }
        } else if (fs::is_regular_file(imagePath)) {
            imageFiles.push_back(imagePath);
        } else {
            std::cerr << "Provided path is not a valid file or directory: " << imagePath << std::endl;
            return -1;
        }
    } else {
        std::cout << "Usage: " << argv[0] << " <image_path_or_folder>\n";
        std::cout << "No image path provided. Using default: " << imagePath << std::endl;
        imageFiles.push_back(imagePath);
    }
    if (argc > 3){
        labelsPath = argv[3];
    }
    // Initialize the YOLO detector with the chosen model and labels
    bool isGPU = true; // Set to false for CPU processing
    // YOLO10Detector detector(modelPath, labelsPath, isGPU);
    YOLODetector detector(modelPath, labelsPath, isGPU);
    for (const auto& imgPath : imageFiles) {
        std::cout << "\nProcessing: " << imgPath << std::endl;
        // Load an image
        cv::Mat image = cv::imread(imgPath);
        if (image.empty()) {
            std::cerr << "Error: Could not open or find the image!\n";
            continue;
        }
        // Detect objects in the image and measure execution time
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Detection> results = detector.detect(image);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - start);
        std::cout << "Detection completed in: " << duration.count() << " ms" << std::endl;
        std::cout << "Number of detections found: " << results.size() << std::endl;
        // Print details of each detection
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "Detection " << i << ": Class=" << results[i].classId 
                      << ", Confidence=" << results[i].conf 
                      << ", Box=(" << results[i].box.x << "," << results[i].box.y 
                      << "," << results[i].box.width << "," << results[i].box.height << ")" << std::endl;
        }

        // Draw bounding boxes on the image
        detector.drawBoundingBox(image, results); // simple bbox drawing
        // detector.drawBoundingBoxMask(image, results); // Uncomment for mask drawing
        // Display the image
        cv::imshow("Detections", image);
        cv::waitKey(0); // Wait for a key press to close the window
    }
    return 0;
}