/**
 * @file example_image_det.cpp
 * @brief Standard object detection on images using YOLO models
 * @details Loads a YOLO detection model and performs inference on images
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <vector>
#include "yolos/tasks/detection.hpp"
#include "utils.hpp"

using namespace yolos::det;

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;
    
    // Default configuration
    std::string modelPath = "../../models/yolo11n.onnx";
    std::string inputPath = "../../data/dog.jpg";
    std::string labelsPath = "../../models/coco.names";
    std::string outputDir = "../../outputs/det/";
    
    // Check for help flag
    if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        utils::printUsage(argv[0], "Object Detection", modelPath, inputPath, labelsPath);
        return 0;
    }
    
    // Parse command line arguments
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) inputPath = argv[2];
    if (argc > 3) labelsPath = argv[3];
    
    // Print usage information
    utils::printUsage(argv[0], "Object Detection", modelPath, inputPath, labelsPath);
    
    // Collect image files
    std::vector<std::string> imageFiles;
    try {
        if (fs::is_directory(inputPath)) {
            for (const auto& entry : fs::directory_iterator(inputPath)) {
                if (entry.is_regular_file() && utils::isImageFile(entry.path().string())) {
                    imageFiles.push_back(fs::absolute(entry.path()).string());
                }
            }
            if (imageFiles.empty()) {
                std::cerr << "❌ No image files found in: " << inputPath << std::endl;
                return -1;
            }
        } else if (fs::is_regular_file(inputPath)) {
            imageFiles.push_back(inputPath);
        } else {
            std::cerr << "❌ Invalid path: " << inputPath << std::endl;
            return -1;
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "❌ Filesystem error: " << e.what() << std::endl;
        return -1;
    }
    
    // Initialize YOLO detector
    bool useGPU = false; // CPU by default
    std::cout << "🔄 Loading detection model: " << modelPath << std::endl;
    
    try {
        YOLODetector detector(modelPath, labelsPath, useGPU);
        std::cout << "✅ Model loaded successfully!" << std::endl;
        
        // Process each image
        for (const auto& imgPath : imageFiles) {
            std::cout << "\n📷 Processing: " << imgPath << std::endl;
            
            // Load image
            cv::Mat image = cv::imread(imgPath);
            if (image.empty()) {
                std::cerr << "❌ Could not load image: " << imgPath << std::endl;
                continue;
            }
            
            // Run detection with timing
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<Detection> detections = detector.detect(image);
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
            
            // Print results
            std::cout << "✅ Detection completed!" << std::endl;
            std::cout << "📊 Found " << detections.size() << " objects" << std::endl;
            
            for (size_t i = 0; i < detections.size(); ++i) {
                std::cout << "   [" << i << "] Class=" << detections[i].classId 
                          << ", Confidence=" << std::fixed << std::setprecision(2) << detections[i].conf
                          << ", Box=(" << detections[i].box.x << "," << detections[i].box.y << ","
                          << detections[i].box.width << "x" << detections[i].box.height << ")" << std::endl;
            }
            
            // Draw detections
            cv::Mat resultImage = image.clone();
            detector.drawDetections(resultImage, detections);
            
            // Save output with timestamp
            std::string outputPath = utils::saveImage(resultImage, imgPath, outputDir);
            std::cout << "💾 Saved result to: " << outputPath << std::endl;
            
            // Display metrics
            utils::printMetrics("Detection", duration.count());
            
            // Display result
            cv::imshow("YOLO Detection", resultImage);
            std::cout << "Press any key to continue..." << std::endl;
            cv::waitKey(0);
        }
        
        cv::destroyAllWindows();
        std::cout << "\n✅ All images processed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
