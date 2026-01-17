/**
 * @file example_image_obb.cpp
 * @brief Oriented Bounding Box detection on images
 * @details Detects rotated objects with oriented bounding boxes
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <vector>
#include "yolos/tasks/obb.hpp"
#include "utils.hpp"

using namespace yolos::obb;

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;
    
    // Default configuration
    std::string modelPath = "../../models/yolo11n-obb.onnx";
    std::string inputPath = "../../data/dog.jpg";
    std::string labelsPath = "../../models/Dota.names";
    std::string outputDir = "../../outputs/obb/";
    
    // Parse command line arguments
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) inputPath = argv[2];
    if (argc > 3) labelsPath = argv[3];
    
    // Print usage information
    utils::printUsage(argv[0], "Oriented Bounding Box", modelPath, inputPath, labelsPath);
    
    // Collect image files
    std::vector<std::string> imageFiles;
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
    
    // Initialize YOLO OBB detector
    bool useGPU = false; // CPU by default
    std::cout << "🔄 Loading OBB model: " << modelPath << std::endl;
    
    try {
        YOLOOBBDetector detector(modelPath, labelsPath, useGPU);
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
            
            // Run OBB detection with timing
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<OBBResult> detections = detector.detect(image);
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
            
            // Print results
            std::cout << "✅ OBB detection completed!" << std::endl;
            std::cout << "📊 Found " << detections.size() << " oriented boxes" << std::endl;
            
            for (size_t i = 0; i < detections.size(); ++i) {
                std::cout << "   [" << i << "] Class=" << detections[i].classId 
                          << ", Confidence=" << std::fixed << std::setprecision(2) << detections[i].conf
                          << ", Center=(" << detections[i].box.x << "," << detections[i].box.y << ")"
                          << ", Size=(" << detections[i].box.width << "x" << detections[i].box.height << ")"
                          << ", Angle=" << (detections[i].box.angle * 180.0 / CV_PI) << "°" << std::endl;
            }
            
            // Draw oriented bounding boxes
            cv::Mat resultImage = image.clone();
            detector.drawDetections(resultImage, detections);
            
            // Save output with timestamp
            std::string outputPath = utils::saveImage(resultImage, imgPath, outputDir);
            std::cout << "💾 Saved result to: " << outputPath << std::endl;
            
            // Display metrics
            utils::printMetrics("OBB Detection", duration.count());
            
            // Display result
            cv::imshow("YOLO OBB Detection", resultImage);
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
