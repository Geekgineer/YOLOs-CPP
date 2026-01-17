/**
 * @file example_image_pose.cpp
 * @brief Pose estimation on images using YOLO pose models
 * @details Detects human poses with keypoints and skeletal connections
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <vector>
#include "yolos/tasks/pose.hpp"
#include "utils.hpp"

using namespace yolos::pose;

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;
    
    // Default configuration
    std::string modelPath = "../../models/yolo11n-pose.onnx";
    std::string inputPath = "../../data/person.jpg";
    std::string labelsPath = "../../models/coco.names";
    std::string outputDir = "../../outputs/pose/";
    
    // Parse command line arguments
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) inputPath = argv[2];
    if (argc > 3) labelsPath = argv[3];
    
    // Print usage information
    utils::printUsage(argv[0], "Pose Estimation", modelPath, inputPath, labelsPath);
    
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
    
    // Initialize YOLO pose detector
    bool useGPU = false; // CPU by default
    std::cout << "🔄 Loading pose estimation model: " << modelPath << std::endl;
    
    try {
        YOLOPoseDetector detector(modelPath, labelsPath, useGPU);
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
            
            // Run pose detection with timing
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<PoseResult> detections = detector.detect(image, 0.4f, 0.5f);
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
            
            // Print results
            std::cout << "✅ Pose detection completed!" << std::endl;
            std::cout << "📊 Found " << detections.size() << " person(s)" << std::endl;
            
            for (size_t i = 0; i < detections.size(); ++i) {
                std::cout << "   [" << i << "] Confidence=" << std::fixed << std::setprecision(2) 
                          << detections[i].conf
                          << ", Box=(" << detections[i].box.x << "," << detections[i].box.y << ","
                          << detections[i].box.width << "x" << detections[i].box.height << ")"
                          << ", Keypoints=" << detections[i].keypoints.size() << std::endl;
            }
            
            // Draw pose keypoints and skeleton
            cv::Mat resultImage = image.clone();
            detector.drawPoses(resultImage, detections);
            
            // Save output with timestamp
            std::string outputPath = utils::saveImage(resultImage, imgPath, outputDir);
            std::cout << "💾 Saved result to: " << outputPath << std::endl;
            
            // Display metrics
            utils::printMetrics("Pose Estimation", duration.count());
            
            // Display result
            cv::imshow("YOLO Pose Estimation", resultImage);
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
