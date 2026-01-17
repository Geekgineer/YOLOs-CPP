/**
 * @file example_image_class.cpp
 * @brief Image classification using YOLO classification models
 * @details Classifies images into predefined categories
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <vector>
#include "yolos/tasks/classification.hpp"
#include "utils.hpp"

using namespace yolos::cls;

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;
    
    // Default configuration
    std::string modelPath = "../../models/yolo11n-cls.onnx";
    std::string inputPath = "../../data/dog.jpg";
    std::string labelsPath = "../../models/imagenet_classes.txt";
    std::string outputDir = "../../outputs/class/";
    
    // Parse command line arguments
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) inputPath = argv[2];
    if (argc > 3) labelsPath = argv[3];
    
    // Print usage information
    utils::printUsage(argv[0], "Classification", modelPath, inputPath, labelsPath);
    
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
    
    // Initialize YOLO classifier
    bool useGPU = false; // CPU by default
    std::cout << "🔄 Loading classification model: " << modelPath << std::endl;
    
    try {
        // Use YOLO11 version by default
        YOLOClassifier classifier(modelPath, labelsPath, useGPU);
        std::cout << "✅ Model loaded successfully!" << std::endl;
        std::cout << "📐 Input shape: " << classifier.getInputShape() << std::endl;
        
        // Process each image
        for (const auto& imgPath : imageFiles) {
            std::cout << "\n📷 Processing: " << imgPath << std::endl;
            
            // Load image
            cv::Mat image = cv::imread(imgPath);
            if (image.empty()) {
                std::cerr << "❌ Could not load image: " << imgPath << std::endl;
                continue;
            }
            
            // Run classification with timing
            auto start = std::chrono::high_resolution_clock::now();
            ClassificationResult result = classifier.classify(image);
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
            
            // Print results
            std::cout << "✅ Classification completed!" << std::endl;
            std::cout << "📊 Result: " << result.className 
                      << " (ID: " << result.classId << ")"
                      << " with confidence: " << std::fixed << std::setprecision(4) 
                      << (result.confidence * 100.0f) << "%" << std::endl;
            
            // Draw result on image
            cv::Mat resultImage = image.clone();
            classifier.drawResult(resultImage, result, cv::Point(10, 30));
            
            // Save output with timestamp
            std::string outputPath = utils::saveImage(resultImage, imgPath, outputDir);
            std::cout << "💾 Saved result to: " << outputPath << std::endl;
            
            // Display metrics
            utils::printMetrics("Classification", duration.count());
            
            // Display result
            cv::imshow("YOLO Classification", resultImage);
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
