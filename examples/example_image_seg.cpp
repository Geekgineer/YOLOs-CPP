/**
 * @file example_image_seg.cpp
 * @brief Image segmentation using YOLO segmentation models
 * @details Performs instance segmentation with masks and bounding boxes
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <vector>
#include "yolos/tasks/segmentation.hpp"
#include "utils.hpp"

using namespace yolos::seg;

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;
    
    // Default configuration
    std::string modelPath = "../../models/yolo11n-seg.onnx";
    std::string inputPath = "../../data/dog.jpg";
    std::string labelsPath = "../../models/coco.names";
    std::string outputDir = "../../outputs/seg/";
    
    // Parse command line arguments
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) inputPath = argv[2];
    if (argc > 3) labelsPath = argv[3];
    
    // Print usage information
    utils::printUsage(argv[0], "Segmentation", modelPath, inputPath, labelsPath);
    
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
    
    // Initialize YOLO segmentation detector
    bool useGPU = false; // CPU by default
    std::cout << "🔄 Loading segmentation model: " << modelPath << std::endl;
    
    try {
        YOLOSegDetector detector(modelPath, labelsPath, useGPU);
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
            
            // Run segmentation with timing
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<Segmentation> results = detector.segment(image);
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
            
            // Print results
            std::cout << "✅ Segmentation completed!" << std::endl;
            std::cout << "📊 Found " << results.size() << " segments" << std::endl;
            
            for (size_t i = 0; i < results.size(); ++i) {
                std::cout << "   [" << i << "] Class=" << results[i].classId 
                          << ", Confidence=" << std::fixed << std::setprecision(2) << results[i].conf
                          << ", Box=(" << results[i].box.x << "," << results[i].box.y << ","
                          << results[i].box.width << "x" << results[i].box.height << ")"
                          << ", Mask size=" << results[i].mask.size() << std::endl;
            }
            
            // Draw segmentations with boxes and masks
            cv::Mat resultImage = image.clone();
            detector.drawSegmentations(resultImage, results);
            
            // Save output with timestamp
            std::string outputPath = utils::saveImage(resultImage, imgPath, outputDir);
            std::cout << "💾 Saved result to: " << outputPath << std::endl;
            
            // Display metrics
            utils::printMetrics("Segmentation", duration.count());
            
            // Display result
            cv::imshow("YOLO Segmentation", resultImage);
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
