/**
 * @file example_video_class.cpp
 * @brief Video classification using YOLO classification models
 * @details Classifies video frames into categories
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <filesystem>
#include "yolos/tasks/classification.hpp"
#include "utils.hpp"

using namespace yolos::cls;

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;
    
    // Default configuration
    std::string modelPath = "../../models/yolo11n-cls.onnx";
    std::string videoPath = "../../data/video.mp4";
    std::string labelsPath = "../../models/imagenet_classes.txt";
    std::string outputDir = "../../outputs/class/";
    
    // Parse command line arguments
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) videoPath = argv[2];
    if (argc > 3) labelsPath = argv[3];
    
    // Print usage information
    utils::printUsage(argv[0], "Video Classification", modelPath, videoPath, labelsPath);
    
    // Initialize YOLO classifier
    bool useGPU = false; // CPU by default
    std::cout << "🔄 Loading classification model: " << modelPath << std::endl;
    
    try {
        YOLOClassifier classifier(modelPath, labelsPath, useGPU);
        std::cout << "✅ Model loaded successfully!" << std::endl;
        std::cout << "📐 Input shape: " << classifier.getInputShape() << std::endl;
        
        // Open video file
        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            std::cerr << "❌ Could not open video: " << videoPath << std::endl;
            return -1;
        }
        
        // Get video properties
        int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        
        std::cout << "📹 Video: " << frameWidth << "x" << frameHeight 
                  << " @ " << fps << " FPS, " << totalFrames << " frames" << std::endl;
        
        // Setup video writer
        std::string outputPath = utils::getVideoOutputPath(videoPath, outputDir);
        cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('m','p','4','v'), 
                              fps, cv::Size(frameWidth, frameHeight));
        
        if (!writer.isOpened()) {
            std::cerr << "❌ Could not open video writer" << std::endl;
            return -1;
        }
        
        std::cout << "💾 Output will be saved to: " << outputPath << std::endl;
        std::cout << "\n🎬 Processing video..." << std::endl;
        
        cv::Mat frame;
        int frameCount = 0;
        auto startTime = std::chrono::high_resolution_clock::now();
        
        while (cap.read(frame)) {
            frameCount++;
            
            // Run classification
            ClassificationResult result = classifier.classify(frame);
            
            // Draw result on frame
            classifier.drawResult(frame, result, cv::Point(10, 30));
            
            // Add frame info
            std::string frameInfo = "Frame: " + std::to_string(frameCount) + "/" + std::to_string(totalFrames);
            cv::putText(frame, frameInfo, cv::Point(10, frameHeight - 20), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            
            // Write frame
            writer.write(frame);
            
            // Display progress
            if (frameCount % 30 == 0) {
                std::cout << "📊 Processed " << frameCount << "/" << totalFrames 
                          << " frames | Current: " << result.className << std::endl;
            }
            
            // Display frame
            cv::imshow("YOLO Video Classification", frame);
            if (cv::waitKey(1) == 'q') break;
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        double avgFps = duration.count() > 0 ? (frameCount * 1000.0) / duration.count() : 0.0;
        
        cap.release();
        writer.release();
        cv::destroyAllWindows();
        
        // Display metrics
        utils::printMetrics("Video Classification", duration.count(), avgFps);
        std::cout << "✅ Video processed successfully!" << std::endl;
        std::cout << "📁 Saved to: " << outputPath << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
