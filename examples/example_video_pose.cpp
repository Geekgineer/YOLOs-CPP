/**
 * @file example_video_pose.cpp
 * @brief Video pose estimation using YOLO pose models
 * @details Detects human poses in video with keypoints and skeleton
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <filesystem>
#include "yolos/tasks/pose.hpp"
#include "utils.hpp"

using namespace yolos::pose;

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;
    
    // Default configuration
    std::string modelPath = "../../models/yolo11n-pose.onnx";
    std::string videoPath = "../../data/video.mp4";
    std::string labelsPath = "../../models/coco.names";
    std::string outputDir = "../../outputs/pose/";
    
    // Parse command line arguments
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) videoPath = argv[2];
    if (argc > 3) labelsPath = argv[3];
    
    // Print usage information
    utils::printUsage(argv[0], "Video Pose Estimation", modelPath, videoPath, labelsPath);
    
    // Initialize YOLO pose detector
    bool useGPU = false; // CPU by default
    std::cout << "🔄 Loading pose estimation model: " << modelPath << std::endl;
    
    try {
        YOLOPoseDetector detector(modelPath, labelsPath, useGPU);
        std::cout << "✅ Model loaded successfully!" << std::endl;
        
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
            
            // Run pose detection
            std::vector<PoseResult> detections = detector.detect(frame, 0.4f, 0.5f);
            
            // Draw pose keypoints
            detector.drawPoses(frame, detections);
            
            // Add frame info
            std::string frameInfo = "Frame: " + std::to_string(frameCount) + "/" + std::to_string(totalFrames) +
                                   " | Persons: " + std::to_string(detections.size());
            cv::putText(frame, frameInfo, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            // Write frame
            writer.write(frame);
            
            // Display progress
            if (frameCount % 30 == 0) {
                std::cout << "📊 Processed " << frameCount << "/" << totalFrames << " frames" << std::endl;
            }
            
            // Display frame
            cv::imshow("YOLO Video Pose Estimation", frame);
            if (cv::waitKey(1) == 'q') break;
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        double avgFps = duration.count() > 0 ? (frameCount * 1000.0) / duration.count() : 0.0;
        
        cap.release();
        writer.release();
        cv::destroyAllWindows();
        
        // Display metrics
        utils::printMetrics("Video Pose Estimation", duration.count(), avgFps);
        std::cout << "✅ Video processed successfully!" << std::endl;
        std::cout << "📁 Saved to: " << outputPath << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
