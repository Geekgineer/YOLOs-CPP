/**
 * @file example_camera_det.cpp
 * @brief Real-time object detection from camera using YOLO models
 * @details Live camera feed with object detection
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "yolos/tasks/detection.hpp"
#include "utils.hpp"

using namespace yolos::det;

int main(int argc, char* argv[]) {
    // Default configuration
    std::string modelPath = "../../models/yolo11n.onnx";
    std::string labelsPath = "../../models/coco.names";
    int cameraId = 0;
    
    // Parse command line arguments
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) labelsPath = argv[2];
    if (argc > 3) cameraId = std::stoi(argv[3]);
    
    std::cout << "\n╔════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  YOLOs-CPP Real-Time Camera Detection" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nModel: " << modelPath << std::endl;
    std::cout << "Labels: " << labelsPath << std::endl;
    std::cout << "Camera ID: " << cameraId << std::endl;
    std::cout << "\nPress 'q' to quit, 's' to save snapshot\n" << std::endl;
    
    // Initialize YOLO detector
    bool useGPU = false; // CPU by default
    std::cout << "🔄 Loading detection model..." << std::endl;
    
    try {
        YOLODetector detector(modelPath, labelsPath, useGPU);
        std::cout << "✅ Model loaded successfully!" << std::endl;
        
        // Open camera
        cv::VideoCapture cap(cameraId);
        if (!cap.isOpened()) {
            std::cerr << "❌ Could not open camera " << cameraId << std::endl;
            return -1;
        }
        
        std::cout << "📹 Camera opened successfully!" << std::endl;
        std::cout << "🎬 Starting real-time detection...\n" << std::endl;
        
        cv::Mat frame;
        int frameCount = 0;
        auto startTime = std::chrono::high_resolution_clock::now();
        
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            
            frameCount++;
            
            // Run detection
            auto detectStart = std::chrono::high_resolution_clock::now();
            std::vector<Detection> detections = detector.detect(frame);
            auto detectEnd = std::chrono::high_resolution_clock::now();
            auto detectDuration = std::chrono::duration_cast<std::chrono::milliseconds>(detectEnd - detectStart);
            
            // Draw detections
            detector.drawDetections(frame, detections);
            
            // Calculate FPS
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime);
            double fps = elapsed.count() > 0 ? frameCount / static_cast<double>(elapsed.count()) : 0;
            
            // Add info overlay
            std::string info = "FPS: " + std::to_string(static_cast<int>(fps)) + 
                              " | Objects: " + std::to_string(detections.size()) +
                              " | Time: " + std::to_string(detectDuration.count()) + "ms";
            cv::putText(frame, info, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            // Display frame
            cv::imshow("YOLO Camera Detection (Press 'q' to quit, 's' to save)", frame);
            
            char key = cv::waitKey(1);
            if (key == 'q') break;
            if (key == 's') {
                std::string snapshotPath = utils::saveImage(frame, "camera_snapshot.jpg", "../../outputs/det/");
                std::cout << "📸 Snapshot saved to: " << snapshotPath << std::endl;
            }
        }
        
        cap.release();
        cv::destroyAllWindows();
        
        std::cout << "\n✅ Camera detection stopped!" << std::endl;
        std::cout << "📊 Processed " << frameCount << " frames" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
