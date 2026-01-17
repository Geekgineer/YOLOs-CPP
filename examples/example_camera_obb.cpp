/**
 * @file example_camera_obb.cpp
 * @brief Real-time OBB detection from camera
 * @details Live camera feed with oriented bounding box detection
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "yolos/tasks/obb.hpp"
#include "utils.hpp"

using namespace yolos::obb;

int main(int argc, char* argv[]) {
    std::string modelPath = "../../models/yolo11n-obb.onnx";
    std::string labelsPath = "../../models/Dota.names";
    int cameraId = 0;
    
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) labelsPath = argv[2];
    if (argc > 3) cameraId = std::stoi(argv[3]);
    
    std::cout << "\n╔════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  YOLOs-CPP Real-Time Camera OBB Detection" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nPress 'q' to quit, 's' to save snapshot\n" << std::endl;
    
    bool useGPU = false;
    std::cout << "🔄 Loading OBB model..." << std::endl;
    
    try {
        YOLOOBBDetector detector(modelPath, labelsPath, useGPU);
        std::cout << "✅ Model loaded!" << std::endl;
        
        cv::VideoCapture cap(cameraId);
        if (!cap.isOpened()) {
            std::cerr << "❌ Could not open camera" << std::endl;
            return -1;
        }
        
        std::cout << "🎬 Starting real-time OBB detection...\n" << std::endl;
        
        cv::Mat frame;
        int frameCount = 0;
        auto startTime = std::chrono::high_resolution_clock::now();
        
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            
            frameCount++;
            std::vector<OBBResult> detections = detector.detect(frame);
            detector.drawDetections(frame, detections);
            
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime);
            double fps = elapsed.count() > 0 ? frameCount / static_cast<double>(elapsed.count()) : 0;
            
            std::string info = "FPS: " + std::to_string(static_cast<int>(fps)) + 
                              " | OBBs: " + std::to_string(detections.size());
            cv::putText(frame, info, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            cv::imshow("YOLO Camera OBB (Press 'q' to quit, 's' to save)", frame);
            
            char key = cv::waitKey(1);
            if (key == 'q') break;
            if (key == 's') {
                std::string path = utils::saveImage(frame, "camera_obb.jpg", "../../outputs/obb/");
                std::cout << "📸 Saved: " << path << std::endl;
            }
        }
        
        cap.release();
        cv::destroyAllWindows();
        std::cout << "\n✅ OBB detection stopped!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
