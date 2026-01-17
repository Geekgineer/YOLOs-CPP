/**
 * @file example_camera_seg.cpp
 * @brief Real-time segmentation from camera using YOLO segmentation models
 * @details Live camera feed with instance segmentation
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "yolos/tasks/segmentation.hpp"
#include "utils.hpp"

using namespace yolos::seg;

int main(int argc, char* argv[]) {
    // Default configuration
    std::string modelPath = "../../models/yolo11n-seg.onnx";
    std::string labelsPath = "../../models/coco.names";
    int cameraId = 0;
    
    // Parse command line arguments
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) labelsPath = argv[2];
    if (argc > 3) cameraId = std::stoi(argv[3]);
    
    std::cout << "\n╔════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  YOLOs-CPP Real-Time Camera Segmentation" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nPress 'q' to quit, 's' to save snapshot\n" << std::endl;
    
    // Initialize YOLO segmentation detector
    bool useGPU = false; // CPU by default
    std::cout << "🔄 Loading segmentation model..." << std::endl;
    
    try {
        YOLOSegDetector detector(modelPath, labelsPath, useGPU);
        std::cout << "✅ Model loaded successfully!" << std::endl;
        
        cv::VideoCapture cap(cameraId);
        if (!cap.isOpened()) {
            std::cerr << "❌ Could not open camera" << std::endl;
            return -1;
        }
        
        std::cout << "🎬 Starting real-time segmentation...\n" << std::endl;
        
        cv::Mat frame;
        int frameCount = 0;
        auto startTime = std::chrono::high_resolution_clock::now();
        
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            
            frameCount++;
            std::vector<Segmentation> results = detector.segment(frame);
            detector.drawSegmentations(frame, results);
            
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime);
            double fps = elapsed.count() > 0 ? frameCount / static_cast<double>(elapsed.count()) : 0;
            
            std::string info = "FPS: " + std::to_string(static_cast<int>(fps)) + 
                              " | Segments: " + std::to_string(results.size());
            cv::putText(frame, info, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            cv::imshow("YOLO Camera Segmentation (Press 'q' to quit, 's' to save)", frame);
            
            char key = cv::waitKey(1);
            if (key == 'q') break;
            if (key == 's') {
                std::string path = utils::saveImage(frame, "camera_seg.jpg", "../../outputs/seg/");
                std::cout << "📸 Saved: " << path << std::endl;
            }
        }
        
        cap.release();
        cv::destroyAllWindows();
        std::cout << "\n✅ Segmentation stopped!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
