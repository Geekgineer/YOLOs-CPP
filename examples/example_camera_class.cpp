/**
 * @file example_camera_class.cpp
 * @brief Real-time classification from camera
 * @details Live camera feed with image classification
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "yolos/tasks/classification.hpp"
#include "utils.hpp"

using namespace yolos::cls;

int main(int argc, char* argv[]) {
    std::string modelPath = "../../models/yolo11n-cls.onnx";
    std::string labelsPath = "../../models/imagenet_classes.txt";
    int cameraId = 0;
    
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) labelsPath = argv[2];
    if (argc > 3) cameraId = std::stoi(argv[3]);
    
    std::cout << "\n╔════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  YOLOs-CPP Real-Time Camera Classification" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nPress 'q' to quit, 's' to save snapshot\n" << std::endl;
    
    bool useGPU = false;
    std::cout << "🔄 Loading classification model..." << std::endl;
    
    try {
        YOLOClassifier classifier(modelPath, labelsPath, useGPU);
        std::cout << "✅ Model loaded!" << std::endl;
        
        cv::VideoCapture cap(cameraId);
        if (!cap.isOpened()) {
            std::cerr << "❌ Could not open camera" << std::endl;
            return -1;
        }
        
        std::cout << "🎬 Starting real-time classification...\n" << std::endl;
        
        cv::Mat frame;
        int frameCount = 0;
        auto startTime = std::chrono::high_resolution_clock::now();
        
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            
            frameCount++;
            ClassificationResult result = classifier.classify(frame);
            classifier.drawResult(frame, result, cv::Point(10, 30));
            
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime);
            double fps = elapsed.count() > 0 ? frameCount / static_cast<double>(elapsed.count()) : 0;
            
            std::string info = "FPS: " + std::to_string(static_cast<int>(fps));
            cv::putText(frame, info, cv::Point(10, frame.rows - 20), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            
            cv::imshow("YOLO Camera Classification (Press 'q' to quit, 's' to save)", frame);
            
            char key = cv::waitKey(1);
            if (key == 'q') break;
            if (key == 's') {
                std::string path = utils::saveImage(frame, "camera_class.jpg", "../../outputs/class/");
                std::cout << "📸 Saved: " << path << std::endl;
            }
        }
        
        cap.release();
        cv::destroyAllWindows();
        std::cout << "\n✅ Classification stopped!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
