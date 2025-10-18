#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>

#include "class/YOLOCLASS.hpp"

int main(int argc, char** argv){
    std::string labelsPath = "../models/coco.names";     // detection labels; use proper labels for your model
    std::string imagePath  = "../data/dog2.jpg";         // change to your image
    std::string modelPath  = "../models/yolo11l-cls.onnx";   // classification ONNX
    int versionArg = 12;

    // Parse command line arguments
    if (argc > 1) {
        modelPath = argv[1];
    }
    if (argc > 2) {
        imagePath = argv[2];
    }
    if (argc > 3) {
        labelsPath = argv[3];
    }
    if (argc > 4) {
        versionArg = std::stoi(argv[4]);
    }

    // Print usage information
    if (argc == 1) {
        std::cout << "Usage: " << argv[0] << " [model_path] [image_path] [labels_path] [version]\n";
        std::cout << "Version: 5 (YOLOv5), 8 (YOLOv8), 11 (YOLOv11), 12 (YOLOv12)\n";
        std::cout << "Using defaults: model=" << modelPath << ", image=" << imagePath 
                  << ", labels=" << labelsPath << ", version=" << versionArg << std::endl;
    }

    // Init classifier
    bool useGPU = true;    
    YOLOClassVersion ver;
    if (versionArg == 5) {
        ver = YOLOClassVersion::V5;
    } else if (versionArg == 8) {
        ver = YOLOClassVersion::V8;
    } else if (versionArg == 11) {
        ver = YOLOClassVersion::V11;
    } else {
        ver = YOLOClassVersion::V12;
    }
    
    std::cout << "Using YOLO version: " << versionArg << std::endl;
    YOLOClassifier classifier(modelPath, labelsPath, useGPU, ver);

    // Load image
    cv::Mat image = cv::imread(imagePath);
    if(image.empty()){
        std::cerr << "Error: could not open image: " << imagePath << std::endl;
        return -1;
    }

    // Run classification
    auto start = std::chrono::high_resolution_clock::now();
    ClassificationResult result = classifier.classify(image);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    std::cout << "Classification took: " << elapsed.count() << " ms\n";

    classifier.drawResult(image, result, cv::Point(18, 28));
    cv::imshow("Classification", image);
    cv::waitKey(0);
    return 0;
}


