#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>

#include "yolos/tasks/classification.hpp"

using namespace yolos::cls;

int main(int argc, char** argv){
    const std::string labelsPath = "../models/coco.names";     // detection labels; use proper labels for your model
    const std::string imagePath  = "../data/dog.jpg";         // change to your image
    const std::string modelPath  = "../models/yolov8n-cls.onnx";   // classification ONNX
    int versionArg = 11;

    // Init classifier
    bool useGPU = false;    
    YOLOClassifier classifier(modelPath, labelsPath, useGPU);

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


