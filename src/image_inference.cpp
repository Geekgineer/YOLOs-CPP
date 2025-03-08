/**
 * @file image_inference.cpp
 * @brief Object detection in a static image using YOLO models (v5, v7, v8, v9, v10, v11, v12).
 * 
 * This file implements an object detection application that utilizes YOLO 
 * (You Only Look Once) models, specifically versions 5, 7, 8, 9, 10, 11 and 12. 
 * The application loads a specified image, processes it to detect objects, 
 * and displays the results with bounding boxes around detected objects.
 *
 * The application supports the following functionality:
 * - Loading a specified image from disk.
 * - Initializing the YOLO detector with the desired model and labels.
 * - Detecting objects within the image.
 * - Drawing bounding boxes around detected objects and displaying the result.
 *
 * Configuration parameters can be adjusted to suit specific requirements:
 * - `isGPU`: Set to true to enable GPU processing for improved performance; 
 *   set to false for CPU processing.
 * - `labelsPath`: Path to the class labels file (e.g., COCO dataset).
 * - `imagePath`: Path to the image file to be processed (e.g., dogs.jpg).
 * - `modelPath`: Path to the desired YOLO model file (e.g., ONNX format).
 *
 * The application can be extended to use different YOLO versions by modifying 
 * the model path and the corresponding detector class.
 *
 * Usage Instructions:
 * 1. Compile the application with the necessary OpenCV and YOLO dependencies.
 * 2. Ensure that the specified image and model files are present in the 
 *    provided paths.
 * 3. Run the executable to initiate the object detection process.
 *
 * @note The code includes commented-out sections to demonstrate how to switch 
 * between different YOLO models and image inputs.
 *
 * Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
 * Date: 29.09.2024
 */

// Include necessary headers
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>

// #include "det/YOLO5.hpp"  // Uncomment for YOLOv5
// #include "det/YOLO7.hpp"  // Uncomment for YOLOv7
// #include "det/YOLO8.hpp"  // Uncomment for YOLOv8
// #include "det/YOLO9.hpp"  // Uncomment for YOLOv9
// #include "det/YOLO10.hpp" // Uncomment for YOLOv10
// #include "det/YOLO11.hpp" // Uncomment for YOLOv11
#include "det/YOLO12.hpp" // Uncomment for YOLOv12



int main(){

    // Paths to the model, labels, and test image
    const std::string labelsPath = "../models/coco.names";
    const std::string imagePath = "../data/dog.jpg";           // Primary image path

    // Uncomment the desired image path for testing
    // const std::string imagePath = "../data/happy_dogs.jpg";  // Alternate image
    // const std::string imagePath = "../data/desk.jpg";        // Another alternate image

    // Model paths for different YOLO versions
    // Uncomment the desired model path for testing
    // const std::string modelPath = "../models/yolo5-n6.onnx";      // YOLOv5
    // const std::string modelPath = "../models/yolo7-tiny.onnx";       // YOLOv7
    // const std::string modelPath = "../models/yolo8n.onnx"; // YOLOv8
    // const std::string modelPath = "../models/yolov9s.onnx"; // YOLOv9 
    // const std::string modelPath = "../models/yolo10n.onnx"; // YOLOv10 
    // const std::string modelPath = "../quantized_models/yolo10n_uint8.onnx"; // Quantized YOLOv10
    // const std::string modelPath = "../models/yolo11n.onnx"; // YOLOv11 
    const std::string modelPath = "../models/yolo12n.onnx"; // YOLOv12 



    // Initialize the YOLO detector with the chosen model and labels
    bool isGPU = true; // Set to false for CPU processing
    // YOLO7Detector detector(modelPath, labelsPath, isGPU);
    // YOLO5Detector detector(modelPath, labelsPath, isGPU);  // Uncomment for YOLOv5
    // YOLO8Detector detector(modelPath, labelsPath, isGPU);  // Uncomment for YOLOv8
    // YOLO9Detector detector(modelPath, labelsPath, isGPU); // Uncomment for YOLOv9
    // YOLO10Detector detector(modelPath, labelsPath, isGPU); // Uncomment for YOLOv10
    // YOLO11Detector detector(modelPath, labelsPath, isGPU); // Uncomment for YOLOv11
    YOLO12Detector detector(modelPath, labelsPath, isGPU); // Uncomment for YOLOv12


    // Load an image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        std::cerr << "Error: Could not open or find the image!\n";
        return -1;
    }

    

    // Detect objects in the image and measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Detection> results = detector.detect(image);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now() - start);

    std::cout << "Detection completed in: " << duration.count() << " ms" << std::endl;


    // Draw bounding boxes on the image
    detector.drawBoundingBox(image, results); // simple bbox drawing
    // detector.drawBoundingBoxMask(image, results); // Uncomment for mask drawing

    // Display the image
    cv::imshow("Detections", image);
    cv::waitKey(0); // Wait for a key press to close the window

    return 0;
}
