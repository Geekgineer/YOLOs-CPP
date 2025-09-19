/**
 * @file batch_image_inference.cpp
 * @brief Batch object detection on multiple images using YOLO models (v5, v7, v8, v9, v10, v11, v12).
 * 
 * This file implements batch object detection that utilizes YOLO models with dynamic batch input support.
 * The application loads multiple images, processes them in batch, and displays the results with bounding boxes.
 *
 * Usage Instructions:
 * 1. Compile the application with the necessary OpenCV and YOLO dependencies.
 * 2. Provide the model path and image folder or list of images as arguments.
 * 3. Run the executable to initiate batch object detection.
 *
 * Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
 * Date: 29.09.2024
 */

#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include "det/YOLO.hpp"

int main(int argc, char* argv[]){
    namespace fs = std::filesystem;
    std::string labelsPath = "../models/coco.names";
    std::string imagePath = "../data/";
    std::string modelPath = "../models/yolo11n.onnx";
    std::vector<std::string> imageFiles;

    if(argc > 1){
        modelPath = argv[1];
    }
    if(argc > 2){
        imagePath = argv[2];
        if(fs::is_directory(imagePath)){
            for(const auto& entry : fs::directory_iterator(imagePath)){
                if(entry.is_regular_file()){
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if(ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tiff" || ext == ".tif"){
                        imageFiles.push_back(fs::absolute(entry.path()).string());
                    }
                }
            }
            if(imageFiles.empty()){
                std::cerr << "No image files found in directory: " << imagePath << std::endl;
                return -1;
            }
        } else if(fs::is_regular_file(imagePath)){
            imageFiles.push_back(imagePath);
        } else {
            std::cerr << "Provided path is not a valid file or directory: " << imagePath << std::endl;
            return -1;
        }
    } else {
        std::cout << "Usage: " << argv[0] << " <model_path> <image_path_or_folder> [labels_path]\n";
        std::cout << "No image path provided. Using default directory: " << imagePath << std::endl;
        for(const auto& entry : fs::directory_iterator(imagePath)){
            if(entry.is_regular_file()){
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if(ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tiff" || ext == ".tif"){
                    imageFiles.push_back(fs::absolute(entry.path()).string());
                }
            }
        }
        if(imageFiles.empty()){
            std::cerr << "No image files found in default directory: " << imagePath << std::endl;
            return -1;
        }
    }
    if(argc > 3){
        labelsPath = argv[3];
    }

    // Load all images
    std::vector<cv::Mat> images;
    for(const auto& imgPath : imageFiles){
        cv::Mat img = cv::imread(imgPath);
        if(img.empty()){
            std::cerr << "Warning: Could not open or find image: " << imgPath << std::endl;
            continue;
        }
        images.push_back(img);
    }
    if(images.empty()){
        std::cerr << "No valid images to process." << std::endl;
        return -1;
    }

    bool isGPU = true; // Set to false for CPU processing
    YOLODetector detector(modelPath, labelsPath, isGPU);

    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& img : images)
        std::cout << "image size " << img.size() << std::endl;
    std::vector<std::vector<Detection>> batchResults = detector.detect(images, 0.45f);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now() - start);
    std::cout << "Batch detection completed in: " << duration.count() << " ms" << std::endl;

    for(size_t i = 0; i < batchResults.size(); ++i){
        std::cout << "\nImage: " << imageFiles[i] << std::endl;
        std::cout << "Number of detections: " << batchResults[i].size() << std::endl;
        for(size_t j = 0; j < batchResults[i].size(); ++j){
            const Detection& det = batchResults[i][j];
            std::cout << "Detection " << j << ": Class=" << det.classId
                      << ", Confidence=" << det.conf
                      << ", Box=(" << det.box.x << "," << det.box.y
                      << "," << det.box.width << "," << det.box.height << ")" << std::endl;
        }
        // Draw bounding boxes on the image
        detector.drawBoundingBox(images[i], batchResults[i]);
        cv::imshow("Detections - " + std::to_string(i), images[i]);
    }
    cv::waitKey(0);
    return 0;
}
