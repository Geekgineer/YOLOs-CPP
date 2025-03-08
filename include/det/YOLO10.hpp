#pragma once

// ===================================
// Single YOLOv10 Detector Header File
// ===================================
//
// This header defines the YOLO10Detector class for performing object detection using the YOLOv10 model.
// It includes necessary libraries, utility structures, and helper functions to facilitate model inference
// and result postprocessing.
//
// Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
// Date: 29.09.2024
//
// ================================

/**
 * @file YOLO10Detector.hpp
 * @brief Header file for the YOLO10Detector class, responsible for object detection
 *        using the YOLOv10 model with optimized performance for minimal latency.
 */


// Include necessary ONNX Runtime and OpenCV headers
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// Include standard libraries for various utilities
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <random>
#include <memory>
#include <thread>


// Include debug and custom ScopedTimer tools for performance measurement
#include "tools/Debug.hpp"
#include "tools/ScopedTimer.hpp"

/**
 * @brief Confidence threshold for filtering detections.
 */
const float CONFIDENCE_THRESHOLD = 0.6f;


/**
 * @brief Struct to represent a single detection with bounding box.
 */
struct Detection {
    int x1, x2, y1, y2; // Coordinates of the bounding box
    int obj_id;         // Object class ID
    float accuracy;     // Confidence score of the detection

    // Constructor to initialize all members
    Detection(int x1, int x2, int y1, int y2, int obj_id, float accuracy)
        : x1(x1), x2(x2), y1(y1), y2(y2), obj_id(obj_id), accuracy(accuracy) {}
};

/**
 * @namespace utils
 * @brief Namespace containing utility functions for the YOLO10Detector.
 */
namespace utils {
    /**
     * @brief Loads class names from a given file path.
     * 
     * @param path Path to the file containing class names.
     * @return std::vector<std::string> Vector of class names.
     */
    std::vector<std::string> getClassNames(const std::string &path) {
        std::vector<std::string> classNames;
        std::ifstream infile(path);

        if (infile) {
            std::string line;
            while (getline(infile, line)) {
                // Remove carriage return if present (for Windows compatibility)
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                classNames.emplace_back(line);
            }
        } else {
            std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
        }

        DEBUG_PRINT("Loaded class names from " + path);
        return classNames;
    }

    /**
     * @brief Computes the product of elements in a vector.
     * 
     * @param vector Vector of integers.
     * @return size_t Product of all elements.
     */
    size_t vectorProduct(const std::vector<int64_t> &vector) {
        return std::accumulate(vector.begin(), vector.end(), 1ull, std::multiplies<size_t>());
    }
    
    /**
     * @brief Resizes an image with letterboxing to maintain aspect ratio.
     * 
     * @param image Input image.
     * @param outImage Output resized and padded image.
     * @param newShape Desired output size.
     * @param color Padding color (default is gray).
     * @param auto_ Automatically adjust padding to be multiple of stride.
     * @param scaleFill Whether to scale to fill the new shape without keeping aspect ratio.
     * @param scaleUp Whether to allow scaling up of the image.
     * @param stride Stride size for padding alignment.
     */
    inline void letterBox(const cv::Mat& image, cv::Mat& outImage,
                        const cv::Size& newShape,
                        const cv::Scalar& color = cv::Scalar(114, 114, 114),
                        bool auto_ = true,
                        bool scaleFill = false,
                        bool scaleUp = true,
                        int stride = 32) {
        // Calculate the scaling ratio to fit the image within the new shape
        float ratio = std::min(static_cast<float>(newShape.height) / image.rows,
                            static_cast<float>(newShape.width) / image.cols);

        // Prevent scaling up if not allowed
        if (!scaleUp) {
            ratio = std::min(ratio, 1.0f);
        }

        // Calculate new dimensions after scaling
        int newUnpadW = static_cast<int>(std::round(image.cols * ratio));
        int newUnpadH = static_cast<int>(std::round(image.rows * ratio));

        // Calculate padding needed to reach the desired shape
        int dw = newShape.width - newUnpadW;
        int dh = newShape.height - newUnpadH;

        if (auto_) {
            // Ensure padding is a multiple of stride for model compatibility
            dw = (dw % stride) / 2;
            dh = (dh % stride) / 2;
        } else if (scaleFill) {
            // Scale to fill without maintaining aspect ratio
            newUnpadW = newShape.width;
            newUnpadH = newShape.height;
            ratio = std::min(static_cast<float>(newShape.width) / image.cols,
                            static_cast<float>(newShape.height) / image.rows);
            dw = 0;
            dh = 0;
        } else {
            // Evenly distribute padding on both sides
            // Calculate separate padding for left/right and top/bottom to handle odd padding
            int padLeft = dw / 2;
            int padRight = dw - padLeft;
            int padTop = dh / 2;
            int padBottom = dh - padTop;

            // Resize the image if the new dimensions differ
            if (image.cols != newUnpadW || image.rows != newUnpadH) {
                cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
            } else {
                // Avoid unnecessary copying if dimensions are the same
                outImage = image;
            }

            // Apply padding to reach the desired shape
            cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, color);
            return; // Exit early since padding is already applied
        }

        // Resize the image if the new dimensions differ
        if (image.cols != newUnpadW || image.rows != newUnpadH) {
            cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
        } else {
            // Avoid unnecessary copying if dimensions are the same
            outImage = image;
        }

        // Calculate separate padding for left/right and top/bottom to handle odd padding
        int padLeft = dw / 2;
        int padRight = dw - padLeft;
        int padTop = dh / 2;
        int padBottom = dh - padTop;

        // Apply padding to reach the desired shape
        cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, color);
    }


    /**
     * @brief Scales detection coordinates back to the original image size.
     * 
     * @param imageShape Shape of the resized image used for inference.
     * @param bbox Detection bounding box to be scaled.
     * @param imageOriginalShape Original image size before resizing.
     */
    void scaleResultCoordsToOriginal(const cv::Size &imageShape, Detection &bbox, const cv::Size &imageOriginalShape) {
        // Calculate the scaling factor used during preprocessing
        float gain = std::min(static_cast<float>(imageShape.width) / static_cast<float>(imageOriginalShape.width),
                              static_cast<float>(imageShape.height) / static_cast<float>(imageOriginalShape.height));

        // Calculate padding added during preprocessing
        int padX = static_cast<int>((imageShape.width - imageOriginalShape.width * gain) / 2.0f);
        int padY = static_cast<int>((imageShape.height - imageOriginalShape.height * gain) / 2.0f);

        // Adjust bounding box coordinates by removing padding and scaling
        bbox.x1 = static_cast<int>((bbox.x1 - padX) / gain);
        bbox.y1 = static_cast<int>((bbox.y1 - padY) / gain);
        bbox.x2 = static_cast<int>((bbox.x2 - padX) / gain);
        bbox.y2 = static_cast<int>((bbox.y2 - padY) / gain);
    }

    /**
     * @brief Draws bounding boxes and labels on the image based on detections.
     * 
     * @param image Image on which to draw.
     * @param detectionVector Vector of detections.
     * @param classNames Vector of class names corresponding to object IDs.
     * @param colors Vector of colors for each class.
     */
    inline void drawBoundingBox(cv::Mat &image, std::vector<Detection> &detectionVector, const std::vector<std::string> &classNames, const std::vector<cv::Scalar>& colors) {
        // Precompute the number of detections to iterate efficiently
        size_t numDetections = detectionVector.size();

        // Iterate through each detection to draw bounding boxes and labels
        for (size_t i = 0; i < numDetections; ++i) {
            const auto& detection = detectionVector[i];

            // Skip detections below the confidence threshold
            if (detection.accuracy <= CONFIDENCE_THRESHOLD)
                continue;

            // Ensure the object ID is within valid range
            if (detection.obj_id < 0 || static_cast<size_t>(detection.obj_id) >= classNames.size())
                continue;

            // Select color based on object ID for consistent coloring
            const cv::Scalar& color = colors[detection.obj_id % colors.size()];

            // Draw the bounding box rectangle
            cv::rectangle(image, cv::Point(detection.x1, detection.y1), cv::Point(detection.x2, detection.y2), color, 2, cv::LINE_AA);

            // Prepare label text with class name and confidence percentage
            const std::string& label = classNames[detection.obj_id];
            std::string confidenceText = std::to_string(static_cast<int>(detection.accuracy * 100)) + "%";

            // Define text properties for labels
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.5;
            int thickness = 1;
            int baseline = 0;

            // Calculate text size for background rectangles
            cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);
            cv::Size confidenceSize = cv::getTextSize(confidenceText, fontFace, fontScale, thickness, &baseline);

            // Define positions for the label and confidence text
            cv::Point textOrg(detection.x1, detection.y1 - 10);
            cv::Point confidenceOrg(detection.x1, detection.y2 + confidenceSize.height + 10);

            // Ensure text does not go out of image boundaries
            textOrg.y = std::max(textOrg.y, textSize.height);
            confidenceOrg.y = std::min(confidenceOrg.y, image.rows - 1);

            // Draw background rectangles for better text visibility
            cv::rectangle(image, textOrg + cv::Point(0, baseline),
                        textOrg + cv::Point(textSize.width, -textSize.height),
                        color, cv::FILLED);

            cv::rectangle(image, confidenceOrg + cv::Point(0, baseline),
                        confidenceOrg + cv::Point(confidenceSize.width, -confidenceSize.height),
                        color, cv::FILLED);

            // Render the class label text
            cv::putText(image, label, textOrg, fontFace, fontScale, cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);

            // Render the confidence percentage text
            cv::putText(image, confidenceText, confidenceOrg, fontFace, fontScale, cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);
        }
    }

    /**
     * @brief Generates a vector of colors for each class name.
     * 
     * @param classNames Vector of class names.
     * @param seed Seed for random color generation to ensure reproducibility.
     * @return const std::vector<cv::Scalar>& Reference to the vector of colors.
     */
    inline const std::vector<cv::Scalar>& generateColors(const std::vector<std::string>& classNames, int seed = 42) {
        // Static cache to store colors based on class names to avoid regenerating
        static std::unordered_map<size_t, std::vector<cv::Scalar>> colorCache;

        // Compute a hash key based on class names to identify unique class configurations
        size_t hashKey = 0;
        for (const auto& name : classNames) {
            hashKey ^= std::hash<std::string>{}(name) + 0x9e3779b9 + (hashKey << 6) + (hashKey >> 2);
        }

        // Check if colors for this class configuration are already cached
        auto it = colorCache.find(hashKey);
        if (it != colorCache.end()) {
            return it->second;
        }

        // Generate unique random colors for each class
        std::vector<cv::Scalar> colors;
        colors.reserve(classNames.size());

        std::mt19937 rng(seed); // Initialize random number generator with fixed seed
        std::uniform_int_distribution<int> uni(0, 255); // Define distribution for color values

        for (size_t i = 0; i < classNames.size(); ++i) {
            colors.emplace_back(cv::Scalar(uni(rng), uni(rng), uni(rng))); // Generate random BGR color
        }

        // Cache the generated colors for future use
        colorCache.emplace(hashKey, colors);

        return colorCache[hashKey];
    }

    inline void drawBoundingBoxMask(cv::Mat &image, const std::vector<Detection> &detections, const std::vector<std::string> &classNames,const std::vector<cv::Scalar>& classColors, float maskAlpha) {
        // Validate input image
        if (image.empty()) {
            std::cerr << "ERROR: Empty image provided to drawBoundingBoxMask." << std::endl;
            return;
        }

        const int imgHeight = image.rows;
        const int imgWidth = image.cols;

        // Precompute dynamic font size and thickness based on image dimensions
        const double fontSize = std::min(imgHeight, imgWidth) * 0.0006;
        const int textThickness = std::max(1, static_cast<int>(std::min(imgHeight, imgWidth) * 0.001));

        // Create a mask image for blending (initialized to zero)
        cv::Mat maskImage(image.size(), image.type(), cv::Scalar::all(0));

        // Precompute necessary data for parallel processing
        size_t numDetections = detections.size();
        std::vector<cv::Rect> validBoxes;
        validBoxes.reserve(numDetections); // Reserve space to avoid reallocations

        // Pre-filter detections to include only those above the confidence threshold and with valid class IDs
        // This reduces the workload in the parallel section
        std::vector<const Detection*> filteredDetections;
        filteredDetections.reserve(numDetections);
        for (const auto& detection : detections) {
            if (detection.accuracy > CONFIDENCE_THRESHOLD && 
                detection.obj_id >= 0 && 
                static_cast<size_t>(detection.obj_id) < classNames.size()) {
                filteredDetections.emplace_back(&detection);
            }
        }

        size_t validDetections = filteredDetections.size();

        // Parallel region for drawing mask rectangles
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < validDetections; ++i) {
            const Detection* detection = filteredDetections[i];
            int x1 = detection->x1;
            int y1 = detection->y1;
            int x2 = detection->x2;
            int y2 = detection->y2;
            int classId = detection->obj_id;

            // Calculate width and height from coordinates
            int width = x2 - x1;
            int height = y2 - y1;

            // Define the bounding box as a cv::Rect
            cv::Rect box(x1, y1, width, height);

            // Retrieve the color associated with the detected class
            const cv::Scalar &color = classColors[classId];

            // Draw filled rectangle on the mask image for the semi-transparent overlay
            cv::rectangle(maskImage, box, color, cv::FILLED);
        }

        // Blend the maskImage with the original image to apply the semi-transparent masks
        cv::addWeighted(maskImage, maskAlpha, image, 1.0, 0, image);

        // Parallel region for drawing bounding boxes and labels
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < validDetections; ++i) {
            const Detection* detection = filteredDetections[i];
            int x1 = detection->x1;
            int y1 = detection->y1;
            int x2 = detection->x2;
            int y2 = detection->y2;
            int classId = detection->obj_id;

            // Calculate width and height from coordinates
            int width = x2 - x1;
            int height = y2 - y1;

            // Define the bounding box as a cv::Rect
            cv::Rect box(x1, y1, width, height);

            // Retrieve the color associated with the detected class
            const cv::Scalar &color = classColors[classId];

            // Draw the bounding box on the original image with anti-aliased lines for smoothness
            cv::rectangle(image, box, color, 2, cv::LINE_AA);

            // Prepare the label with class name and confidence percentage
            // Preallocate a buffer for the label string to avoid dynamic memory allocation
            char labelBuffer[256];
            std::snprintf(labelBuffer, sizeof(labelBuffer), "%s: %.0f%%", classNames[classId].c_str(), detection->accuracy * 100.0f);
            std::string label(labelBuffer);

            // Determine the baseline for text placement
            int baseLine = 0;

            // Calculate the size of the label text for background rectangle
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontSize, textThickness, &baseLine);

            // Ensure the label does not go above the image
            int labelY = std::max(y1, labelSize.height + 5);

            // Define the top-left and bottom-right points for the label background rectangle
            cv::Point labelTopLeft(x1, labelY - labelSize.height - 5);
            cv::Point labelBottomRight(x1 + labelSize.width + 5, labelY + baseLine - 5);

            // Draw the background rectangle for the label for better visibility
            cv::rectangle(image, labelTopLeft, labelBottomRight, color, cv::FILLED);

            // Put the label text within the background rectangle
            cv::putText(image, label, cv::Point(x1 + 2, labelY - 2), cv::FONT_HERSHEY_SIMPLEX, fontSize, cv::Scalar(255, 255, 255), textThickness, cv::LINE_AA);
        }

        DEBUG_PRINT("Bounding boxes and masks drawn on image.");
    }

}

// YOLO10Detector class handles loading the YOLO model, preprocessing images, running inference, and postprocessing results
class YOLO10Detector {
public:
    /**
     * @brief Constructor to initialize the YOLO detector with model and label paths.
     * 
     * @param modelPath Path to the ONNX model file.
     * @param labelsPath Path to the file containing class labels.
     * @param useGPU Whether to use GPU for inference (default is false).
     */
    YOLO10Detector(const std::string &modelPath, const std::string &labelsPath, bool useGPU = false);

    /**
     * @brief Runs detection on the provided image.
     * 
     * @param image Input image for detection.
     * @return std::vector<Detection> Vector of detections.
     */
    std::vector<Detection> detect(const cv::Mat &image);

    /**
     * @brief Draws bounding boxes on the image based on detections.
     * 
     * @param image Image on which to draw.
     * @param detectionVector Vector of detections.
     */
    void drawBoundingBox(cv::Mat &image, std::vector<Detection> &detectionVector) {
        utils::drawBoundingBox(image, detectionVector, classNames, classColors);
    }

    void drawBoundingBoxMask(cv::Mat &image, const std::vector<Detection> &detections, float maskAlpha = 0.4)
    {
        utils::drawBoundingBoxMask(image, detections, classNames, classColors, maskAlpha);
    }


private:
    Ort::Env env{nullptr};                     // ONNX Runtime environment
    Ort::SessionOptions sessionOptions{nullptr}; // Session options for ONNX Runtime
    Ort::Session session{nullptr};             // ONNX Runtime session for running inference
    bool isDynamicInputShape{};                // Flag indicating if input shape is dynamic
    cv::Size inputImageShape;                  // Expected input image shape for the model

    // Vectors to hold allocated input and output node names
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<const char *> inputNames;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<const char *> outputNames;

    size_t numInputNodes, numOutputNodes;      // Number of input and output nodes in the model

    /**
     * @brief Preprocesses the input image for model inference.
     * 
     * @param image Input image.
     * @param blob Reference to pointer where preprocessed data will be stored.
     * @param inputTensorShape Reference to vector representing input tensor shape.
     * @return cv::Mat Resized and padded image.
     */
    cv::Mat preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape);

    /**
     * @brief Postprocesses the model output to extract detections.
     * 
     * @param originalImageSize Size of the original input image.
     * @param resizedImageShape Size of the image after preprocessing.
     * @param outputTensors Vector of output tensors from the model.
     * @return std::vector<Detection> Vector of detections.
     */
    std::vector<Detection> postprocess(const cv::Size &originalImageSize, const cv::Size &resizedImageShape, std::vector<Ort::Value> &outputTensors);

    std::vector<std::string> classNames;        // Vector of class names loaded from file
    std::vector<cv::Scalar> classColors;        // Vector of colors for each class
};

// Implementation of YOLO10Detector constructor
YOLO10Detector::YOLO10Detector(const std::string &modelPath, const std::string &labelsPath, bool useGPU) {
    // Initialize ONNX Runtime environment with warning level
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    // Set number of intra-op threads for parallelism
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetIntraOpNumThreads(std::min(6, (int) std::thread::hardware_concurrency()));
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Set the path for the optimized model file (optional)
    sessionOptions.SetOptimizedModelFilePath(modelPath.c_str());

    // Retrieve available execution providers (e.g., CPU, CUDA)
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    // Configure session options based on whether GPU is to be used and available
    if (useGPU && cudaAvailable == availableProviders.end()) {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    } else if (useGPU && cudaAvailable != availableProviders.end()) {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption); // Append CUDA execution provider
    } else {
        std::cout << "Inference device: CPU" << std::endl;
    }

    // Load the ONNX model into the session
#ifdef _WIN32
    std::wstring w_modelPath(modelPath.begin(), modelPath.end());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    // Retrieve input tensor shape information
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    isDynamicInputShape = inputTensorShape[2] == -1 && inputTensorShape[3] == -1; // Check for dynamic dimensions

    // Allocate and store input node names
    auto input_name = session.GetInputNameAllocated(0, allocator);
    inputNodeNameAllocatedStrings.push_back(std::move(input_name));
    inputNames.push_back(inputNodeNameAllocatedStrings.back().get());

    // Allocate and store output node names
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    outputNodeNameAllocatedStrings.push_back(std::move(output_name));
    outputNames.push_back(outputNodeNameAllocatedStrings.back().get());

    // Set the expected input image shape based on the model's input tensor
    inputImageShape = cv::Size(inputTensorShape[3], inputTensorShape[2]);

    // Get the number of input and output nodes
    numInputNodes = session.GetInputCount();
    numOutputNodes = session.GetOutputCount();

    // Load class names and generate corresponding colors
    classNames = utils::getClassNames(labelsPath);
    classColors = utils::generateColors(classNames);

    std::cout << "Model loaded successfully with " << numInputNodes << " input nodes and " << numOutputNodes << " output nodes." << std::endl;
}

// Preprocess function implementation
cv::Mat YOLO10Detector::preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape) {
    // Start timing the preprocessing step
    ScopedTimer timer("Preprocessing");

    cv::Mat resizedImage;
    // Resize and pad the image using letterBox utility
    utils::letterBox(image, resizedImage, inputImageShape, cv::Scalar(114, 114, 114), isDynamicInputShape, false, true, 32);

    // Update input tensor shape based on resized image dimensions
    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    // Convert image to float and normalize to [0, 1]
    resizedImage.convertTo(resizedImage, CV_32FC3, 1 / 255.0);
    
    // Allocate memory for the image blob in CHW format
    blob = new float[resizedImage.cols * resizedImage.rows * resizedImage.channels()];
    cv::Size floatImageSize{resizedImage.cols, resizedImage.rows};

    // Split the image into separate channels and store in the blob
    std::vector<cv::Mat> chw(resizedImage.channels());
    for (int i = 0; i < resizedImage.channels(); ++i) {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(resizedImage, chw); // Split channels into the blob

    DEBUG_PRINT("Preprocessing completed");

    return image;
}

// Postprocess function implementation
std::vector<Detection> YOLO10Detector::postprocess(const cv::Size &originalImageSize, const cv::Size &resizedImageShape, std::vector<Ort::Value> &outputTensors) {
    // Start timing the postprocessing step
    ScopedTimer timer("Postprocessing");

    // Retrieve raw output data from the first output tensor
    auto *rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    std::vector<Detection> detectionVector;

    // Assume the second dimension represents the number of detections
    int numDetections = outputShape[1];

    DEBUG_PRINT("Number of detections before filtering: " << numDetections);

    // Iterate through each detection and filter based on confidence threshold
    for (int i = 0; i < numDetections; i++) {
        float x1 = rawOutput[i * 6 + 0];
        float y1 = rawOutput[i * 6 + 1];
        float x2 = rawOutput[i * 6 + 2];
        float y2 = rawOutput[i * 6 + 3];
        float confidence = rawOutput[i * 6 + 4];
        int classId = static_cast<int>(rawOutput[i * 6 + 5]);

        if (confidence < CONFIDENCE_THRESHOLD)
            continue; // Skip detections below the confidence threshold

        // Create a Detection object with the raw data
        Detection detection(static_cast<int>(x1), static_cast<int>(x2), static_cast<int>(y1), static_cast<int>(y2), classId, confidence);

        // Scale detection coordinates back to the original image size
        utils::scaleResultCoordsToOriginal(resizedImageShape, detection, originalImageSize);

        // Add the detection to the vector
        detectionVector.push_back(detection);
    }

    DEBUG_PRINT("Number of detections: " << detectionVector.size());

    DEBUG_PRINT("Postprocessing completed");

    return detectionVector;
}

// Detect function implementation
std::vector<Detection> YOLO10Detector::detect(const cv::Mat& image) {
    // Start timing the overall detection process
    ScopedTimer timer("Overall detection"); // Commented out for performance

    float* blobPtr = nullptr; // Pointer to hold preprocessed image data
    // Define the shape of the input tensor (batch size, channels, height, width)
    std::vector<int64_t> inputTensorShape = {1, 3, inputImageShape.height, inputImageShape.width};

    // Preprocess the image and obtain a pointer to the blob
    cv::Mat inputImage = preprocess(image, blobPtr, inputTensorShape);

    // Compute the total number of elements in the input tensor
    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);

    // Create a vector from the blob data for ONNX Runtime input
    std::vector<float> inputTensorValues(blobPtr, blobPtr + inputTensorSize);

    delete[] blobPtr; // Free the allocated memory for the blob

    // Create an Ort memory info object (can be cached if used repeatedly)
    static Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create input tensor object using the preprocessed data
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensorValues.data(),
        inputTensorSize,
        inputTensorShape.data(),
        inputTensorShape.size()
    );

    // Run the inference session with the input tensor and retrieve output tensors
    std::vector<Ort::Value> outputTensors = session.Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        &inputTensor,
        numInputNodes,
        outputNames.data(),
        numOutputNodes
    );

    // Determine the resized image shape based on input tensor shape
    cv::Size resizedImageShape(static_cast<int>(inputTensorShape[3]), static_cast<int>(inputTensorShape[2]));

    // Postprocess the output tensors to obtain detections
    std::vector<Detection> detections = postprocess(image.size(), resizedImageShape, outputTensors);

    return detections; // Return the vector of detections
}
