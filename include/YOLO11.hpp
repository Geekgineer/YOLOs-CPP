#pragma once

// ===================================
// Single YOLOv11 Detector Header File
// ===================================
//
// This header defines the YOLO11Detector class for performing object detection using the YOLOv11 model.
// It includes necessary libraries, utility structures, and helper functions to facilitate model inference
// and result postprocessing.
//
// Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
// Date: 29.09.2024
//
// ================================

/**
 * @file YOLO11Detector.hpp
 * @brief Header file for the YOLO11Detector class, responsible for object detection
 *        using the YOLOv11 model with optimized performance for minimal latency.
 */

// Include necessary ONNX Runtime and OpenCV headers
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include <unordered_map>
#include <thread>

// Include debug and custom ScopedTimer tools for performance measurement
#include "tools/Debug.hpp"
#include "tools/ScopedTimer.hpp"


/**
 * @brief Confidence threshold for filtering detections.
 */
const float CONFIDENCE_THRESHOLD = 0.4f;

/**
 * @brief  IoU threshold for filtering detections.
 */
const float IOU_THRESHOLD = 0.45f;


/**
 * @brief Struct to represent a bounding box.
 */

// Struct to represent a bounding box
struct BoundingBox {
    int x;
    int y;
    int width;
    int height;

    BoundingBox() : x(0), y(0), width(0), height(0) {}
    BoundingBox(int x_, int y_, int width_, int height_)
        : x(x_), y(y_), width(width_), height(height_) {}

    float area() const { return static_cast<float>(width * height); }

    BoundingBox intersect(const BoundingBox &other) const {
        int xStart = std::max(x, other.x);
        int yStart = std::max(y, other.y);
        int xEnd = std::min(x + width, other.x + other.width);
        int yEnd = std::min(y + height, other.y + other.height);
        int intersectWidth = std::max(0, xEnd - xStart);
        int intersectHeight = std::max(0, yEnd - yStart);
        return BoundingBox{xStart, yStart, intersectWidth, intersectHeight};
    }
};

/**
 * @brief Struct to represent a detection.
 */
struct Detection {
    BoundingBox box;
    float conf{};
    int classId{};
};

/**
 * @namespace utils
 * @brief Namespace containing utility functions for the YOLO11Detector.
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

        DEBUG_PRINT("Loaded " << classNames.size() << " class names from " + path);
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
     * @param p_Clip Whether to clip the coordinates to the image boundaries.
     * @return BoundingBox Scaled bounding box.
     */
    BoundingBox scaleCoords(const cv::Size &imageShape, BoundingBox coords,
                            const cv::Size &imageOriginalShape, bool p_Clip) {
        BoundingBox result;
        float gain = std::min(static_cast<float>(imageShape.height) / static_cast<float>(imageOriginalShape.height),
                              static_cast<float>(imageShape.width) / static_cast<float>(imageOriginalShape.width));

        int padX = static_cast<int>(std::round((imageShape.width - imageOriginalShape.width * gain) / 2.0f));
        int padY = static_cast<int>(std::round((imageShape.height - imageOriginalShape.height * gain) / 2.0f));

        result.x = static_cast<int>(std::round((coords.x - padX) / gain));
        result.y = static_cast<int>(std::round((coords.y - padY) / gain));
        result.width = static_cast<int>(std::round(coords.width / gain));
        result.height = static_cast<int>(std::round(coords.height / gain));

        if (p_Clip) {
            result.x = std::clamp(result.x, 0, imageOriginalShape.width);
            result.y = std::clamp(result.y, 0, imageOriginalShape.height);
            result.width = std::clamp(result.width, 0, imageOriginalShape.width - result.x);
            result.height = std::clamp(result.height, 0, imageOriginalShape.height - result.y);
        }
        return result;
    }

    /**
     * @brief Extracts the best class information from a detection row.
     * 
     * @param p_Mat Row data containing class scores.
     * @param numClasses Number of classes.
     * @param bestConf Reference to store the best confidence score.
     * @param bestClassId Reference to store the best class ID.
     */
    void getBestClassInfo(const std::vector<float> &p_Mat, const int &numClasses,
                          float &bestConf, int &bestClassId) 
                          
    {
        bestClassId = 0;
        bestConf = 0;

        for (int i = 0; i < numClasses; i++)
        {
            if (p_Mat[i + 4] > bestConf)
            {
                bestConf = p_Mat[i + 4];
                bestClassId = i;
            }
        }
    
    }

    /**
     * @brief Performs Non-Maximum Suppression (NMS) on the bounding boxes.
     * 
     * @param boundingBoxes Vector of bounding boxes.
     * @param scores Vector of confidence scores corresponding to each bounding box.
     * @param scoreThreshold Confidence threshold to filter boxes.
     * @param nmsThreshold IoU threshold for NMS.
     * @param indices Output vector of indices that survive NMS.
     */
    // Optimized Non-Maximum Suppression Function
    void NMSBoxes(const std::vector<BoundingBox>& boundingBoxes,
                const std::vector<float>& scores,
                float scoreThreshold,
                float nmsThreshold,
                std::vector<int>& indices)
    {
        indices.clear();

        const size_t numBoxes = boundingBoxes.size();
        if (numBoxes == 0) {
            DEBUG_PRINT("No bounding boxes to process in NMS");
            return;
        }

        // Step 1: Filter out boxes with scores below the threshold
        // and create a list of indices sorted by descending scores
        std::vector<int> sortedIndices;
        sortedIndices.reserve(numBoxes);
        for (size_t i = 0; i < numBoxes; ++i) {
            if (scores[i] >= scoreThreshold) {
                sortedIndices.push_back(static_cast<int>(i));
            }
        }

        // If no boxes remain after thresholding
        if (sortedIndices.empty()) {
            DEBUG_PRINT("No bounding boxes above score threshold");
            return;
        }

        // Sort the indices based on scores in descending order
        std::sort(sortedIndices.begin(), sortedIndices.end(),
                [&scores](int idx1, int idx2) {
                    return scores[idx1] > scores[idx2];
                });

        // Step 2: Precompute the areas of all boxes
        std::vector<float> areas(numBoxes, 0.0f);
        for (size_t i = 0; i < numBoxes; ++i) {
            areas[i] = boundingBoxes[i].width * boundingBoxes[i].height;
        }

        // Step 3: Suppression mask to mark boxes that are suppressed
        std::vector<bool> suppressed(numBoxes, false);

        // Step 4: Iterate through the sorted list and suppress boxes with high IoU
        for (size_t i = 0; i < sortedIndices.size(); ++i) {
            int currentIdx = sortedIndices[i];
            if (suppressed[currentIdx]) {
                continue;
            }

            // Select the current box as a valid detection
            indices.push_back(currentIdx);

            const BoundingBox& currentBox = boundingBoxes[currentIdx];
            const float x1_max = currentBox.x;
            const float y1_max = currentBox.y;
            const float x2_max = currentBox.x + currentBox.width;
            const float y2_max = currentBox.y + currentBox.height;
            const float area_current = areas[currentIdx];

            // Compare IoU of the current box with the rest
            for (size_t j = i + 1; j < sortedIndices.size(); ++j) {
                int compareIdx = sortedIndices[j];
                if (suppressed[compareIdx]) {
                    continue;
                }

                const BoundingBox& compareBox = boundingBoxes[compareIdx];
                const float x1 = std::max(x1_max, static_cast<float>(compareBox.x));
                const float y1 = std::max(y1_max, static_cast<float>(compareBox.y));
                const float x2 = std::min(x2_max, static_cast<float>(compareBox.x + compareBox.width));
                const float y2 = std::min(y2_max, static_cast<float>(compareBox.y + compareBox.height));

                const float interWidth = x2 - x1;
                const float interHeight = y2 - y1;

                if (interWidth <= 0 || interHeight <= 0) {
                    continue;
                }

                const float intersection = interWidth * interHeight;
                const float unionArea = area_current + areas[compareIdx] - intersection;
                const float iou = (unionArea > 0.0f) ? (intersection / unionArea) : 0.0f;

                if (iou > nmsThreshold) {
                    suppressed[compareIdx] = true;
                }
            }
        }

        DEBUG_PRINT("NMS completed with " + std::to_string(indices.size()) + " indices remaining");
    }


    /**
     * @brief Generates a vector of colors for each class name.
     * 
     * @param classNames Vector of class names.
     * @param seed Seed for random color generation to ensure reproducibility.
     * @return std::vector<cv::Scalar> Vector of colors.
     */
    inline std::vector<cv::Scalar> generateColors(const std::vector<std::string> &classNames, int seed = 42) {
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

    /**
     * @brief Draws bounding boxes and labels on the image based on detections.
     * 
     * @param image Image on which to draw.
     * @param detections Vector of detections.
     * @param classNames Vector of class names corresponding to object IDs.
     * @param colors Vector of colors for each class.
     */
    inline void drawBoundingBox(cv::Mat &image, const std::vector<Detection> &detections,
                                const std::vector<std::string> &classNames, const std::vector<cv::Scalar> &colors) {
        // Iterate through each detection to draw bounding boxes and labels
        for (const auto& detection : detections) {
            // Skip detections below the confidence threshold
            if (detection.conf <= CONFIDENCE_THRESHOLD)
                continue;

            // Ensure the object ID is within valid range
            if (detection.classId < 0 || static_cast<size_t>(detection.classId) >= classNames.size())
                continue;

            // Select color based on object ID for consistent coloring
            const cv::Scalar& color = colors[detection.classId % colors.size()];

            // Draw the bounding box rectangle
            cv::rectangle(image, cv::Point(detection.box.x, detection.box.y),
                          cv::Point(detection.box.x + detection.box.width, detection.box.y + detection.box.height),
                          color, 2, cv::LINE_AA);

            // Prepare label text with class name and confidence percentage
            std::string label = classNames[detection.classId] + ": " + std::to_string(static_cast<int>(detection.conf * 100)) + "%";

            // Define text properties for labels
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = std::min(image.rows, image.cols) * 0.0008;
            const int thickness = std::max(1, static_cast<int>(std::min(image.rows, image.cols) * 0.002));
            int baseline = 0;

            // Calculate text size for background rectangles
            cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);

            // Define positions for the label
            int labelY = std::max(detection.box.y, textSize.height + 5);
            cv::Point labelTopLeft(detection.box.x, labelY - textSize.height - 5);
            cv::Point labelBottomRight(detection.box.x + textSize.width + 5, labelY + baseline - 5);

            // Draw background rectangle for label
            cv::rectangle(image, labelTopLeft, labelBottomRight, color, cv::FILLED);

            // Put label text
            cv::putText(image, label, cv::Point(detection.box.x + 2, labelY - 2), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
        }
    }

    /**
     * @brief Draws bounding boxes and semi-transparent masks on the image based on detections.
     * 
     * @param image Image on which to draw.
     * @param detections Vector of detections.
     * @param classNames Vector of class names corresponding to object IDs.
     * @param classColors Vector of colors for each class.
     * @param maskAlpha Alpha value for the mask transparency.
     */
    inline void drawBoundingBoxMask(cv::Mat &image, const std::vector<Detection> &detections,
                                    const std::vector<std::string> &classNames, const std::vector<cv::Scalar> &classColors,
                                    float maskAlpha = 0.4f) {
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

        // Pre-filter detections to include only those above the confidence threshold and with valid class IDs
        std::vector<const Detection*> filteredDetections;
        for (const auto& detection : detections) {
            if (detection.conf > CONFIDENCE_THRESHOLD && 
                detection.classId >= 0 && 
                static_cast<size_t>(detection.classId) < classNames.size()) {
                filteredDetections.emplace_back(&detection);
            }
        }

        // Draw filled rectangles on the mask image for the semi-transparent overlay
        for (const auto* detection : filteredDetections) {
            cv::Rect box(detection->box.x, detection->box.y, detection->box.width, detection->box.height);
            const cv::Scalar &color = classColors[detection->classId];
            cv::rectangle(maskImage, box, color, cv::FILLED);
        }

        // Blend the maskImage with the original image to apply the semi-transparent masks
        cv::addWeighted(maskImage, maskAlpha, image, 1.0f, 0, image);

        // Draw bounding boxes and labels on the original image
        for (const auto* detection : filteredDetections) {
            cv::Rect box(detection->box.x, detection->box.y, detection->box.width, detection->box.height);
            const cv::Scalar &color = classColors[detection->classId];
            cv::rectangle(image, box, color, 2, cv::LINE_AA);

            std::string label = classNames[detection->classId] + ": " + std::to_string(static_cast<int>(detection->conf * 100)) + "%";
            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontSize, textThickness, &baseLine);

            int labelY = std::max(detection->box.y, labelSize.height + 5);
            cv::Point labelTopLeft(detection->box.x, labelY - labelSize.height - 5);
            cv::Point labelBottomRight(detection->box.x + labelSize.width + 5, labelY + baseLine - 5);

            // Draw background rectangle for label
            cv::rectangle(image, labelTopLeft, labelBottomRight, color, cv::FILLED);

            // Put label text
            cv::putText(image, label, cv::Point(detection->box.x + 2, labelY - 2), cv::FONT_HERSHEY_SIMPLEX, fontSize, cv::Scalar(255, 255, 255), textThickness, cv::LINE_AA);
        }

        DEBUG_PRINT("Bounding boxes and masks drawn on image.");
    }
};

/**
 * @brief YOLO11Detector class handles loading the YOLO model, preprocessing images, running inference, and postprocessing results.
 */
class YOLO11Detector {
public:
    /**
     * @brief Constructor to initialize the YOLO detector with model and label paths.
     * 
     * @param modelPath Path to the ONNX model file.
     * @param labelsPath Path to the file containing class labels.
     * @param useGPU Whether to use GPU for inference (default is false).
     */
    YOLO11Detector(const std::string &modelPath, const std::string &labelsPath, bool useGPU = false);
    
    /**
     * @brief Runs detection on the provided image.
     * 
     * @param image Input image for detection.
     * @param confThreshold Confidence threshold to filter detections (default is 0.4).
     * @param iouThreshold IoU threshold for Non-Maximum Suppression (default is 0.45).
     * @return std::vector<Detection> Vector of detections.
     */
    std::vector<Detection> detect(const cv::Mat &image, float confThreshold = 0.4f, float iouThreshold = 0.45f);
    
    /**
     * @brief Draws bounding boxes on the image based on detections.
     * 
     * @param image Image on which to draw.
     * @param detections Vector of detections.
     */
    void drawBoundingBox(cv::Mat &image, const std::vector<Detection> &detections) const {
        utils::drawBoundingBox(image, detections, classNames, classColors);
    }
    
    /**
     * @brief Draws bounding boxes and semi-transparent masks on the image based on detections.
     * 
     * @param image Image on which to draw.
     * @param detections Vector of detections.
     * @param maskAlpha Alpha value for mask transparency (default is 0.4).
     */
    void drawBoundingBoxMask(cv::Mat &image, const std::vector<Detection> &detections, float maskAlpha = 0.4f) const {
        utils::drawBoundingBoxMask(image, detections, classNames, classColors, maskAlpha);
    }

private:
    Ort::Env env{nullptr};                         // ONNX Runtime environment
    Ort::SessionOptions sessionOptions{nullptr};   // Session options for ONNX Runtime
    Ort::Session session{nullptr};                 // ONNX Runtime session for running inference
    bool isDynamicInputShape{};                    // Flag indicating if input shape is dynamic
    cv::Size inputImageShape;                      // Expected input image shape for the model

    // Vectors to hold allocated input and output node names
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<const char *> inputNames;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<const char *> outputNames;

    size_t numInputNodes, numOutputNodes;          // Number of input and output nodes in the model

    std::vector<std::string> classNames;            // Vector of class names loaded from file
    std::vector<cv::Scalar> classColors;            // Vector of colors for each class

    /**
     * @brief Preprocesses the input image for model inference.
     * 
     * @param image Input image.
     * @param blob Reference to pointer where preprocessed data will be stored.
     * @param inputTensorShape Reference to vector representing input tensor shape.
     * @return cv::Mat Resized image after preprocessing.
     */
    cv::Mat preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape);
    
    /**
     * @brief Postprocesses the model output to extract detections.
     * 
     * @param originalImageSize Size of the original input image.
     * @param resizedImageShape Size of the image after preprocessing.
     * @param outputTensors Vector of output tensors from the model.
     * @param confThreshold Confidence threshold to filter detections.
     * @param iouThreshold IoU threshold for Non-Maximum Suppression.
     * @return std::vector<Detection> Vector of detections.
     */
    std::vector<Detection> postprocess(const cv::Size &originalImageSize, const cv::Size &resizedImageShape,
                                      const std::vector<Ort::Value> &outputTensors,
                                      float confThreshold, float iouThreshold);
    
    /**
     * @brief Extracts the best class information from a detection row.
     * 
     * @param p_Mat Row data containing class scores.
     * @param numClasses Number of classes.
     * @param bestConf Reference to store the best confidence score.
     * @param bestClassId Reference to store the best class ID.
     */
    void getBestClassInfo(const std::vector<float> &p_Mat, const int &numClasses,
                          float &bestConf, int &bestClassId);
};

// Implementation of YOLO11Detector constructor
YOLO11Detector::YOLO11Detector(const std::string &modelPath, const std::string &labelsPath, bool useGPU) {
    // Initialize ONNX Runtime environment with warning level
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    // Set number of intra-op threads for parallelism
    sessionOptions.SetIntraOpNumThreads(std::min(6, static_cast<int>(std::thread::hardware_concurrency())));
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Retrieve available execution providers (e.g., CPU, CUDA)
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    // Configure session options based on whether GPU is to be used and available
    if (useGPU && cudaAvailable != availableProviders.end()) {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption); // Append CUDA execution provider
    } else {
        if (useGPU) {
            std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        }
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
    std::vector<int64_t> inputTensorShapeVec = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    isDynamicInputShape = (inputTensorShapeVec.size() >= 4) && (inputTensorShapeVec[2] == -1 && inputTensorShapeVec[3] == -1); // Check for dynamic dimensions

    // Allocate and store input node names
    auto input_name = session.GetInputNameAllocated(0, allocator);
    inputNodeNameAllocatedStrings.push_back(std::move(input_name));
    inputNames.push_back(inputNodeNameAllocatedStrings.back().get());

    // Allocate and store output node names
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    outputNodeNameAllocatedStrings.push_back(std::move(output_name));
    outputNames.push_back(outputNodeNameAllocatedStrings.back().get());

    // Set the expected input image shape based on the model's input tensor
    if (inputTensorShapeVec.size() >= 4) {
        inputImageShape = cv::Size(static_cast<int>(inputTensorShapeVec[3]), static_cast<int>(inputTensorShapeVec[2]));
    } else {
        throw std::runtime_error("Invalid input tensor shape.");
    }

    // Get the number of input and output nodes
    numInputNodes = session.GetInputCount();
    numOutputNodes = session.GetOutputCount();

    // Load class names and generate corresponding colors
    classNames = utils::getClassNames(labelsPath);
    classColors = utils::generateColors(classNames);

    std::cout << "Model loaded successfully with " << numInputNodes << " input nodes and " << numOutputNodes << " output nodes." << std::endl;
}

// Preprocess function implementation
cv::Mat YOLO11Detector::preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape) {
    ScopedTimer timer("preprocessing");

    cv::Mat resizedImage;
    // Resize and pad the image using letterBox utility
    utils::letterBox(image, resizedImage, inputImageShape, cv::Scalar(114, 114, 114), isDynamicInputShape, false, true, 32);

    // Update input tensor shape based on resized image dimensions
    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    // Convert image to float and normalize to [0, 1]
    resizedImage.convertTo(resizedImage, CV_32FC3, 1 / 255.0f);

    // Allocate memory for the image blob in CHW format
    blob = new float[resizedImage.cols * resizedImage.rows * resizedImage.channels()];

    // Split the image into separate channels and store in the blob
    std::vector<cv::Mat> chw(resizedImage.channels());
    for (int i = 0; i < resizedImage.channels(); ++i) {
        chw[i] = cv::Mat(resizedImage.rows, resizedImage.cols, CV_32FC1, blob + i * resizedImage.cols * resizedImage.rows);
    }
    cv::split(resizedImage, chw); // Split channels into the blob

    DEBUG_PRINT("Preprocessing completed")

    return resizedImage;
}

std::vector<Detection> YOLO11Detector::postprocess(
    const cv::Size &originalImageSize,
    const cv::Size &resizedImageShape,
    const std::vector<Ort::Value> &outputTensors,
    float confThreshold,
    float iouThreshold
) {
    ScopedTimer timer("postprocessing");

    std::vector<Detection> detections;

    // Retrieve raw output data from the first output tensor
    const float* rawOutput = outputTensors[0].GetTensorData<float>();
    const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const size_t num_features = outputShape[1];
    const size_t num_detections = outputShape[2];

    // Early exit if no detections
    if (num_detections == 0) {
        return detections;
    }

    const int numClasses = static_cast<int>(num_features) - 4;
    if (numClasses <= 0) {
        // Invalid number of classes
        return detections;
    }

    // Preallocate memory to avoid dynamic resizing
    std::vector<BoundingBox> boxes;
    boxes.reserve(num_detections);
    std::vector<float> confs;
    confs.reserve(num_detections);
    std::vector<int> classIds;
    classIds.reserve(num_detections);
    std::vector<BoundingBox> nms_boxes;
    nms_boxes.reserve(num_detections);

    // Constants for indexing
    const float* ptr = rawOutput;

    for (size_t d = 0; d < num_detections; ++d) {
        // Extract bounding box coordinates
        float centerX = ptr[0 * num_detections + d];
        float centerY = ptr[1 * num_detections + d];
        float width = ptr[2 * num_detections + d];
        float height = ptr[3 * num_detections + d];

        // Initialize object confidence and class ID
        float objConf = ptr[4 * num_detections + d];
        int classId = 0;

        // Find the class with the highest confidence
        for (int c = 1; c < numClasses; ++c) {
            float classConf = ptr[(4 + c) * num_detections + d];
            if (classConf > objConf) {
                objConf = classConf;
                classId = c;
            }
        }

        // Filter out low-confidence detections
        if (objConf > confThreshold) {
            // Convert center coordinates to top-left
            float left = centerX - width / 2.0f;
            float top = centerY - height / 2.0f;

            // Scale coordinates to original image size
            BoundingBox scaledBox = utils::scaleCoords(
                resizedImageShape,
                BoundingBox(left, top, width, height),
                originalImageSize,
                true
            );

            // Round coordinates and adjust for class ID to prevent NMS overlap between classes
            BoundingBox roundedBox;
            roundedBox.x = std::round(scaledBox.x);
            roundedBox.y = std::round(scaledBox.y);
            roundedBox.width = std::round(scaledBox.width);
            roundedBox.height = std::round(scaledBox.height);

            // Adjusted box for NMS to differentiate classes
            BoundingBox nmsBox = roundedBox;
            nmsBox.x += classId * 7680; // Assuming 7680 is larger than image width
            nmsBox.y += classId * 7680;

            // Append to NMS containers
            nms_boxes.emplace_back(nmsBox);
            boxes.emplace_back(roundedBox);
            confs.emplace_back(objConf);
            classIds.emplace_back(classId);
        }
    }

    // Perform Non-Maximum Suppression (NMS)
    std::vector<int> indices;
    utils::NMSBoxes(nms_boxes, confs, confThreshold, iouThreshold, indices);

    // Collect final detections
    detections.reserve(indices.size());
    for (const int idx : indices) {
        detections.emplace_back(Detection{
            boxes[idx],
            confs[idx],
            classIds[idx]
        });
    }

    return detections;
}

// Implementation of getBestClassInfo
void YOLO11Detector::getBestClassInfo(const std::vector<float> &p_Mat, const int &numClasses,
                                      float &bestConf, int &bestClassId) {
    bestClassId = 0;
    bestConf = 0.0f;

    for (int i = 0; i < numClasses; ++i) {
        float classConf = p_Mat[i + 4]; // Assuming first 4 elements are bbox coordinates
        if (classConf > bestConf) {
            bestConf = classConf;
            bestClassId = i;
        }
    }
}

// Detect function implementation
std::vector<Detection> YOLO11Detector::detect(const cv::Mat& image, float confThreshold, float iouThreshold) {
    ScopedTimer timer("Overall detection");

    float* blobPtr = nullptr; // Pointer to hold preprocessed image data
    // Define the shape of the input tensor (batch size, channels, height, width)
    std::vector<int64_t> inputTensorShape = {1, 3, inputImageShape.height, inputImageShape.width};

    // Preprocess the image and obtain a pointer to the blob
    cv::Mat preprocessedImage = preprocess(image, blobPtr, inputTensorShape);

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
    std::vector<Detection> detections = postprocess(image.size(), resizedImageShape, outputTensors, confThreshold, iouThreshold);

    return detections; // Return the vector of detections
}
