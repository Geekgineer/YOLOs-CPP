#pragma once

// ===================================
// Single YOLOv11 Pose Detector Header File
// ===================================
//
// This header defines the YOLO11PoseDetector class for performing human pose estimation using the YOLOv11 model.
// It includes necessary libraries, utility structures, and helper functions to facilitate model inference,
// keypoint detection, and result visualization.
//
// Authors: 
// 1- Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
// 2- Mohamed Samir, www.linkedin.com/in/mohamed-samir-7a730b237/
//
// Date: 16.03.2025
// ================================

/**
 * @file YOLO11-POSE.hpp
 * @brief Header file for the YOLO11PoseDetector class, responsible for human pose estimation
 *        using the YOLOv11 model with optimized performance for real-time applications.
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
 * @brief Struct representing a detected keypoint in pose estimation.
 *
 * This struct holds the x and y coordinates of a keypoint along with 
 * its confidence score, indicating the model's certainty in the prediction.
 */
struct KeyPoint {
    float x;         ///< X-coordinate of the keypoint
    float y;         ///< Y-coordinate of the keypoint
    float confidence; ///< Confidence score of the keypoint

    /**
     * @brief Constructor to initialize a KeyPoint.
     * 
     * @param x_ X-coordinate of the keypoint.
     * @param y_ Y-coordinate of the keypoint.
     * @param conf_ Confidence score of the keypoint.
     */
    KeyPoint(float x_ = 0, float y_ = 0, float conf_ = 0) 
        : x(x_), y(y_), confidence(conf_) {}
};

/**
 * @brief Struct representing a bounding box for object detection.
 *
 * Stores the coordinates and dimensions of a detected object within an image.
 */
struct BoundingBox {
    int x;      ///< X-coordinate of the top-left corner
    int y;      ///< Y-coordinate of the top-left corner
    int width;  ///< Width of the bounding box
    int height; ///< Height of the bounding box

    /**
     * @brief Default constructor initializing an empty bounding box.
     */
    BoundingBox() : x(0), y(0), width(0), height(0) {}

    /**
     * @brief Constructor to initialize a bounding box with given values.
     *
     * @param x_ X-coordinate of the top-left corner.
     * @param y_ Y-coordinate of the top-left corner.
     * @param width_ Width of the bounding box.
     * @param height_ Height of the bounding box.
     */
    BoundingBox(int x_, int y_, int width_, int height_)
        : x(x_), y(y_), width(width_), height(height_) {}
};

/**
 * @brief Struct representing a detected object in an image.
 *
 * This struct contains the bounding box, confidence score, class ID,
 * and keypoints (if applicable for pose estimation).
 */
struct Detection {
    BoundingBox box;           ///< Bounding box of the detected object
    float conf{};              ///< Confidence score of the detection
    int classId{};             ///< ID of the detected class
    std::vector<KeyPoint> keypoints; ///< List of keypoints (for pose estimation)
};


/**
 * @brief List of COCO skeleton connections for human pose estimation.
 *
 * Defines the connections between keypoints in a skeleton structure using
 * 0-based indices, which are standard in COCO pose datasets.
 */
const std::vector<std::pair<int, int>> POSE_SKELETON = {
    // Face connections
    {0,1}, {0,2}, {1,3}, {2,4},
    // Head-to-shoulder connections
    {3,5}, {4,6},
    // Arms
    {5,7}, {7,9}, {6,8}, {8,10},
    // Body
    {5,6}, {5,11}, {6,12}, {11,12},
    // Legs
    {11,13}, {13,15}, {12,14}, {14,16}
};


/**
 * @namespace utils
 * @brief Namespace containing utility functions for the YOLO11POSE-Detector.
 */
namespace utils {

    /**
     * @brief A robust implementation of a clamp function.
     *        Restricts a value to lie within a specified range [low, high].
     *
     * @tparam T The type of the value to clamp. Should be an arithmetic type (int, float, etc.).
     * @param value The value to clamp.
     * @param low The lower bound of the range.
     * @param high The upper bound of the range.
     * @return const T& The clamped value, constrained to the range [low, high].
     *
     * @note If low > high, the function swaps the bounds automatically to ensure valid behavior.
     */
    template <typename T>
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type
    inline clamp(const T &value, const T &low, const T &high)
    {
        // Ensure the range [low, high] is valid; swap if necessary
        T validLow = low < high ? low : high;
        T validHigh = low < high ? high : low;

        // Clamp the value to the range [validLow, validHigh]
        if (value < validLow)
            return validLow;
        if (value > validHigh)
            return validHigh;
        return value;
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
     * @brief Draws pose estimations including bounding boxes, keypoints, and skeleton
     * 
     * @param image Input/output image
     * @param detections Vector of pose detections
     * @param confidenceThreshold Minimum confidence to visualize
     * @param kptRadius Radius of keypoint circles
     * @param kptThreshold Minimum keypoint confidence to draw
     */
    inline void drawPoseEstimation(cv::Mat &image,
        const std::vector<Detection> &detections,
        float confidenceThreshold = 0.5,
        float kptThreshold = 0.5)
    {
        // Calculate dynamic sizes based on image dimensions
        const int min_dim = std::min(image.rows, image.cols);
        const float scale_factor = min_dim / 1280.0f;  // Reference 1280px size
    
        // Dynamic sizing parameters
        const int line_thickness = std::max(1, static_cast<int>(2 * scale_factor));
        const int kpt_radius = std::max(2, static_cast<int>(4 * scale_factor));
        const float font_scale = 0.5f * scale_factor;
        const int text_thickness = std::max(1, static_cast<int>(1 * scale_factor));
        const int text_offset = static_cast<int>(10 * scale_factor);
    
        // Define the Ultralytics pose palette (BGR format)
        // Original RGB values: [255,128,0], [255,153,51], [255,178,102], [230,230,0], [255,153,255],
        // [153,204,255], [255,102,255], [255,51,255], [102,178,255], [51,153,255],
        // [255,153,153], [255,102,102], [255,51,51], [153,255,153], [102,255,102],
        // [51,255,51], [0,255,0], [0,0,255], [255,0,0], [255,255,255]
        // Converted to BGR:
        static const std::vector<cv::Scalar> pose_palette = {
            cv::Scalar(0,128,255),    // 0
            cv::Scalar(51,153,255),   // 1
            cv::Scalar(102,178,255),  // 2
            cv::Scalar(0,230,230),    // 3
            cv::Scalar(255,153,255),  // 4
            cv::Scalar(255,204,153),  // 5
            cv::Scalar(255,102,255),  // 6
            cv::Scalar(255,51,255),   // 7
            cv::Scalar(255,178,102),  // 8
            cv::Scalar(255,153,51),   // 9
            cv::Scalar(153,153,255),  // 10
            cv::Scalar(102,102,255),  // 11
            cv::Scalar(51,51,255),    // 12
            cv::Scalar(153,255,153),  // 13
            cv::Scalar(102,255,102),  // 14
            cv::Scalar(51,255,51),    // 15
            cv::Scalar(0,255,0),      // 16
            cv::Scalar(255,0,0),      // 17
            cv::Scalar(0,0,255),      // 18
            cv::Scalar(255,255,255)   // 19
        };
    
        // Define per-keypoint color indices (for keypoints 0 to 16)
        static const std::vector<int> kpt_color_indices = {16,16,16,16,16,0,0,0,0,0,0,9,9,9,9,9,9};
        // Define per-limb color indices for each skeleton connection.
        // Make sure the number of entries here matches the number of pairs in POSE_SKELETON.
        static const std::vector<int> limb_color_indices = {9,9,9,9,7,7,7,0,0,0,0,0,16,16,16,16,16,16,16};
    
        // Loop through each detection
        for (const auto& detection : detections) {
            if (detection.conf < confidenceThreshold)
                continue;
    
            // Draw bounding box (optional – remove if you prefer only pose visualization)
            const auto& box = detection.box;
            cv::rectangle(image,
                cv::Point(box.x, box.y),
                cv::Point(box.x + box.width, box.y + box.height),
                cv::Scalar(0, 255, 0), // You can change the box color if desired
                line_thickness);
    
            // Prepare a vector to hold keypoint positions and validity flags.
            const size_t numKpts = detection.keypoints.size();
            std::vector<cv::Point> kpt_points(numKpts, cv::Point(-1, -1));
            std::vector<bool> valid(numKpts, false);
    
            // Draw keypoints using the corresponding palette colors
            for (size_t i = 0; i < numKpts; i++) {
                if (detection.keypoints[i].confidence >= kptThreshold) {
                    int x = std::round(detection.keypoints[i].x);
                    int y = std::round(detection.keypoints[i].y);
                    kpt_points[i] = cv::Point(x, y);
                    valid[i] = true;
                    int color_index = (i < kpt_color_indices.size()) ? kpt_color_indices[i] : 0;
                    cv::circle(image, cv::Point(x, y), kpt_radius, pose_palette[color_index], -1, cv::LINE_AA);
                }
            }
    
            // Draw skeleton connections based on a predefined POSE_SKELETON (vector of pairs)
            // Make sure that POSE_SKELETON is defined with 0-indexed keypoint indices.
            for (size_t j = 0; j < POSE_SKELETON.size(); j++) {
                auto [src, dst] = POSE_SKELETON[j];
                if (src < numKpts && dst < numKpts && valid[src] && valid[dst]) {
                    // Use the corresponding limb color from the palette
                    int limb_color_index = (j < limb_color_indices.size()) ? limb_color_indices[j] : 0;
                    cv::line(image, kpt_points[src], kpt_points[dst],
                             pose_palette[limb_color_index],
                             line_thickness, cv::LINE_AA);
                }
            }
    
            // (Optional) Add text labels such as confidence scores here if desired.
        }
    }  
}


/**
 * @brief YOLO11POSEDetector class handles loading the YOLO model, preprocessing images, running inference, and postprocessing results.
 */
class YOLO11POSEDetector {
public:
    /**
     * @brief Constructor to initialize the YOLO detector with model and label paths.
     * 
     * @param modelPath Path to the ONNX model file.
     * @param labelsPath Path to the file containing class labels.
     * @param useGPU Whether to use GPU for inference (default is false).
     */
    YOLO11POSEDetector(const std::string &modelPath, const std::string &labelsPath, bool useGPU = false);
    
    /**
     * @brief Runs detection on the provided image.
     * 
     * @param image Input image for detection.
     * @param confThreshold Confidence threshold to filter detections (default is 0.4).
     * @param iouThreshold IoU threshold for Non-Maximum Suppression (default is 0.45).
     * @return std::vector<Detection> Vector of detections.
     */
    std::vector<Detection> detect(const cv::Mat &image, float confThreshold = 0.4f, float iouThreshold = 0.5f);
 
    /**
     * @brief Draws bounding boxes and keypoints (if available) on the provided image.
     * 
     * @param image Input/output image on which detections will be visualized.
     * @param detections Vector of detected objects containing bounding boxes and keypoints.
     */
    void drawBoundingBox(cv::Mat &image, const std::vector<Detection> &detections) const;

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
    
};


// Implementation of YOLO11POSEDetector constructor
YOLO11POSEDetector::YOLO11POSEDetector(const std::string &modelPath, const std::string &labelsPath, bool useGPU) {
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


    std::cout << "Model loaded successfully with " << numInputNodes << " input nodes and " << numOutputNodes << " output nodes." << std::endl;
}

// Preprocess function implementation
cv::Mat YOLO11POSEDetector::preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape) {
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


/**
 * @brief Draws bounding boxes and pose keypoints (if available) on the image.
 * 
 * @param image Input/output image where detections will be drawn.
 * @param detections Vector of detected objects, including bounding boxes and keypoints.
 *
 * @note Uses `utils::drawPoseEstimation()` for visualization.
 */
void YOLO11POSEDetector::drawBoundingBox(cv::Mat &image, const std::vector<Detection> &detections) const {
    utils::drawPoseEstimation(image, detections);
}


/**
 * @brief Processes the raw output tensors from the YOLO model and extracts detections.
 *
 * @param originalImageSize The original size of the input image before resizing.
 * @param resizedImageShape The resized image dimensions used during inference.
 * @param outputTensors The output tensors obtained from the model inference.
 * @param confThreshold Confidence threshold to filter weak detections (default is 0.4).
 * @param iouThreshold IoU threshold for Non-Maximum Suppression (NMS) to remove redundant detections (default is 0.5).
 * @return std::vector<Detection> A vector of final detections after processing.
 *
 * @note This function applies confidence filtering, NMS, and necessary transformations 
 *       to map detections back to the original image size.
 */
std::vector<Detection> YOLO11POSEDetector::postprocess(
    const cv::Size &originalImageSize,
    const cv::Size &resizedImageShape,
    const std::vector<Ort::Value> &outputTensors,
    float confThreshold,
    float iouThreshold
) {
    ScopedTimer timer("postprocessing");
    std::vector<Detection> detections;
    
    const float* rawOutput = outputTensors[0].GetTensorData<float>();
    const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

    // Validate output dimensions
    const size_t numFeatures = outputShape[1];
    const size_t numDetections = outputShape[2];
    const int numKeypoints = 17;
    const int featuresPerKeypoint = 3;

    if (numFeatures != 4 + 1 + numKeypoints * featuresPerKeypoint) {
        std::cerr << "Invalid output shape for pose estimation model" << std::endl;
        return detections;
    }

    // Calculate letterbox padding parameters
    const float scale = std::min(static_cast<float>(resizedImageShape.width) / originalImageSize.width,
                static_cast<float>(resizedImageShape.height) / originalImageSize.height);
    const cv::Size scaledSize(originalImageSize.width * scale, originalImageSize.height * scale);
    const cv::Point2f padding((resizedImageShape.width - scaledSize.width) / 2.0f,
                            (resizedImageShape.height - scaledSize.height) / 2.0f);

    // Process each detection
    std::vector<BoundingBox> boxes;
    std::vector<float> confidences;
    std::vector<std::vector<KeyPoint>> allKeypoints;

    for (size_t d = 0; d < numDetections; ++d) {
        const float objConfidence = rawOutput[4 * numDetections + d];
        if (objConfidence < confThreshold) continue;

        // Decode bounding box
        const float cx = rawOutput[0 * numDetections + d];
        const float cy = rawOutput[1 * numDetections + d];
        const float w = rawOutput[2 * numDetections + d];
        const float h = rawOutput[3 * numDetections + d];
        
          // Convert to original image coordinates
          BoundingBox box;
          box.x = static_cast<int>((cx - padding.x - w / 2) / scale);
          box.y = static_cast<int>((cy - padding.y - h / 2) / scale);
          box.width = static_cast<int>(w / scale);
          box.height = static_cast<int>(h / scale);
  
          // Clip to image boundaries
          box.x = utils::clamp(box.x, 0, originalImageSize.width - box.width);
          box.y = utils::clamp(box.y, 0, originalImageSize.height - box.height);
          box.width = utils::clamp(box.width, 0, originalImageSize.width - box.x);
          box.height = utils::clamp(box.height, 0, originalImageSize.height - box.y);

        // Extract keypoints
        std::vector<KeyPoint> keypoints;
        for (int k = 0; k < numKeypoints; ++k) {
            const int offset = 5 + k * featuresPerKeypoint;
            KeyPoint kpt;
            kpt.x = (rawOutput[offset * numDetections + d] - padding.x) / scale;
            kpt.y = (rawOutput[(offset + 1) * numDetections + d] - padding.y) / scale;
            kpt.confidence = 1.0f / (1.0f + std::exp(-rawOutput[(offset + 2) * numDetections + d]));

            // Clip keypoints to image boundaries
            kpt.x = utils::clamp(kpt.x, 0.0f, static_cast<float>(originalImageSize.width - 1));
            kpt.y = utils::clamp(kpt.y, 0.0f, static_cast<float>(originalImageSize.height - 1));

            keypoints.emplace_back(kpt);
        }

        // Store detection components
        boxes.emplace_back(box);
        confidences.emplace_back(objConfidence);
        allKeypoints.emplace_back(keypoints);
    }

    // Apply Non-Maximum Suppression
    std::vector<int> indices;
    utils::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, indices);

    // Create final detections
    for (int idx : indices) {
        Detection det;
        // After (convert cv::Rect to BoundingBox)
        det.box.x = boxes[idx].x;
        det.box.y = boxes[idx].y;
        det.box.width = boxes[idx].width;
        det.box.height = boxes[idx].height;
        det.conf = confidences[idx];
        det.classId = 0; // Single class (person)
        det.keypoints = allKeypoints[idx];
        detections.emplace_back(det);
    }

    return detections;
}

// Detect function implementation
std::vector<Detection> YOLO11POSEDetector::detect(const cv::Mat& image, float confThreshold, float iouThreshold) {
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