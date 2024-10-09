#pragma once

// ==================================
// Single YOLOv5 Detector Header File
// ==================================
//
// This header defines the YOLO5Detector class for performing object detection using the YOLOv5 model.
// It includes necessary libraries, utility structures, and helper functions to facilitate model inference
// and result postprocessing.
//
// Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
// Date: 29.09.2024
//
// ================================

/**
 * @file YOLO5Detector.hpp
 * @brief Header file for the YOLO5Detector class, responsible for object detection
 *        using the YOLOv5 model with optimized performance for minimal latency.
 */

// Include necessary OpenCV and ONNX Runtime headers
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// Include standard libraries for various utilities
#include <algorithm>
#include <chrono>
#include <codecvt>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

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
 * @brief Struct representing a bounding box in an image.
 */
struct BoundingBox {
    int x;      /**< X-coordinate of the top-left corner */
    int y;      /**< Y-coordinate of the top-left corner */
    int width;  /**< Width of the bounding box */
    int height; /**< Height of the bounding box */

    /**
     * @brief Default constructor initializing all members to zero.
     */
    BoundingBox() : x(0), y(0), width(0), height(0) {}

    /**
     * @brief Parameterized constructor to initialize bounding box coordinates and size.
     * 
     * @param x_ X-coordinate of the top-left corner.
     * @param y_ Y-coordinate of the top-left corner.
     * @param width_ Width of the bounding box.
     * @param height_ Height of the bounding box.
     */
    BoundingBox(int x_, int y_, int width_, int height_)
        : x(x_), y(y_), width(width_), height(height_) {}

    /**
     * @brief Calculates the area of the bounding box.
     * 
     * @return float Area of the bounding box.
     */
    float area() const { return static_cast<float>(width * height); }

    /**
     * @brief Computes the intersection of this bounding box with another.
     * 
     * @param other The other bounding box to intersect with.
     * @return BoundingBox The intersected bounding box.
     */
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
 * @brief Struct representing a single detection result.
 */
struct Detection {
    BoundingBox box; /**< Bounding box of the detected object */
    float conf{};    /**< Confidence score of the detection */
    int classId{};   /**< Class ID of the detected object */
};

/**
 * @brief Namespace containing utility functions for the YOLO5Detector.
 */
namespace utils {

    /**
     * @brief Loads class names from a specified file path.
     * 
     * @param path Path to the file containing class names.
     * @return std::vector<std::string> Vector of class names.
     */
    inline std::vector<std::string> getClassNames(const std::string &path) {
        std::vector<std::string> classNames;
        std::ifstream infile(path);

        if (infile) {
            std::string line;
            while (std::getline(infile, line)) {
                // Remove carriage return for Windows compatibility
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
    inline size_t vectorProduct(const std::vector<int64_t> &vector) {
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
     * @param coords Detection bounding box to be scaled.
     * @param imageOriginalShape Original image size before resizing.
     * @param p_Clip Whether to clip the coordinates to the image boundaries.
     * @return BoundingBox Scaled bounding box.
     */
    inline BoundingBox scaleCoords(const cv::Size &imageShape, BoundingBox coords,
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
     * @brief Performs Non-Maximum Suppression (NMS) on the bounding boxes.
     * 
     * @param boundingBoxes Vector of bounding boxes.
     * @param scores Vector of confidence scores corresponding to each bounding box.
     * @param scoreThreshold Confidence threshold to filter boxes.
     * @param nmsThreshold IoU threshold for NMS.
     * @param indices Output vector of indices that survive NMS.
     */
    inline void NMSBoxes(std::vector<BoundingBox> &boundingBoxes, std::vector<float> &scores,
                         float scoreThreshold, float nmsThreshold, std::vector<int> &indices) {
        

        indices.clear();

        if (boundingBoxes.empty()) {
            return;
        }

        std::vector<int> idxs(boundingBoxes.size());
        std::iota(idxs.begin(), idxs.end(), 0);

        std::sort(idxs.begin(), idxs.end(), [&scores](int idx1, int idx2) {
            return scores[idx1] > scores[idx2];
        });

        while (!idxs.empty()) {
            int bestBoxIdx = idxs[0];

            if (scores[bestBoxIdx] < scoreThreshold) {
                idxs.erase(idxs.begin());
                continue;
            }

            indices.push_back(bestBoxIdx);
            auto it = idxs.begin() + 1;
            while (it != idxs.end()) {
                BoundingBox box1 = boundingBoxes[bestBoxIdx];
                BoundingBox box2 = boundingBoxes[*it];

                BoundingBox intersectBox = box1.intersect(box2);
                float intersectArea = intersectBox.area();
                float unionArea = box1.area() + box2.area() - intersectArea;
                float iou = unionArea <= 0.0f ? 0.0f : intersectArea / unionArea;

                if (iou > nmsThreshold) {
                    it = idxs.erase(it);
                } else {
                    ++it;
                }
            }

            idxs.erase(idxs.begin());
        }

        DEBUG_PRINT("NMS completed with " + std::to_string(indices.size()) + " indices remaining");
    }

    /**
     * @brief Generates a vector of unique colors for each class name.
     * 
     * @param classNames Vector of class names.
     * @param seed Seed for random color generation to ensure reproducibility.
     * @return std::vector<cv::Scalar> Vector of colors.
     */
    inline std::vector<cv::Scalar> generateColors(const std::vector<std::string> &classNames, int seed = 42) {
        std::vector<cv::Scalar> colors;
        std::mt19937 rng(seed); // Fixed seed for reproducibility
        std::uniform_int_distribution<int> uni(0, 255);

        colors.reserve(classNames.size());
        for (size_t i = 0; i < classNames.size(); ++i) {
            colors.emplace_back(cv::Scalar(uni(rng), uni(rng), uni(rng))); // Generate a unique random color for each class
        }
        return colors;
    }

    /**
     * @brief Draws bounding boxes and labels on the image based on detections.
     * 
     * @param image Image on which to draw.
     * @param detections Vector of detections.
     * @param classNames Vector of class names corresponding to object IDs.
     * @param classColors Vector of colors for each class.
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
     * @brief Draws bounding boxes with semi-transparent masks on the image based on detections.
     * 
     * @param image Image on which to draw.
     * @param detections Vector of detections.
     * @param classNames Vector of class names corresponding to object IDs.
     * @param classColors Vector of colors for each class.
     * @param maskAlpha Alpha value for mask transparency (default is 0.4).
     */
 inline void drawBoundingBoxMask(cv::Mat &image, const std::vector<Detection> &detections,
                                 const std::vector<std::string> &classNames, const std::vector<cv::Scalar> &classColors,
                                 float maskAlpha = 0.4f) {
        DEBUG_PRINT("Drawing bounding boxes with masks...");

        // Validate input image
        if (image.empty()) {
            std::cerr << "ERROR: Empty image provided to drawBoundingBoxMask." << std::endl;
            return;
        }

        const int imgHeight = image.rows;
        const int imgWidth = image.cols;

        // Precompute dynamic font size and thickness based on image dimensions
        const double fontSize = std::min(imgHeight, imgWidth) * 0.0008;
        const int textThickness = std::max(1, static_cast<int>(std::min(imgHeight, imgWidth) * 0.001));

        // Create a mask image for blending (initialized to zero)
        cv::Mat maskImage(image.size(), image.type(), cv::Scalar::all(0));

        // Precompute necessary data for parallel processing
        size_t numDetections = detections.size();
        
        // Parallel region for drawing mask rectangles
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < numDetections; ++i) {
            const auto& detection = detections[i];

            // Ensure class ID is within valid range
            if (detection.classId < 0 || static_cast<size_t>(detection.classId) >= classNames.size()) {
                std::cerr << "Invalid class ID: " << detection.classId << std::endl;
                continue;
            }

            // Convert BoundingBox to cv::Rect
            cv::Rect box(detection.box.x, detection.box.y, detection.box.width, detection.box.height);

            // Select color based on class ID
            const cv::Scalar& color = classColors[detection.classId];

            // Draw filled rectangle on the mask image for the semi-transparent overlay
            cv::rectangle(maskImage, box, color, cv::FILLED);
        }

        // Blend the maskImage with the original image to apply the semi-transparent masks
        cv::addWeighted(maskImage, maskAlpha, image, 1.0, 0, image);

        // Parallel region for drawing bounding boxes and labels
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < numDetections; ++i) {
            const auto& detection = detections[i];

            // Ensure class ID is within valid range
            if (detection.classId < 0 || static_cast<size_t>(detection.classId) >= classNames.size()) {
                std::cerr << "Invalid class ID: " << detection.classId << std::endl;
                continue;
            }

            // Convert BoundingBox to cv::Rect
            cv::Rect box(detection.box.x, detection.box.y, detection.box.width, detection.box.height);

            // Select color based on class ID
            const cv::Scalar& color = classColors[detection.classId];

            // Draw bounding box on the original image
            cv::rectangle(image, box, color, 2, cv::LINE_AA);

            // Prepare the label text with class name and confidence percentage
            std::string label = classNames[detection.classId] + ": " + std::to_string(static_cast<int>(detection.conf * 100)) + "%";

            // Determine the size of the label for background rectangle
            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontSize, textThickness, &baseLine);

            // Adjust y-coordinate to prevent label from going above the image
            int labelY = std::max(detection.box.y, labelSize.height + baseLine);

            // Define top-left and bottom-right points for the label background rectangle
            cv::Point labelTopLeft(detection.box.x, labelY - labelSize.height - baseLine);
            cv::Point labelBottomRight(detection.box.x + labelSize.width, labelY + baseLine - baseLine);

            // Draw filled rectangle behind the label for better visibility
            cv::rectangle(image, labelTopLeft, labelBottomRight, color, cv::FILLED);

            // Put the label text on the image
            cv::putText(image, label, cv::Point(detection.box.x, labelY), cv::FONT_HERSHEY_SIMPLEX,
                        fontSize, cv::Scalar(255, 255, 255), textThickness, cv::LINE_AA);
        }

        DEBUG_PRINT("Bounding boxes with masks drawn successfully.");
    }

}

/**
 * @brief YOLO5Detector class handles loading the YOLOv5 model, preprocessing images, running inference,
 *        and postprocessing results to perform object detection.
 */
class YOLO5Detector {
public:
    /**
     * @brief Constructor to initialize the YOLO5Detector with model and label paths.
     * 
     * @param modelPath Path to the ONNX model file.
     * @param labelsPath Path to the file containing class labels.
     * @param isGPU Flag indicating whether to use GPU for inference.
     */
    YOLO5Detector(const std::string &modelPath, const std::string &labelsPath, const bool &isGPU);

    /**
     * @brief Performs object detection on the provided image.
     * 
     * @param image Input image for detection.
     * @param confThreshold Confidence threshold to filter detections (default is 0.4).
     * @param iouThreshold IoU threshold for Non-Maximum Suppression (default is 0.45).
     * @return std::vector<Detection> Vector of detection results.
     */
    std::vector<Detection> detect(cv::Mat &image, const float &confThreshold = 0.4f, const float &iouThreshold = 0.45f);

    /**
     * @brief Draws bounding boxes on the image based on detections.
     * 
     * @param image Image on which to draw.
     * @param detections Vector of detections.
     */
    void drawBoundingBox(cv::Mat &image, const std::vector<Detection> &detections);

    /**
     * @brief Draws bounding boxes with semi-transparent masks on the image based on detections.
     * 
     * @param image Image on which to draw.
     * @param detections Vector of detections.
     * @param maskAlpha Alpha value for mask transparency (default is 0.4).
     */
    void drawBoundingBoxMask(cv::Mat &image, const std::vector<Detection> &detections, float maskAlpha = 0.4f);

private:
    int num_class;                /**< Number of classes */
    int num_proposal;             /**< Number of proposals */

    Ort::Env env{nullptr};        /**< ONNX Runtime environment */
    Ort::SessionOptions sessionOptions{nullptr}; /**< Session options for ONNX Runtime */
    Ort::Session session{nullptr}; /**< ONNX Runtime session for running inference */

    bool isDynamicInputShape{};   /**< Flag indicating if input shape is dynamic */
    std::vector<const char *> inputNames; /**< Names of input nodes */
    std::vector<const char *> outputNames; /**< Names of output nodes */

#if ORT_API_VERSION >= 13
    std::vector<std::string> inputNamesString; /**< Allocated input node names (for ORT API >= 13) */
    std::vector<std::string> outputNamesString; /**< Allocated output node names (for ORT API >= 13) */
#endif

    std::vector<std::vector<int64_t>> input_node_dims;  /**< Dimensions of input nodes */
    std::vector<std::vector<int64_t>> output_node_dims; /**< Dimensions of output nodes */
    cv::Size inputImageShape;                          /**< Expected input image shape */

    std::vector<std::string> classNames;               /**< Vector of class names */
    std::vector<cv::Scalar> classColors;               /**< Vector of colors for each class */

    /**
     * @brief Retrieves and stores input node details from the ONNX model.
     * 
     * @param allocator Allocator for handling memory.
     */
    void getInputDetails(Ort::AllocatorWithDefaultOptions &allocator);

    /**
     * @brief Retrieves and stores output node details from the ONNX model.
     * 
     * @param allocator Allocator for handling memory.
     */
    void getOutputDetails(Ort::AllocatorWithDefaultOptions &allocator);

    /**
     * @brief Preprocesses the input image for model inference.
     * 
     * @param image Input image.
     * @param blob Reference to pointer where preprocessed data will be stored.
     * @param inputTensorShape Reference to vector representing input tensor shape.
     */
    void preprocess(const cv::Mat &image, std::unique_ptr<float[]> &blob, std::vector<int64_t> &inputTensorShape);

    /**
     * @brief Postprocesses the model output to extract detections.
     * 
     * @param resizedImageShape Shape of the resized image used for inference.
     * @param originalImageShape Original image size before preprocessing.
     * @param outputTensors Vector of output tensors from the model.
     * @param confThreshold Confidence threshold to filter detections.
     * @param iouThreshold IoU threshold for Non-Maximum Suppression.
     * @return std::vector<Detection> Vector of detections.
     */
    std::vector<Detection> postprocessing(const cv::Size &resizedImageShape,
                                          const cv::Size &originalImageShape,
                                          std::vector<Ort::Value> &outputTensors,
                                          const float &confThreshold, const float &iouThreshold);

    /**
     * @brief Extracts the best class information from a detection row.
     * 
     * @param it Iterator pointing to the start of the detection row.
     * @param numClasses Number of classes.
     * @param bestConf Reference to store the best confidence score.
     * @param bestClassId Reference to store the best class ID.
     */
    void getBestClassInfo(std::vector<float>::iterator it, const int &numClasses,
                          float &bestConf, int &bestClassId);
};

/**
 * @brief Constructor implementation for YOLO5Detector.
 * 
 * Initializes the ONNX Runtime environment, loads the model, sets up session options,
 * and prepares class names and colors.
 * 
 * @param modelPath Path to the ONNX model file.
 * @param labelsPath Path to the file containing class labels.
 * @param isGPU Flag indicating whether to use GPU for inference.
 */
YOLO5Detector::YOLO5Detector(const std::string &modelPath, const std::string &labelsPath, const bool &isGPU) {
    // Initialize ONNX Runtime environment with warning level
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    // Set number of intra-op threads for parallelism
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetIntraOpNumThreads(std::min(6, static_cast<int>(std::thread::hardware_concurrency())));
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // To enable model serialization after graph optimization
    sessionOptions.SetOptimizedModelFilePath(modelPath.c_str());

    // Retrieve available execution providers (e.g., CPU, CUDA)
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    // Configure session options based on whether GPU is to be used and available
    if (isGPU && (cudaAvailable == availableProviders.end())) {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGPU && (cudaAvailable != availableProviders.end())) {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption); // Append CUDA execution provider
    }
    else {
        std::cout << "Inference device: CPU" << std::endl;
    }

#ifdef _WIN32
    // Convert model path to wide string for Windows compatibility
    std::wstring w_modelPath = std::wstring(modelPath.begin(), modelPath.end());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    // Load the ONNX model into the session
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    this->getInputDetails(allocator);
    this->getOutputDetails(allocator);

    // Set the expected input image shape based on the model's input tensor
    inputImageShape = cv::Size(input_node_dims[0][3], input_node_dims[0][2]);

    // Extract number of proposals and classes from output tensor dimensions
    this->num_proposal = output_node_dims[0][2];
    this->num_class = output_node_dims[0][1] - 5;
    std::cout << "Num proposals: " << this->num_proposal << std::endl;
    std::cout << "Class num: " << this->num_class << std::endl;

    // Load class names and generate corresponding colors
    classNames = utils::getClassNames(labelsPath);
    classColors = utils::generateColors(classNames); // Generate colors
}

/**
 * @brief Retrieves and stores input node details from the ONNX model.
 * 
 * @param allocator Allocator for handling memory.
 */
void YOLO5Detector::getInputDetails(Ort::AllocatorWithDefaultOptions &allocator) {
    for (int layer = 0; layer < this->session.GetInputCount(); ++layer) {
#if ORT_API_VERSION < 13
        inputNames.push_back(this->session.GetInputName(layer, allocator));
#else
        Ort::AllocatedStringPtr input_name_Ptr = this->session.GetInputNameAllocated(layer, allocator);
        inputNamesString.push_back(input_name_Ptr.get());
        inputNames.push_back(inputNamesString[layer].c_str());
#endif

        std::vector<int64_t> inputTensorShape = this->session.GetInputTypeInfo(layer).GetTensorTypeAndShapeInfo().GetShape();
        if (inputTensorShape.size() >= 4 && inputTensorShape[2] == -1 && inputTensorShape[3] == -1) {
            this->isDynamicInputShape = true;
        }
        input_node_dims.push_back(inputTensorShape);
    }

    DEBUG_PRINT("Input details obtained")
}

/**
 * @brief Retrieves and stores output node details from the ONNX model.
 * 
 * @param allocator Allocator for handling memory.
 */
void YOLO5Detector::getOutputDetails(Ort::AllocatorWithDefaultOptions &allocator) {
    for (int layer = 0; layer < this->session.GetOutputCount(); ++layer) {
#if ORT_API_VERSION < 13
        outputNames.push_back(this->session.GetOutputName(layer, allocator));
#else
        Ort::AllocatedStringPtr output_name_Ptr = this->session.GetOutputNameAllocated(layer, allocator);
        outputNamesString.push_back(output_name_Ptr.get());
        outputNames.push_back(outputNamesString[layer].c_str());
#endif

        auto outputTensorShape = this->session.GetOutputTypeInfo(layer).GetTensorTypeAndShapeInfo().GetShape();
        output_node_dims.push_back(outputTensorShape);
    }

    DEBUG_PRINT("Output details obtained")
}

/**
 * @brief Preprocesses the input image for model inference.
 * 
 * @param image Input image.
 * @param blob Reference to pointer where preprocessed data will be stored.
 * @param inputTensorShape Reference to vector representing input tensor shape.
 */
void YOLO5Detector::preprocess(const cv::Mat &image, std::unique_ptr<float[]> &blob, std::vector<int64_t> &inputTensorShape) {
    
    ScopedTimer timer("preprocessing");

    cv::Mat resizedImage;
    // Resize and pad the image using letterBox utility
    utils::letterBox(image, resizedImage, inputImageShape, cv::Scalar(114, 114, 114), isDynamicInputShape, false, true, 32);

    // Update input tensor shape based on resized image dimensions
    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    // Convert image to float and normalize to [0, 1]
    resizedImage.convertTo(resizedImage, CV_32F, 1 / 255.0f);

    // Allocate memory for the image blob in CHW format
    blob = std::make_unique<float[]>(resizedImage.cols * resizedImage.rows * resizedImage.channels());

#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < resizedImage.rows; ++h) {
            for (int w = 0; w < resizedImage.cols; ++w) {
                blob[c * resizedImage.cols * resizedImage.rows + h * resizedImage.cols + w] = resizedImage.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    DEBUG_PRINT("Preprocessing completed")

}

/**
 * @brief Performs object detection on the provided image.
 * 
 * @param image Input image for detection.
 * @param confThreshold Confidence threshold to filter detections.
 * @param iouThreshold IoU threshold for Non-Maximum Suppression.
 * @return std::vector<Detection> Vector of detection results.
 */
std::vector<Detection> YOLO5Detector::detect(cv::Mat &image, const float &confThreshold, const float &iouThreshold) {
    ScopedTimer timer("Overall detection");

    std::unique_ptr<float[]> blob;
    std::vector<int64_t> inputTensorShape{1, 3, -1, -1};

    // Preprocess the image and prepare the input tensor
    this->preprocess(image, blob, inputTensorShape);

    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);
    std::vector<float> inputTensorValues(blob.get(), blob.get() + inputTensorSize);

    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Create input tensor object using the preprocessed data
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputTensorShape.data(), inputTensorShape.size()));

    // Run the inference session with the input tensor and retrieve output tensors
    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                              inputNames.data(),
                                                              inputTensors.data(),
                                                              inputNames.size(),
                                                              outputNames.data(),
                                                              outputNames.size());

    // Determine the resized image shape based on input tensor shape
    cv::Size resizedShape = cv::Size(inputTensorShape[3], inputTensorShape[2]);

    // Postprocess the output tensors to obtain detections
    std::vector<Detection> result = this->postprocessing(resizedShape,
                                                         image.size(),
                                                         outputTensors,
                                                         confThreshold, iouThreshold);

    return result; // Return the vector of detections
}

/**
 * @brief Draws bounding boxes on the image based on detections.
 * 
 * @param image Image on which to draw.
 * @param detections Vector of detections.
 */
void YOLO5Detector::drawBoundingBox(cv::Mat &image, const std::vector<Detection> &detections) {
    utils::drawBoundingBox(image, detections, classNames, classColors);
}

/**
 * @brief Draws bounding boxes with semi-transparent masks on the image based on detections.
 * 
 * @param image Image on which to draw.
 * @param detections Vector of detections.
 * @param maskAlpha Alpha value for mask transparency.
 */
void YOLO5Detector::drawBoundingBoxMask(cv::Mat &image, const std::vector<Detection> &detections, float maskAlpha) {
    utils::drawBoundingBoxMask(image, detections, classNames, classColors, maskAlpha);
}

/**
 * @brief Postprocesses the model output to extract detections.
 * 
 * @param resizedImageShape Shape of the resized image used for inference.
 * @param originalImageShape Original image size before preprocessing.
 * @param outputTensors Vector of output tensors from the model.
 * @param confThreshold Confidence threshold to filter detections.
 * @param iouThreshold IoU threshold for Non-Maximum Suppression.
 * @return std::vector<Detection> Vector of detection results.
 */
std::vector<Detection> YOLO5Detector::postprocessing(const cv::Size &resizedImageShape,
                                                     const cv::Size &originalImageShape,
                                                     std::vector<Ort::Value> &outputTensors,
                                                     const float &confThreshold, const float &iouThreshold) {
    ScopedTimer timer("postprocessing");
    

    std::vector<Detection> detections;
    auto *rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);

    int numClasses = static_cast<int>(outputShape[2]) - 5;
    int elementsInBatch = static_cast<int>(outputShape[1] * outputShape[2]);

    for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2]) {
        float clsConf = it[4];

        if (clsConf > confThreshold) {
            int centerX = static_cast<int>(it[0]);
            int centerY = static_cast<int>(it[1]);
            int width = static_cast<int>(it[2]);
            int height = static_cast<int>(it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            float objConf;
            int classId;
            this->getBestClassInfo(it, numClasses, objConf, classId);

            float confidence = clsConf * objConf;

            BoundingBox box = BoundingBox(left, top, width, height);
            BoundingBox scaledBox = utils::scaleCoords(resizedImageShape, box, originalImageShape, true);

            Detection detection;
            detection.box = scaledBox;
            detection.conf = confidence;
            detection.classId = classId;

            detections.push_back(detection);
        }
    }

    // Extract BoundingBoxes and corresponding scores for NMS
    std::vector<BoundingBox> boundingBoxes;
    std::vector<float> scores;
    for (const auto &detection : detections) {
        boundingBoxes.push_back(detection.box);
        scores.push_back(detection.conf);
    }

    std::vector<int> nmsIndices;
    utils::NMSBoxes(boundingBoxes, scores, confThreshold, iouThreshold, nmsIndices);

    std::vector<Detection> finalDetections;
    for (int idx : nmsIndices) {
        finalDetections.push_back(detections[idx]);
    }

    return finalDetections;
}

/**
 * @brief Extracts the best class information from a detection row.
 * 
 * @param it Iterator pointing to the start of the detection row.
 * @param numClasses Number of classes.
 * @param bestConf Reference to store the best confidence score.
 * @param bestClassId Reference to store the best class ID.
 */
void YOLO5Detector::getBestClassInfo(std::vector<float>::iterator it, const int &numClasses,
                                     float &bestConf, int &bestClassId) {
    bestClassId = 5;
    bestConf = 0.0f;

    for (int i = 5; i < numClasses + 5; i++) {
        if (it[i] > bestConf) {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }
}
