#pragma once

// ===================================
// Single YOLOv7 Detector Header File
// ===================================
//
// This header defines the YOLO7Detector class for performing object detection using the YOLOv7 model.
// It includes necessary libraries, utility structures, and helper functions to facilitate model inference
// and result postprocessing.
//
// Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
// Date: 29.09.2024
//
// ================================

/**
 * @file YOLO7Detector.hpp
 * @brief Header file for the YOLO7Detector class, responsible for object detection
 *        using the YOLOv7 model with optimized performance for minimal latency.
 */

// Include necessary ONNX Runtime and OpenCV headers
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// Include standard libraries for various utilities
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

// Include debug and timing tools for performance measurement
#include "tools/Debug.hpp"
#include "tools/ScopedTimer.hpp"


/**
 * @brief Confidence threshold for filtering detections.
 */
const float CONFIDENCE_THRESHOLD = 0.6f;


/**
 * @brief Struct to represent a bounding box.
 */
struct BoundingBox {
    int x;
    int y;
    int width;
    int height;

    BoundingBox() : x(0), y(0), width(0), height(0) {}
    BoundingBox(int x_, int y_, int width_, int height_)
        : x(x_), y(y_), width(width_), height(height_) {}

    /**
     * @brief Calculates the area of the bounding box.
     * @return float Area of the bounding box.
     */
    float area() const { return static_cast<float>(width * height); }

    /**
     * @brief Calculates the intersection of this bounding box with another.
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
 * @brief Struct to represent a detection.
 */
struct Detection {
    BoundingBox box;    ///< Bounding box of the detection.
    float confidence;   ///< Confidence score of the detection.
    int classId;        ///< Class ID of the detected object.

    Detection() : box(), confidence(0.0f), classId(0) {}
    Detection(const BoundingBox& b, float conf, int id)
        : box(b), confidence(conf), classId(id) {}
};


/**
 * @namespace utils
 * @brief Namespace containing utility functions for the YOLO7Detector.
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
            while (std::getline(infile, line)) {
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
     * @param detection Detection bounding box to be scaled.
     * @param imageOriginalShape Original image size before resizing.
     */
    inline void scaleResultCoordsToOriginal(const cv::Size &imageShape, Detection &detection, const cv::Size &imageOriginalShape) {
        float gain = std::min(static_cast<float>(imageShape.width) / static_cast<float>(imageOriginalShape.width),
                              static_cast<float>(imageShape.height) / static_cast<float>(imageOriginalShape.height));

        int padX = static_cast<int>((imageShape.width - imageOriginalShape.width * gain) / 2.0f);
        int padY = static_cast<int>((imageShape.height - imageOriginalShape.height * gain) / 2.0f);

        detection.box.x = static_cast<int>((detection.box.x - padX) / gain);
        detection.box.y = static_cast<int>((detection.box.y - padY) / gain);
        detection.box.width = static_cast<int>(detection.box.width / gain);
        detection.box.height = static_cast<int>(detection.box.height / gain);
    }

    /**
     * @brief Draws bounding boxes on the image based on detections.
     * 
     * @param image Image on which to draw.
     * @param detections Vector of detections.
     * @param classNames Vector of class names corresponding to object IDs.
     */
    inline void drawBoundingBox(cv::Mat &image, const std::vector<Detection> &detections,
                                const std::vector<std::string> &classNames, const std::vector<cv::Scalar> &classColors) {
        // Precompute font parameters based on image size
        const double fontScale = std::min(image.rows, image.cols) * 0.0008;
        const int thickness = std::max(1, static_cast<int>(std::min(image.rows, image.cols) * 0.002));

        for (const auto &detection : detections) {
            if (detection.confidence > CONFIDENCE_THRESHOLD) {
                // Ensure classId is within range
                if (detection.classId < 0 || static_cast<size_t>(detection.classId) >= classNames.size())
                    continue;

                const cv::Scalar &color = classColors[detection.classId];

                // Draw bounding box
                cv::rectangle(image, cv::Point(detection.box.x, detection.box.y),
                              cv::Point(detection.box.x + detection.box.width, detection.box.y + detection.box.height),
                              color, thickness, cv::LINE_AA);

                // Prepare label with class name and confidence
                std::string label = classNames[detection.classId] + ": " + std::to_string(static_cast<int>(detection.confidence * 100)) + "%";

                // Calculate text size
                int baseline = 0;
                cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);

                // Position for the label
                int labelY = std::max(detection.box.y, textSize.height + 10);

                // Draw background rectangle for label
                cv::rectangle(image, cv::Point(detection.box.x, labelY - textSize.height - 10),
                              cv::Point(detection.box.x + textSize.width + 10, labelY + baseline - 10),
                              color, cv::FILLED);

                // Put label text
                cv::putText(image, label, cv::Point(detection.box.x + 5, labelY - 5),
                            cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
            }
        }
    }


    /**
     * @brief Generates a vector of colors for each class name.
     * 
     * @param classNames Vector of class names.
     * @param seed Seed for random color generation to ensure reproducibility.
     * @return std::vector<cv::Scalar> Vector of colors.
     */
    inline std::vector<cv::Scalar> generateColors(const std::vector<std::string> &classNames, int seed = 42) {
        std::vector<cv::Scalar> colors;
        if (!colors.empty())
            return colors; // Avoid redundant color generation

        std::mt19937 rng(seed); // Fixed seed for reproducibility
        std::uniform_int_distribution<int> uni(0, 255);

        colors.resize(classNames.size());
        for (size_t i = 0; i < classNames.size(); ++i) {
            colors[i] = cv::Scalar(uni(rng), uni(rng), uni(rng)); // Generate a unique random color for each class
        }

        return colors;
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
        if (image.empty()) {
            std::cerr << "ERROR: Empty image provided to drawBoundingBoxMask." << std::endl;
            return;
        }

        // Precompute font parameters based on image size
        const int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        const double fontScale = std::min(image.rows, image.cols) * 0.0008;
        const int thickness = std::max(1, static_cast<int>(std::min(image.rows, image.cols) * 0.002));

        // Create a mask image for blending
        cv::Mat maskImage(image.size(), image.type(), cv::Scalar::all(0));

        for (const auto &detection : detections) {
            if (detection.confidence > CONFIDENCE_THRESHOLD) {
                // Ensure classId is within range
                if (detection.classId < 0 || static_cast<size_t>(detection.classId) >= classNames.size())
                    continue;

                const cv::Scalar &color = classColors[detection.classId];

                // Draw filled rectangle on the mask
                cv::rectangle(maskImage, cv::Point(detection.box.x, detection.box.y),
                              cv::Point(detection.box.x + detection.box.width, detection.box.y + detection.box.height),
                              color, cv::FILLED, cv::LINE_AA);

                // Draw bounding box
                cv::rectangle(image, cv::Point(detection.box.x, detection.box.y),
                              cv::Point(detection.box.x + detection.box.width, detection.box.y + detection.box.height),
                              color, thickness, cv::LINE_AA);

                // Prepare label with class name and confidence
                std::string label = classNames[detection.classId] + ": " + std::to_string(static_cast<int>(detection.confidence * 100)) + "%";

                // Calculate text size
                int baseline = 0;
                cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);

                // Position for the label
                int labelY = std::max(detection.box.y, textSize.height + 10);

                // Draw background rectangle for label
                cv::rectangle(image, cv::Point(detection.box.x, labelY - textSize.height - 10),
                              cv::Point(detection.box.x + textSize.width + 10, labelY + baseline - 10),
                              color, cv::FILLED, cv::LINE_AA);

                // Put label text
                cv::putText(image, label, cv::Point(detection.box.x + 5, labelY - 5),
                            cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
            }
        }

        // Blend the maskImage with the original image to apply the semi-transparent masks
        cv::addWeighted(maskImage, maskAlpha, image, 1.0f, 0, image);
    }

} // namespace utils

// ============================================================================
 // Class: YOLO7Detector
// ============================================================================
/**
 * @class YOLO7Detector
 * @brief Class responsible for loading the YOLOv7 model, preprocessing images, running inference,
 *        and postprocessing results for object detection with optimized performance.
 */
class YOLO7Detector {
public:
    /**
     * @brief Constructor to initialize the YOLO7Detector with model and label paths.
     * 
     * @param modelPath Path to the ONNX model file.
     * @param labelsPath Path to the file containing class labels.
     * @param isGPU Whether to use GPU for inference.
     */
    YOLO7Detector(const std::string &modelPath, const std::string &labelsPath, bool isGPU);

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
    void drawBoundingBox(cv::Mat &image, const std::vector<Detection> &detectionVector) const {
        utils::drawBoundingBox(image, detectionVector, classNames, classColors);

    }

    /**
     * @brief Draws bounding boxes and semi-transparent masks on the image based on detections.
     * 
     * @param image Image on which to draw.
     * @param detectionVector Vector of detections.
     * @param maskAlpha Alpha value for mask transparency (default is 0.4).
     */
    void drawBoundingBoxMask(cv::Mat &image, const std::vector<Detection> &detectionVector, float maskAlpha = 0.4f) const {
        utils::drawBoundingBoxMask(image, detectionVector, classNames, classColors, maskAlpha);
    }

private:
    Ort::Env env;                                      ///< ONNX Runtime environment.
    Ort::SessionOptions sessionOptions;                ///< Session options for ONNX Runtime.
    Ort::Session session;                              ///< ONNX Runtime session for running inference.
    bool isDynamicInputShape;                         ///< Flag indicating if input shape is dynamic.
    cv::Size inputImageShape;                         ///< Expected input image shape for the model.

    // Vectors to hold allocated input and output node names.
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<const char *> inputNames;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<const char *> outputNames;

    size_t numInputNodes;                              ///< Number of input nodes in the model.
    size_t numOutputNodes;                             ///< Number of output nodes in the model.

    std::vector<std::string> classNames;               ///< Vector of class names loaded from file.
    std::vector<cv::Scalar> classColors;               ///< Vector of colors for each class.

    // Cached MemoryInfo to avoid recreating it for each inference
    Ort::MemoryInfo memoryInfo;

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
     * @return std::vector<Detection> Vector of detections.
     */
    std::vector<Detection> postprocess(const cv::Size &originalImageSize, const cv::Size &resizedImageShape, std::vector<Ort::Value> &outputTensors);
};

// ============================================================================
 // Implementation: YOLO7Detector
// ============================================================================

/**
 * @brief Constructor implementation for YOLO7Detector.
 */
YOLO7Detector::YOLO7Detector(const std::string &modelPath, const std::string &labelsPath, bool isGPU)
    : env(ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION"),
      sessionOptions(),
      session(nullptr),
      isDynamicInputShape(false),
      inputImageShape(),
      numInputNodes(0),
      numOutputNodes(0),
      classNames(),
      classColors(),
      memoryInfo(nullptr)
{
    // Configure session options
    sessionOptions.SetIntraOpNumThreads(std::min(6, static_cast<int>(std::thread::hardware_concurrency())));
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Retrieve available execution providers (e.g., CPU, CUDA)
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    // Configure session options based on whether GPU is to be used and available
    if (isGPU && cudaAvailable != availableProviders.end()) {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption); // Append CUDA execution provider
    } else {
        if (isGPU) {
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
    isDynamicInputShape = (inputTensorShapeVec.size() >= 4) && (inputTensorShapeVec[2] == -1 && inputTensorShapeVec[3] == -1);

    // Allocate and store input node names
    auto input_name = session.GetInputNameAllocated(0, allocator);
    inputNodeNameAllocatedStrings.emplace_back(std::move(input_name));
    inputNames.push_back(inputNodeNameAllocatedStrings.back().get());

    // Allocate and store output node names
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    outputNodeNameAllocatedStrings.emplace_back(std::move(output_name));
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

    // Cache the MemoryInfo object for future use
    memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::cout << "YOLO7Detector initialized with " << numInputNodes << " input nodes and "
              << numOutputNodes << " output nodes." << std::endl;
}

/**
 * @brief Preprocesses the input image for model inference.
 */
cv::Mat YOLO7Detector::preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape) {
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
    // Reuse a pre-allocated buffer if possible (Here, dynamically allocated for simplicity)
    blob = new float[resizedImage.cols * resizedImage.rows * resizedImage.channels()];

    // Split the image into separate channels and store in the blob
    std::vector<cv::Mat> chw(resizedImage.channels());
    for (int i = 0; i < resizedImage.channels(); ++i) {
        // Utilize OpenCV's efficient memory sharing to prevent unnecessary data copying
        chw[i] = cv::Mat(resizedImage.rows, resizedImage.cols, CV_32FC1, blob + i * resizedImage.cols * resizedImage.rows);
    }
    cv::split(resizedImage, chw); // Split channels into the blob

    DEBUG_PRINT("Preprocessing completed")
    return resizedImage;
}

/**
 * @brief Postprocesses the model output to extract detections.
 */
std::vector<Detection> YOLO7Detector::postprocess(const cv::Size &originalImageSize, const cv::Size &resizedImageShape, std::vector<Ort::Value> &outputTensors) {
    ScopedTimer timer("postprocessing");

    std::vector<Detection> detectionVector;

    // Retrieve raw output data from the first output tensor
    auto *rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);

    // Iterate through each detection
    for (int i = 0; i < outputShape[0]; i++) {
        float confidence = output[i * outputShape[1] + 0];
        float x1 = output[i * outputShape[1] + 1];
        float y1 = output[i * outputShape[1] + 2];
        float x2 = output[i * outputShape[1] + 3];
        float y2 = output[i * outputShape[1] + 4];
        int classPrediction = static_cast<int>(output[i * outputShape[1] + 5]);

        float accuracy = outputShape[1] > 6 ? output[i * outputShape[1] + 6] : confidence;

        Detection detection;
        detection.box.x = static_cast<int>(x1);
        detection.box.y = static_cast<int>(y1);
        detection.box.width = static_cast<int>(x2 - x1);
        detection.box.height = static_cast<int>(y2 - y1);
        detection.classId = classPrediction;
        detection.confidence = accuracy;

        // Scale detection coordinates back to original image size
        utils::scaleResultCoordsToOriginal(resizedImageShape, detection, originalImageSize);

        detectionVector.emplace_back(detection);
    }

    DEBUG_PRINT("Postprocessing completed")
    return detectionVector;
}

/**
 * @brief Runs detection on the provided image.
 */
std::vector<Detection> YOLO7Detector::detect(const cv::Mat &image) {
    ScopedTimer timer("Overall detection");

    float *blob = nullptr;
    std::vector<int64_t> inputTensorShape{1, 3, inputImageShape.height, inputImageShape.width};
    cv::Mat inputImage = preprocess(image, blob, inputTensorShape);

    // Compute the total number of elements in the input tensor
    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);

    // Create a vector from the blob data for ONNX Runtime input
    // To minimize latency, move semantics can be used if the blob is no longer needed
    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    // Free the allocated memory for the blob
    delete[] blob;

    // Create input tensor object using the preprocessed data
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensorValues.data(),
        inputTensorSize,
        inputTensorShape.data(),
        inputTensorShape.size()
    );

    // Run the inference session with the input tensor and retrieve output tensors
    // Using asynchronous inference could further reduce latency
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
