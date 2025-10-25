#pragma once

// =====================================
// Single Image Classifier Header File
// =====================================
//
// This header defines the YOLO11Classifier class for performing image classification
// using an ONNX model. It includes necessary libraries, utility structures,
// and helper functions to facilitate model inference and result interpretation.
//
// Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
// Date: 2025-05-15
//
// =====================================

/**
 * @file YOLO11CLASS.hpp
 * @brief Header file for the YOLO11Classifier class, responsible for image classification
 * using an ONNX model with optimized performance for minimal latency.
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
#include <iomanip> // For std::fixed and std::setprecision
#include <sstream> // For std::ostringstream

// #define DEBUG_MODE // Enable debug mode for detailed logging

// Include debug and custom ScopedTimer tools for performance measurement
// Assuming these are in a common 'tools' directory relative to this header
#include "tools/Debug.hpp"
#include "tools/ScopedTimer.hpp"

/**
 * @brief Struct to represent a classification result.
 */
struct ClassificationResult {
    int classId{-1};        // Predicted class ID, initialized to -1 for easier error checking
    float confidence{0.0f}; // Confidence score for the prediction
    std::string className{}; // Name of the predicted class

    ClassificationResult() = default;
    ClassificationResult(int id, float conf, std::string name)
        : classId(id), confidence(conf), className(std::move(name)) {}
};

/**
 * @namespace utils
 * @brief Namespace containing utility functions for the YOLO11Classifier.
 */
namespace utils {

    // ... (clamp, getClassNames, vectorProduct, preprocessImageToTensor, drawClassificationResult utilities remain the same as previous correct version) ...
    /**
     * @brief A robust implementation of a clamp function.
     * Restricts a value to lie within a specified range [low, high].
     */
    template <typename T>
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type
    inline clamp(const T &value, const T &low, const T &high)
    {
        T validLow = low < high ? low : high;
        T validHigh = low < high ? high : low;

        if (value < validLow) return validLow;
        if (value > validHigh) return validHigh;
        return value;
    }

    /**
     * @brief Loads class names from a given file path.
     */
    std::vector<std::string> getClassNames(const std::string &path) {
        std::vector<std::string> classNames;
        std::ifstream infile(path);

        if (infile) {
            std::string line;
            while (getline(infile, line)) {
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
     */
    size_t vectorProduct(const std::vector<int64_t> &vector) {
        if (vector.empty()) return 0;
        return std::accumulate(vector.begin(), vector.end(), 1LL, std::multiplies<int64_t>());
    }

    /**
     * @brief Prepares an image for model input by resizing and padding (letterboxing style) or simple resize.
     */
    inline void preprocessImageToTensor(const cv::Mat& image, cv::Mat& outImage,
                                      const cv::Size& targetShape,
                                      const cv::Scalar& color = cv::Scalar(0, 0, 0),
                                      bool scaleUp = true,
                                      const std::string& strategy = "resize") {
        if (image.empty()) {
            std::cerr << "ERROR: Input image to preprocessImageToTensor is empty." << std::endl;
            return;
        }

        if (strategy == "letterbox") {
            float r = std::min(static_cast<float>(targetShape.height) / image.rows,
                               static_cast<float>(targetShape.width) / image.cols);
            if (!scaleUp) {
                r = std::min(r, 1.0f);
            }
            int newUnpadW = static_cast<int>(std::round(image.cols * r));
            int newUnpadH = static_cast<int>(std::round(image.rows * r));

            cv::Mat resizedTemp;
            cv::resize(image, resizedTemp, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);

            int dw = targetShape.width - newUnpadW;
            int dh = targetShape.height - newUnpadH;

            int top = dh / 2;
            int bottom = dh - top;
            int left = dw / 2;
            int right = dw - left;

            cv::copyMakeBorder(resizedTemp, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
        } else { // Default to "resize"
            if (image.size() == targetShape) {
                outImage = image.clone();
            } else {
                cv::resize(image, outImage, targetShape, 0, 0, cv::INTER_LINEAR);
            }
        }
    }

    /**
     * @brief Draws the classification result on the image.
     */
    inline void drawClassificationResult(cv::Mat &image, const ClassificationResult &result,
                                         const cv::Point& position = cv::Point(10, 10),
                                         const cv::Scalar& textColor = cv::Scalar(0, 255, 0),
                                         double fontScaleMultiplier = 0.0008,
                                         const cv::Scalar& bgColor = cv::Scalar(0,0,0) ) {
        if (image.empty()) {
            std::cerr << "ERROR: Empty image provided to drawClassificationResult." << std::endl;
            return;
        }
        if (result.classId == -1) {
            DEBUG_PRINT("Skipping drawing due to invalid classification result.");
            return;
        }

        std::ostringstream ss;
        ss << result.className << ": " << std::fixed << std::setprecision(2) << result.confidence * 100 << "%";
        std::string text = ss.str();

        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = std::min(image.rows, image.cols) * fontScaleMultiplier;
        if (fontScale < 0.4) fontScale = 0.4;
        const int thickness = std::max(1, static_cast<int>(fontScale * 1.8));
        int baseline = 0;

        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
        baseline += thickness;

        cv::Point textPosition = position;
        if (textPosition.x < 0) textPosition.x = 0;
        if (textPosition.y < textSize.height) textPosition.y = textSize.height + 2; 

        cv::Point backgroundTopLeft(textPosition.x, textPosition.y - textSize.height - baseline / 3);
        cv::Point backgroundBottomRight(textPosition.x + textSize.width, textPosition.y + baseline / 2);
        
        backgroundTopLeft.x = utils::clamp(backgroundTopLeft.x, 0, image.cols -1);
        backgroundTopLeft.y = utils::clamp(backgroundTopLeft.y, 0, image.rows -1);
        backgroundBottomRight.x = utils::clamp(backgroundBottomRight.x, 0, image.cols -1);
        backgroundBottomRight.y = utils::clamp(backgroundBottomRight.y, 0, image.rows -1);

        cv::rectangle(image, backgroundTopLeft, backgroundBottomRight, bgColor, cv::FILLED);
        cv::putText(image, text, cv::Point(textPosition.x, textPosition.y), fontFace, fontScale, textColor, thickness, cv::LINE_AA);

        DEBUG_PRINT("Classification result drawn on image: " << text);
    }

}; // end namespace utils


/**
 * @brief YOLO11Classifier class handles loading the classification model,
 * preprocessing images, running inference, and postprocessing results.
 */
class YOLO11Classifier {
public:
    /**
     * @brief Constructor to initialize the classifier with model and label paths.
     */
    YOLO11Classifier(const std::string &modelPath, const std::string &labelsPath,
                    bool useGPU = false, const cv::Size& targetInputShape = cv::Size(224, 224));

    /**
     * @brief Runs classification on the provided image.
     */
    ClassificationResult classify(const cv::Mat &image);

    /**
     * @brief Draws the classification result on the image.
     */
    void drawResult(cv::Mat &image, const ClassificationResult &result,
                    const cv::Point& position = cv::Point(10, 10)) const {
        utils::drawClassificationResult(image, result, position);
    }

    cv::Size getInputShape() const { return inputImageShape_; } // CORRECTED
    bool isModelInputShapeDynamic() const { return isDynamicInputShape_; } // CORRECTED


private:
    Ort::Env env_{nullptr};
    Ort::SessionOptions sessionOptions_{nullptr};
    Ort::Session session_{nullptr};

    bool isDynamicInputShape_{};
    cv::Size inputImageShape_{};

    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings_{};
    std::vector<const char *> inputNames_{};
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings_{};
    std::vector<const char *> outputNames_{};

    size_t numInputNodes_{}, numOutputNodes_{};
    int numClasses_{0};

    std::vector<std::string> classNames_{};

    void preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape);
    ClassificationResult postprocess(const std::vector<Ort::Value> &outputTensors);
};

// Implementation of YOLO11Classifier constructor
YOLO11Classifier::YOLO11Classifier(const std::string &modelPath, const std::string &labelsPath,
                                 bool useGPU, const cv::Size& targetInputShape)
    : inputImageShape_(targetInputShape) {
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_CLASSIFICATION_ENV");
    sessionOptions_ = Ort::SessionOptions();

    sessionOptions_.SetIntraOpNumThreads(std::min(4, static_cast<int>(std::thread::hardware_concurrency())));
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption{};

    if (useGPU && cudaAvailable != availableProviders.end()) {
        DEBUG_PRINT("Attempting to use GPU for inference.");
        sessionOptions_.AppendExecutionProvider_CUDA(cudaOption);
    } else {
        if (useGPU) {
            std::cout << "Warning: GPU requested but CUDAExecutionProvider is not available. Falling back to CPU." << std::endl;
        }
        DEBUG_PRINT("Using CPU for inference.");
    }

#ifdef _WIN32
    std::wstring w_modelPath = std::wstring(modelPath.begin(), modelPath.end());
    session_ = Ort::Session(env_, w_modelPath.c_str(), sessionOptions_);
#else
    session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    numInputNodes_ = session_.GetInputCount();
    numOutputNodes_ = session_.GetOutputCount();

    if (numInputNodes_ == 0) throw std::runtime_error("Model has no input nodes.");
    if (numOutputNodes_ == 0) throw std::runtime_error("Model has no output nodes.");

    auto input_node_name = session_.GetInputNameAllocated(0, allocator);
    inputNodeNameAllocatedStrings_.push_back(std::move(input_node_name));
    inputNames_.push_back(inputNodeNameAllocatedStrings_.back().get());

    Ort::TypeInfo inputTypeInfo = session_.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> modelInputTensorShapeVec = inputTensorInfo.GetShape();

    if (modelInputTensorShapeVec.size() == 4) {
        isDynamicInputShape_ = (modelInputTensorShapeVec[2] == -1 || modelInputTensorShapeVec[3] == -1);
        DEBUG_PRINT("Model input tensor shape from metadata: "
                    << modelInputTensorShapeVec[0] << "x" << modelInputTensorShapeVec[1] << "x"
                    << modelInputTensorShapeVec[2] << "x" << modelInputTensorShapeVec[3]);

        if (!isDynamicInputShape_) {
            int modelH = static_cast<int>(modelInputTensorShapeVec[2]);
            int modelW = static_cast<int>(modelInputTensorShapeVec[3]);
            if (modelH != inputImageShape_.height || modelW != inputImageShape_.width) {
                std::cout << "Warning: Target preprocessing shape (" << inputImageShape_.height << "x" << inputImageShape_.width
                          << ") differs from model's fixed input shape (" << modelH << "x" << modelW << "). "
                          << "Image will be preprocessed to " << inputImageShape_.height << "x" << inputImageShape_.width << "."
                          << " Consider aligning these for optimal performance/accuracy." << std::endl;
            }
        } else {
            DEBUG_PRINT("Model has dynamic input H/W. Preprocessing to specified target: "
                        << inputImageShape_.height << "x" << inputImageShape_.width);
        }
    } else {
        std::cerr << "Warning: Model input tensor does not have 4 dimensions as expected (NCHW). Shape: [";
        for(size_t i=0; i<modelInputTensorShapeVec.size(); ++i) std::cerr << modelInputTensorShapeVec[i] << (i==modelInputTensorShapeVec.size()-1 ? "" : ", ");
        std::cerr << "]. Assuming dynamic shape and proceeding with target HxW: "
                  << inputImageShape_.height << "x" << inputImageShape_.width << std::endl;
        isDynamicInputShape_ = true;
    }

    auto output_node_name = session_.GetOutputNameAllocated(0, allocator);
    outputNodeNameAllocatedStrings_.push_back(std::move(output_node_name));
    outputNames_.push_back(outputNodeNameAllocatedStrings_.back().get());

    Ort::TypeInfo outputTypeInfo = session_.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputTensorShapeVec = outputTensorInfo.GetShape();
    
    if (!outputTensorShapeVec.empty()) {
        if (outputTensorShapeVec.size() == 2 && outputTensorShapeVec[0] > 0) { 
            numClasses_ = static_cast<int>(outputTensorShapeVec[1]);
        } else if (outputTensorShapeVec.size() == 1 && outputTensorShapeVec[0] > 0) { 
            numClasses_ = static_cast<int>(outputTensorShapeVec[0]);
        } else { 
            for (long long dim : outputTensorShapeVec) {
                if (dim > 1 && numClasses_ == 0) numClasses_ = static_cast<int>(dim); 
            }
             if (numClasses_ == 0 && !outputTensorShapeVec.empty()) numClasses_ = static_cast<int>(outputTensorShapeVec.back());
        }
    }

    if (numClasses_ > 0) {
        // CORRECTED SECTION for printing outputTensorShapeVec
        std::ostringstream oss_shape;
        oss_shape << "[";
        for(size_t i=0; i<outputTensorShapeVec.size(); ++i) {
            oss_shape << outputTensorShapeVec[i];
            if (i < outputTensorShapeVec.size() - 1) {
                oss_shape << ", ";
            }
        }
        oss_shape << "]";
        DEBUG_PRINT("Model predicts " << numClasses_ << " classes based on output shape: " << oss_shape.str());
        // END CORRECTED SECTION
    } else {
        std::cerr << "Warning: Could not reliably determine number of classes from output shape: [";
        for(size_t i=0; i<outputTensorShapeVec.size(); ++i) { // Directly print to cerr
            std::cerr << outputTensorShapeVec[i] << (i == outputTensorShapeVec.size() - 1 ? "" : ", ");
        }
        std::cerr << "]. Postprocessing might be incorrect or assume a default." << std::endl;
    }

    classNames_ = utils::getClassNames(labelsPath);
    if (numClasses_ > 0 && !classNames_.empty() && classNames_.size() != static_cast<size_t>(numClasses_)) {
        std::cerr << "Warning: Number of classes from model (" << numClasses_
                  << ") does not match number of labels in " << labelsPath
                  << " (" << classNames_.size() << ")." << std::endl;
    }
    if (classNames_.empty() && numClasses_ > 0) {
        std::cout << "Warning: Class names file is empty or failed to load. Predictions will use numeric IDs if labels are not available." << std::endl;
    }

    std::cout << "YOLO11Classifier initialized successfully. Model: " << modelPath << std::endl;
}

// ... (preprocess, postprocess, and classify methods remain the same as previous correct version) ...
void YOLO11Classifier::preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape) {
    ScopedTimer timer("Preprocessing (Ultralytics-style)");

    if (image.empty()) {
        throw std::runtime_error("Input image to preprocess is empty.");
    }

    cv::Mat processedImage;
    // 1. Resize to target input shape (e.g., 224x224).
    //    The color cv::Scalar(0,0,0) is for padding if 'letterbox' strategy were used; unused for 'resize'.
    utils::preprocessImageToTensor(image, processedImage, inputImageShape_, cv::Scalar(0, 0, 0), true, "resize");

    // 2. Convert BGR (OpenCV default) to RGB
    cv::Mat rgbImageMat; // Use a different name to avoid confusion if processedImage was already RGB
    cv::cvtColor(processedImage, rgbImageMat, cv::COLOR_BGR2RGB);

    // 3. Convert to float32. At this point, values are typically in [0, 255] range.
    cv::Mat floatRgbImage;
    rgbImageMat.convertTo(floatRgbImage, CV_32F);

    // Set the actual NCHW tensor shape for this specific input
    // Model expects NCHW: Batch=1, Channels=3 (RGB), Height, Width
    inputTensorShape = {1, 3, static_cast<int64_t>(floatRgbImage.rows), static_cast<int64_t>(floatRgbImage.cols)};

    if (static_cast<int>(inputTensorShape[2]) != inputImageShape_.height || static_cast<int>(inputTensorShape[3]) != inputImageShape_.width) {
        std::cerr << "CRITICAL WARNING: Preprocessed image dimensions (" << inputTensorShape[2] << "x" << inputTensorShape[3]
                  << ") do not match target inputImageShape_ (" << inputImageShape_.height << "x" << inputImageShape_.width
                  << ") after resizing! This indicates an issue in utils::preprocessImageToTensor or logic." << std::endl;
    }

    size_t tensorSize = utils::vectorProduct(inputTensorShape); // 1 * C * H * W
    blob = new float[tensorSize];

    // 4. Scale pixel values to [0.0, 1.0] and convert HWC to CHW
    int h = static_cast<int>(inputTensorShape[2]); // Height
    int w = static_cast<int>(inputTensorShape[3]); // Width
    int num_channels = static_cast<int>(inputTensorShape[1]); // Should be 3

    if (num_channels != 3) {
         delete[] blob; // Clean up allocated memory
         throw std::runtime_error("Expected 3 channels for image blob after RGB conversion, but tensor shape indicates: " + std::to_string(num_channels));
    }
    if (floatRgbImage.channels() != 3) {
        delete[] blob;
        throw std::runtime_error("Expected 3 channels in cv::Mat floatRgbImage, but got: " + std::to_string(floatRgbImage.channels()));
    }

    for (int c_idx = 0; c_idx < num_channels; ++c_idx) {      // Iterate over R, G, B channels
        for (int i = 0; i < h; ++i) {     // Iterate over rows (height)
            for (int j = 0; j < w; ++j) { // Iterate over columns (width)
                // floatRgbImage is HWC (i is row, j is col, c_idx is channel index in Vec3f)
                // floatRgbImage pixel values are float and in [0, 255] range.
                float pixel_value = floatRgbImage.at<cv::Vec3f>(i, j)[c_idx];

                // Scale to [0.0, 1.0]
                float scaled_pixel = pixel_value / 255.0f;

                // Store in blob (CHW format)
                blob[c_idx * (h * w) + i * w + j] = scaled_pixel;
            }
        }
    }

    DEBUG_PRINT("Preprocessing completed (RGB, scaled [0,1]). Actual input tensor shape: "
                << inputTensorShape[0] << "x" << inputTensorShape[1] << "x"
                << inputTensorShape[2] << "x" << inputTensorShape[3]);
}
ClassificationResult YOLO11Classifier::postprocess(const std::vector<Ort::Value> &outputTensors) {
    ScopedTimer timer("Postprocessing");

    if (outputTensors.empty()) {
        std::cerr << "Error: No output tensors for postprocessing." << std::endl;
        return {};
    }

    const float* rawOutput = outputTensors[0].GetTensorData<float>();
    if (!rawOutput) {
        std::cerr << "Error: rawOutput pointer is null." << std::endl;
        return {};
    }

    const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t numScores = utils::vectorProduct(outputShape);

    // Debug output shape
    std::ostringstream oss_shape;
    oss_shape << "Output tensor shape: [";
    for (size_t i = 0; i < outputShape.size(); ++i) {
        oss_shape << outputShape[i] << (i == outputShape.size() - 1 ? "" : ", ");
    }
    oss_shape << "]";
    DEBUG_PRINT(oss_shape.str());

    // Determine the effective number of classes
    int currentNumClasses = numClasses_ > 0 ? numClasses_ : static_cast<int>(classNames_.size());
    if (currentNumClasses <= 0) {
        std::cerr << "Error: No valid number of classes determined." << std::endl;
        return {};
    }

    // Debug first few raw scores
    std::ostringstream oss_scores;
    oss_scores << "First few raw scores: ";
    for (size_t i = 0; i < std::min(size_t(5), numScores); ++i) {
        oss_scores << rawOutput[i] << " ";
    }
    DEBUG_PRINT(oss_scores.str());

    // Find maximum score and its corresponding class
    int bestClassId = -1;
    float maxScore = -std::numeric_limits<float>::infinity();
    std::vector<float> scores(currentNumClasses);

    // Handle different output shapes
    if (outputShape.size() == 2 && outputShape[0] == 1) {
        // Case 1: [1, num_classes] shape
        for (int i = 0; i < currentNumClasses && i < static_cast<int>(outputShape[1]); ++i) {
            scores[i] = rawOutput[i];
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                bestClassId = i;
            }
        }
    } else if (outputShape.size() == 1 || (outputShape.size() == 2 && outputShape[0] > 1)) {
        // Case 2: [num_classes] shape or [batch_size, num_classes] shape (take first batch)
        for (int i = 0; i < currentNumClasses && i < static_cast<int>(numScores); ++i) {
            scores[i] = rawOutput[i];
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                bestClassId = i;
            }
        }
    }

    if (bestClassId == -1) {
        std::cerr << "Error: Could not determine best class ID." << std::endl;
        return {};
    }

    // Apply softmax to get probabilities
    float sumExp = 0.0f;
    std::vector<float> probabilities(currentNumClasses);
    
    // Compute softmax with numerical stability
    for (int i = 0; i < currentNumClasses; ++i) {
        probabilities[i] = std::exp(scores[i] - maxScore);
        sumExp += probabilities[i];
    }

    // Calculate final confidence
    float confidence = sumExp > 0 ? probabilities[bestClassId] / sumExp : 0.0f;

    // Get class name
    std::string className = "Unknown";
    if (bestClassId >= 0 && static_cast<size_t>(bestClassId) < classNames_.size()) {
        className = classNames_[bestClassId];
    } else if (bestClassId >= 0) {
        className = "ClassID_" + std::to_string(bestClassId);
    }

    DEBUG_PRINT("Best class ID: " << bestClassId << ", Name: " << className << ", Confidence: " << confidence);
    return ClassificationResult(bestClassId, confidence, className);
}

ClassificationResult YOLO11Classifier::classify(const cv::Mat& image) {
    ScopedTimer timer("Overall classification task");

    if (image.empty()) {
        std::cerr << "Error: Input image for classification is empty." << std::endl;
        return {};
    }

    float* blobPtr = nullptr;
    std::vector<int64_t> currentInputTensorShape; 

    try {
        preprocess(image, blobPtr, currentInputTensorShape);
    } catch (const std::exception& e) {
        std::cerr << "Exception during preprocessing: " << e.what() << std::endl;
        if (blobPtr) delete[] blobPtr;
        return {};
    }

    if (!blobPtr) {
        std::cerr << "Error: Preprocessing failed to produce a valid data blob." << std::endl;
        return {};
    }

    size_t inputTensorSize = utils::vectorProduct(currentInputTensorShape);
    if (inputTensorSize == 0) {
        std::cerr << "Error: Input tensor size is zero after preprocessing." << std::endl;
        delete[] blobPtr;
        return {};
    }

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        blobPtr, 
        inputTensorSize,
        currentInputTensorShape.data(),
        currentInputTensorShape.size()
    );

    delete[] blobPtr;
    blobPtr = nullptr;

    std::vector<Ort::Value> outputTensors;
    try {
        outputTensors = session_.Run(
            Ort::RunOptions{nullptr},
            inputNames_.data(),
            &inputTensor,
            numInputNodes_,
            outputNames_.data(),
            numOutputNodes_
        );
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Exception during Run(): " << e.what() << std::endl;
        return {};
    }

    if (outputTensors.empty()) {
        std::cerr << "Error: ONNX Runtime Run() produced no output tensors." << std::endl;
        return {};
    }

    try {
        return postprocess(outputTensors);
    } catch (const std::exception& e) {
        std::cerr << "Exception during postprocessing: " << e.what() << std::endl;
        return {};
    }
}