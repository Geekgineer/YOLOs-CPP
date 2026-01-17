#pragma once

// ============================================================================
// YOLO Image Classification
// ============================================================================
// Image classification using YOLO models (v11, v12, YOLO26).
// Supports efficient classification with Ultralytics-style preprocessing.
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "yolos/core/version.hpp"
#include "yolos/core/utils.hpp"

namespace yolos {
namespace cls {

// ============================================================================
// Classification Result Structure
// ============================================================================

/// @brief Classification result containing class ID, confidence, and class name
struct ClassificationResult {
    int classId{-1};          ///< Predicted class ID
    float confidence{0.0f};   ///< Confidence score
    std::string className{};  ///< Human-readable class name

    ClassificationResult() = default;
    ClassificationResult(int id, float conf, std::string name)
        : classId(id), confidence(conf), className(std::move(name)) {}
};

// ============================================================================
// Drawing Utility for Classification
// ============================================================================

/// @brief Draw classification result on an image
/// @param image Image to draw on
/// @param result Classification result
/// @param position Position for the text
/// @param textColor Text color
/// @param bgColor Background color
inline void drawClassificationResult(cv::Mat& image,
                                     const ClassificationResult& result,
                                     const cv::Point& position = cv::Point(10, 30),
                                     const cv::Scalar& textColor = cv::Scalar(0, 255, 0),
                                     const cv::Scalar& bgColor = cv::Scalar(0, 0, 0)) {
    if (image.empty() || result.classId == -1) return;

    std::ostringstream ss;
    ss << result.className << ": " << std::fixed << std::setprecision(1) << result.confidence * 100 << "%";
    std::string text = ss.str();

    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = std::min(image.rows, image.cols) * 0.001;
    fontScale = std::max(fontScale, 0.5);
    int thickness = std::max(1, static_cast<int>(fontScale * 2));
    int baseline = 0;

    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

    cv::Point textPos = position;
    textPos.y = std::max(textPos.y, textSize.height + 5);

    cv::Point bgTopLeft(textPos.x - 2, textPos.y - textSize.height - 5);
    cv::Point bgBottomRight(textPos.x + textSize.width + 2, textPos.y + 5);

    bgTopLeft.x = utils::clamp(bgTopLeft.x, 0, image.cols - 1);
    bgTopLeft.y = utils::clamp(bgTopLeft.y, 0, image.rows - 1);
    bgBottomRight.x = utils::clamp(bgBottomRight.x, 0, image.cols - 1);
    bgBottomRight.y = utils::clamp(bgBottomRight.y, 0, image.rows - 1);

    cv::rectangle(image, bgTopLeft, bgBottomRight, bgColor, cv::FILLED);
    cv::putText(image, text, textPos, fontFace, fontScale, textColor, thickness, cv::LINE_AA);
}

// ============================================================================
// YOLOClassifier Base Class
// ============================================================================

/// @brief YOLO classifier for image classification
class YOLOClassifier {
public:
    /// @brief Constructor
    /// @param modelPath Path to the ONNX model file
    /// @param labelsPath Path to the class names file
    /// @param useGPU Whether to use GPU for inference
    /// @param targetInputShape Target input shape for preprocessing
    YOLOClassifier(const std::string& modelPath,
                   const std::string& labelsPath,
                   bool useGPU = false,
                   const cv::Size& targetInputShape = cv::Size(224, 224))
        : inputImageShape_(targetInputShape),
          env_(ORT_LOGGING_LEVEL_WARNING, "YOLOClassifier") {
        
        initSession(modelPath, useGPU);
        classNames_ = utils::getClassNames(labelsPath);
    }

    virtual ~YOLOClassifier() = default;

    /// @brief Run classification on an image
    /// @param image Input image (BGR format)
    /// @return Classification result
    ClassificationResult classify(const cv::Mat& image) {
        if (image.empty()) return {};

        // Preprocess
        std::vector<int64_t> inputTensorShape;
        preprocess(image, inputTensorShape);

        // Create input tensor
        size_t inputTensorSize = utils::vectorProduct(inputTensorShape);
        static Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputBuffer_.data(), inputTensorSize,
            inputTensorShape.data(), inputTensorShape.size());

        // Run inference
        std::vector<Ort::Value> outputTensors = session_.Run(
            Ort::RunOptions{nullptr},
            inputNames_.data(), &inputTensor, numInputNodes_,
            outputNames_.data(), numOutputNodes_);

        if (outputTensors.empty()) return {};

        return postprocess(outputTensors);
    }

    /// @brief Draw classification result on an image
    void drawResult(cv::Mat& image, const ClassificationResult& result,
                    const cv::Point& position = cv::Point(10, 30)) const {
        drawClassificationResult(image, result, position);
    }

    /// @brief Get input shape
    [[nodiscard]] cv::Size getInputShape() const { return inputImageShape_; }

    /// @brief Check if input shape is dynamic
    [[nodiscard]] bool isDynamicInputShape() const { return isDynamicInputShape_; }

    /// @brief Get class names
    [[nodiscard]] const std::vector<std::string>& getClassNames() const { return classNames_; }

protected:
    cv::Size inputImageShape_;
    Ort::Env env_{nullptr};
    Ort::SessionOptions sessionOptions_{nullptr};
    Ort::Session session_{nullptr};
    bool isDynamicInputShape_{false};
    std::vector<float> inputBuffer_;

    std::vector<Ort::AllocatedStringPtr> inputNameAllocs_;
    std::vector<const char*> inputNames_;
    std::vector<Ort::AllocatedStringPtr> outputNameAllocs_;
    std::vector<const char*> outputNames_;

    size_t numInputNodes_{0};
    size_t numOutputNodes_{0};
    int numClasses_{0};
    std::vector<std::string> classNames_;

    void initSession(const std::string& modelPath, bool useGPU) {
        sessionOptions_ = Ort::SessionOptions();
        sessionOptions_.SetIntraOpNumThreads(std::min(4, static_cast<int>(std::thread::hardware_concurrency())));
        sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        std::vector<std::string> providers = Ort::GetAvailableProviders();
        if (useGPU && std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end()) {
            OrtCUDAProviderOptions cudaOptions{};
            sessionOptions_.AppendExecutionProvider_CUDA(cudaOptions);
            std::cout << "[INFO] Classification using GPU (CUDA)" << std::endl;
        } else {
            std::cout << "[INFO] Classification using CPU" << std::endl;
        }

#ifdef _WIN32
        std::wstring wModelPath(modelPath.begin(), modelPath.end());
        session_ = Ort::Session(env_, wModelPath.c_str(), sessionOptions_);
#else
        session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);
#endif

        Ort::AllocatorWithDefaultOptions allocator;

        numInputNodes_ = session_.GetInputCount();
        numOutputNodes_ = session_.GetOutputCount();

        // Input node
        auto inputName = session_.GetInputNameAllocated(0, allocator);
        inputNameAllocs_.push_back(std::move(inputName));
        inputNames_.push_back(inputNameAllocs_.back().get());

        Ort::TypeInfo inputTypeInfo = session_.GetInputTypeInfo(0);
        std::vector<int64_t> inputShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        if (inputShape.size() == 4) {
            isDynamicInputShape_ = (inputShape[2] == -1 || inputShape[3] == -1);
            if (!isDynamicInputShape_) {
                inputImageShape_ = cv::Size(static_cast<int>(inputShape[3]), static_cast<int>(inputShape[2]));
            }
        }

        // Output node
        auto outputName = session_.GetOutputNameAllocated(0, allocator);
        outputNameAllocs_.push_back(std::move(outputName));
        outputNames_.push_back(outputNameAllocs_.back().get());

        Ort::TypeInfo outputTypeInfo = session_.GetOutputTypeInfo(0);
        std::vector<int64_t> outputShape = outputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        if (outputShape.size() >= 2) {
            numClasses_ = static_cast<int>(outputShape.back());
        } else if (outputShape.size() == 1) {
            numClasses_ = static_cast<int>(outputShape[0]);
        }

        std::cout << "[INFO] Classification model loaded: " << modelPath << std::endl;
        std::cout << "[INFO] Input shape: " << inputImageShape_.width << "x" << inputImageShape_.height << std::endl;
        std::cout << "[INFO] Number of classes: " << numClasses_ << std::endl;
    }

    /// @brief Preprocess image for classification (Ultralytics-style)
    void preprocess(const cv::Mat& image, std::vector<int64_t>& inputTensorShape) {
        int targetSize = inputImageShape_.width;
        int h = image.rows;
        int w = image.cols;

        // Resize: shortest side to target_size, maintaining aspect ratio
        // Use truncation (not round) to match torchvision.transforms.Resize behavior
        int newH, newW;
        if (h < w) {
            newH = targetSize;
            newW = static_cast<int>(w * targetSize / h);  // Truncate like Python int()
        } else {
            newW = targetSize;
            newH = static_cast<int>(h * targetSize / w);  // Truncate like Python int()
        }

        cv::Mat rgbImage;
        cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);

        cv::Mat resized;
        cv::resize(rgbImage, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

        // Center crop to target_size x target_size
        int yStart = std::max(0, (newH - targetSize) / 2);
        int xStart = std::max(0, (newW - targetSize) / 2);
        cv::Mat cropped = resized(cv::Rect(xStart, yStart, targetSize, targetSize));

        // Normalize to [0, 1]
        cv::Mat floatImage;
        cropped.convertTo(floatImage, CV_32F, 1.0 / 255.0);

        inputTensorShape = {1, 3, floatImage.rows, floatImage.cols};
        const int finalH = floatImage.rows;
        const int finalW = floatImage.cols;
        size_t tensorSize = 3 * finalH * finalW;
        inputBuffer_.resize(tensorSize);

        // Convert HWC to CHW format
        std::vector<cv::Mat> channels(3);
        cv::split(floatImage, channels);
        for (int c = 0; c < 3; ++c) {
            std::memcpy(inputBuffer_.data() + c * finalH * finalW,
                       channels[c].ptr<float>(), finalH * finalW * sizeof(float));
        }
    }

    /// @brief Postprocess classification output
    ClassificationResult postprocess(const std::vector<Ort::Value>& outputTensors) {
        const float* rawOutput = outputTensors[0].GetTensorData<float>();
        const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        int numScores = numClasses_ > 0 ? numClasses_ : static_cast<int>(classNames_.size());
        if (numScores <= 0) return {};

        // Find max score (YOLO classification ONNX export includes softmax, outputs are probabilities)
        int bestClassId = 0;
        float maxProb = rawOutput[0];

        for (int i = 1; i < numScores; ++i) {
            if (rawOutput[i] > maxProb) {
                maxProb = rawOutput[i];
                bestClassId = i;
            }
        }

        std::string className = (bestClassId >= 0 && static_cast<size_t>(bestClassId) < classNames_.size())
                               ? classNames_[bestClassId]
                               : ("Class_" + std::to_string(bestClassId));

        return ClassificationResult(bestClassId, maxProb, className);
    }
};

// ============================================================================
// Version-Specific Classifier Subclasses
// ============================================================================

/// @brief YOLOv11 classifier
class YOLO11Classifier : public YOLOClassifier {
public:
    YOLO11Classifier(const std::string& modelPath, const std::string& labelsPath, bool useGPU = false)
        : YOLOClassifier(modelPath, labelsPath, useGPU) {}
};

/// @brief YOLOv12 classifier
class YOLO12Classifier : public YOLOClassifier {
public:
    YOLO12Classifier(const std::string& modelPath, const std::string& labelsPath, bool useGPU = false)
        : YOLOClassifier(modelPath, labelsPath, useGPU) {}
};

/// @brief YOLO26 classifier
class YOLO26Classifier : public YOLOClassifier {
public:
    YOLO26Classifier(const std::string& modelPath, const std::string& labelsPath, bool useGPU = false)
        : YOLOClassifier(modelPath, labelsPath, useGPU) {}
};

// ============================================================================
// Factory Function
// ============================================================================

/// @brief Create a classifier with explicit version selection
/// @param modelPath Path to the ONNX model
/// @param labelsPath Path to the class names file
/// @param version YOLO version
/// @param useGPU Whether to use GPU
/// @return Unique pointer to classifier
inline std::unique_ptr<YOLOClassifier> createClassifier(const std::string& modelPath,
                                                        const std::string& labelsPath,
                                                        YOLOVersion version = YOLOVersion::V11,
                                                        bool useGPU = false) {
    switch (version) {
        case YOLOVersion::V26:
            return std::make_unique<YOLO26Classifier>(modelPath, labelsPath, useGPU);
        case YOLOVersion::V12:
            return std::make_unique<YOLO12Classifier>(modelPath, labelsPath, useGPU);
        default:
            return std::make_unique<YOLO11Classifier>(modelPath, labelsPath, useGPU);
    }
}

} // namespace cls
} // namespace yolos
