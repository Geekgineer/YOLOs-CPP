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
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "yolos/core/version.hpp"
#include "yolos/core/utils.hpp"
#include "yolos/core/preprocessing.hpp"

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

        configureSessionOptions(useGPU);

#ifdef _WIN32
        std::wstring wModelPath(modelPath.begin(), modelPath.end());
        session_ = Ort::Session(env_, wModelPath.c_str(), sessionOptions_);
#else
        session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);
#endif

        introspectSession(modelPath);
        classNames_ = utils::getClassNames(labelsPath);
    }

    /// @brief Constructor loading the model from memory instead of a file
    /// @param modelData Pointer to the serialized ONNX model bytes
    /// @param modelSize Size of the buffer in bytes
    /// @param classNames Class names in class-id order
    /// @param useGPU Whether to use GPU for inference
    /// @param targetInputShape Target input shape for preprocessing
    /// @note ONNX Runtime copies the buffer during session creation, so
    ///       @p modelData may be freed once the constructor returns.
    YOLOClassifier(const void* modelData,
                   size_t modelSize,
                   const std::vector<std::string>& classNames,
                   bool useGPU = false,
                   const cv::Size& targetInputShape = cv::Size(224, 224))
        : inputImageShape_(targetInputShape),
          env_(ORT_LOGGING_LEVEL_WARNING, "YOLOClassifier") {

        if (modelData == nullptr || modelSize == 0) {
            throw std::invalid_argument("Model buffer is empty (modelData == nullptr or modelSize == 0).");
        }

        configureSessionOptions(useGPU);

        session_ = Ort::Session(env_, modelData, modelSize, sessionOptions_);

        introspectSession("<memory buffer, " + std::to_string(modelSize) + " bytes>");
        classNames_ = classNames;
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

    /// @brief Run classification on several images with a single ONNX Runtime call
    /// @param images Input images (BGR format)
    /// @return One result per input image, in input order
    /// @note Packs the batch into one N*3*H*W tensor when the model accepts the
    ///       batch size (dynamic batch dimension, or a fixed one that matches
    ///       images.size()); otherwise falls back to classify() per image.
    ///       Empty inputs yield a default-constructed result at their position.
    std::vector<ClassificationResult> batchClassify(const std::vector<cv::Mat>& images) {
        std::vector<ClassificationResult> results;
        if (images.empty()) return results;

        const bool anyEmpty = std::any_of(images.begin(), images.end(),
                                          [](const cv::Mat& m) { return m.empty(); });
        const bool canBatch = !anyEmpty &&
                              (isDynamicBatchSize_ || modelBatchSize_ == static_cast<int>(images.size()));

        if (canBatch) {
            try {
                return batchClassifyPacked(images);
            } catch (const Ort::Exception& e) {
                // Some exports declare a dynamic batch dimension the graph cannot
                // actually run (hard-coded Reshape targets, for instance).
                std::cerr << "[WARNING] Batched inference failed (" << e.what()
                          << "). Falling back to per-image inference." << std::endl;
            }
        }

        results.reserve(images.size());
        for (const auto& image : images) {
            results.push_back(classify(image));
        }
        return results;
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

    /// @brief Check if the model's batch dimension is dynamic
    [[nodiscard]] bool isDynamicBatchSize() const { return isDynamicBatchSize_; }

    /// @brief Fixed batch size baked into the model, or -1 when the batch dim is dynamic
    [[nodiscard]] int getModelBatchSize() const { return modelBatchSize_; }

    /// @brief Get class names
    [[nodiscard]] const std::vector<std::string>& getClassNames() const { return classNames_; }

protected:
    cv::Size inputImageShape_;
    Ort::Env env_{nullptr};
    Ort::SessionOptions sessionOptions_{nullptr};
    Ort::Session session_{nullptr};
    bool isDynamicInputShape_{false};
    bool isDynamicBatchSize_{false};
    int modelBatchSize_{1};
    std::vector<float> inputBuffer_;

    std::vector<Ort::AllocatedStringPtr> inputNameAllocs_;
    std::vector<const char*> inputNames_;
    std::vector<Ort::AllocatedStringPtr> outputNameAllocs_;
    std::vector<const char*> outputNames_;

    size_t numInputNodes_{0};
    size_t numOutputNodes_{0};
    int numClasses_{0};
    std::vector<std::string> classNames_;

    /// @brief Single batched ONNX call plus per-image postprocessing
    /// Throws Ort::Exception if the model cannot run the requested batch size.
    std::vector<ClassificationResult> batchClassifyPacked(const std::vector<cv::Mat>& images) {
        // Every image is center-cropped to the same square, so one shared shape works
        const int targetSize = inputImageShape_.width;
        const size_t sliceSize = static_cast<size_t>(3) * targetSize * targetSize;
        inputBuffer_.resize(sliceSize * images.size());

        for (size_t i = 0; i < images.size(); ++i) {
            preprocessInto(images[i], inputBuffer_.data() + i * sliceSize);
        }

        std::vector<int64_t> inputTensorShape = {
            static_cast<int64_t>(images.size()), 3, targetSize, targetSize
        };

        static Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputBuffer_.data(), inputBuffer_.size(),
            inputTensorShape.data(), inputTensorShape.size());

        std::vector<Ort::Value> outputTensors = session_.Run(
            Ort::RunOptions{nullptr},
            inputNames_.data(), &inputTensor, numInputNodes_,
            outputNames_.data(), numOutputNodes_);

        if (outputTensors.empty()) {
            return std::vector<ClassificationResult>(images.size());
        }

        const float* rawOutput = outputTensors[0].GetTensorData<float>();
        const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        const size_t scoresPerImage = outputShape.size() >= 2
            ? static_cast<size_t>(outputShape.back())
            : static_cast<size_t>(numClasses_);

        std::vector<ClassificationResult> results;
        results.reserve(images.size());
        for (size_t i = 0; i < images.size(); ++i) {
            results.push_back(postprocessScores(rawOutput + i * scoresPerImage));
        }

        return results;
    }

    void configureSessionOptions(bool useGPU) {
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
    }

    /// @brief Read node names, shapes and class count from an already-created session
    /// @param modelLabel Human-readable model identifier used only for logging
    void introspectSession(const std::string& modelLabel) {
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
            isDynamicBatchSize_ = (inputShape[0] <= 0);
            modelBatchSize_ = isDynamicBatchSize_ ? -1 : static_cast<int>(inputShape[0]);
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

        std::cout << "[INFO] Classification model loaded: " << modelLabel << std::endl;
        std::cout << "[INFO] Input shape: " << inputImageShape_.width << "x" << inputImageShape_.height << std::endl;
        std::cout << "[INFO] Batch size: "
                  << (isDynamicBatchSize_ ? std::string("dynamic") : std::to_string(modelBatchSize_))
                  << std::endl;
        std::cout << "[INFO] Number of classes: " << numClasses_ << std::endl;
    }

    /// @brief Preprocess image for classification into the shared input buffer
    ///
    /// Mirrors ultralytics.data.augment.classify_transforms(), which is
    ///     T.Resize(size, BILINEAR) -> T.CenterCrop(size) -> T.ToTensor()
    ///       -> T.Normalize(mean=0, std=1)   [a no-op]
    /// applied to a PIL image. The work itself lives in preprocessInto().
    void preprocess(const cv::Mat& image, std::vector<int64_t>& inputTensorShape) {
        const int targetSize = inputImageShape_.width;
        inputTensorShape = {1, 3, targetSize, targetSize};
        inputBuffer_.resize(static_cast<size_t>(3) * targetSize * targetSize);
        preprocessInto(image, inputBuffer_.data());
    }

    /// @brief Preprocess image for classification (Ultralytics-style) into a CHW slice
    /// @param image Input BGR image
    /// @param dst Destination slice; must hold 3 * targetSize * targetSize floats
    /// @note Safe to point at one image of a batched N*3*H*W buffer.
    void preprocessInto(const cv::Mat& image, float* dst) {
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

        // Ultralytics resizes through PIL, whose BILINEAR filter antialiases.
        // cv::resize(INTER_LINEAR) does not, which inflated confidences and could
        // flip the top-1 class relative to Ultralytics (issue #137).
        cv::Mat resized;
        preprocessing::resizeAntialiasBilinear(rgbImage, resized, cv::Size(newW, newH));

        // Center crop to target_size x target_size.
        // torchvision uses int(round(diff / 2.0)); Python's round() breaks ties to
        // even, which is what std::nearbyint does under the default rounding mode.
        int yStart = std::max(0, static_cast<int>(std::nearbyint((newH - targetSize) / 2.0)));
        int xStart = std::max(0, static_cast<int>(std::nearbyint((newW - targetSize) / 2.0)));
        cv::Mat cropped = resized(cv::Rect(xStart, yStart, targetSize, targetSize));

        // Normalize to [0, 1]
        cv::Mat floatImage;
        cropped.convertTo(floatImage, CV_32F, 1.0 / 255.0);

        const int finalH = floatImage.rows;
        const int finalW = floatImage.cols;

        // Convert HWC to CHW format
        std::vector<cv::Mat> channels(3);
        cv::split(floatImage, channels);
        for (int c = 0; c < 3; ++c) {
            std::memcpy(dst + static_cast<size_t>(c) * finalH * finalW,
                       channels[c].ptr<float>(), static_cast<size_t>(finalH) * finalW * sizeof(float));
        }
    }

    /// @brief Postprocess classification output
    ClassificationResult postprocess(const std::vector<Ort::Value>& outputTensors) {
        return postprocessScores(outputTensors[0].GetTensorData<float>());
    }

    /// @brief Turn one image's raw score row into a result
    /// @param rawOutput Pointer to this image's scores (numClasses_ floats)
    ClassificationResult postprocessScores(const float* rawOutput) {
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

    YOLO11Classifier(const void* modelData, size_t modelSize,
                     const std::vector<std::string>& classNames, bool useGPU = false)
        : YOLOClassifier(modelData, modelSize, classNames, useGPU) {}
};

/// @brief YOLOv12 classifier
class YOLO12Classifier : public YOLOClassifier {
public:
    YOLO12Classifier(const std::string& modelPath, const std::string& labelsPath, bool useGPU = false)
        : YOLOClassifier(modelPath, labelsPath, useGPU) {}

    YOLO12Classifier(const void* modelData, size_t modelSize,
                     const std::vector<std::string>& classNames, bool useGPU = false)
        : YOLOClassifier(modelData, modelSize, classNames, useGPU) {}
};

/// @brief YOLO26 classifier
class YOLO26Classifier : public YOLOClassifier {
public:
    YOLO26Classifier(const std::string& modelPath, const std::string& labelsPath, bool useGPU = false)
        : YOLOClassifier(modelPath, labelsPath, useGPU) {}

    YOLO26Classifier(const void* modelData, size_t modelSize,
                     const std::vector<std::string>& classNames, bool useGPU = false)
        : YOLOClassifier(modelData, modelSize, classNames, useGPU) {}
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

/// @brief Create a classifier from a model held in memory
/// @param modelData Pointer to the serialized ONNX model bytes
/// @param modelSize Size of the buffer in bytes
/// @param classNames Class names in class-id order
/// @param version YOLO version
/// @param useGPU Whether to use GPU
/// @return Unique pointer to classifier
inline std::unique_ptr<YOLOClassifier> createClassifierFromMemory(const void* modelData,
                                                                  size_t modelSize,
                                                                  const std::vector<std::string>& classNames,
                                                                  YOLOVersion version = YOLOVersion::V11,
                                                                  bool useGPU = false) {
    switch (version) {
        case YOLOVersion::V26:
            return std::make_unique<YOLO26Classifier>(modelData, modelSize, classNames, useGPU);
        case YOLOVersion::V12:
            return std::make_unique<YOLO12Classifier>(modelData, modelSize, classNames, useGPU);
        default:
            return std::make_unique<YOLO11Classifier>(modelData, modelSize, classNames, useGPU);
    }
}

} // namespace cls
} // namespace yolos
