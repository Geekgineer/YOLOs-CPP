#pragma once

// One-file unified YOLO classifier (merges YOLO11 and YOLO12 implementations)

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
#include <iomanip>
#include <sstream>
#include <variant>

#include "tools/Debug.hpp"
#include "tools/ScopedTimer.hpp"

struct ClassificationResult {
    int classId{-1};
    float confidence{0.0f};
    std::string className{};

    ClassificationResult() = default;
    ClassificationResult(int id, float conf, std::string name)
        : classId(id), confidence(conf), className(std::move(name)) {}
};

namespace utils {
    template <typename T>
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type
    inline clamp(const T &value, const T &low, const T &high) {
        T validLow = low < high ? low : high;
        T validHigh = low < high ? high : low;
        if (value < validLow) return validLow;
        if (value > validHigh) return validHigh;
        return value;
    }

    inline std::vector<std::string> getClassNames(const std::string &path) {
        std::vector<std::string> classNames;
        std::ifstream infile(path);
        if (infile) {
            std::string line;
            while (getline(infile, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                classNames.emplace_back(line);
            }
        } else {
            std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
        }
        return classNames;
    }

    inline size_t vectorProduct(const std::vector<int64_t> &vector) {
        if (vector.empty()) return 0;
        return std::accumulate(vector.begin(), vector.end(), 1LL, std::multiplies<int64_t>());
    }

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
            if (!scaleUp) r = std::min(r, 1.0f);
            int newUnpadW = static_cast<int>(std::round(image.cols * r));
            int newUnpadH = static_cast<int>(std::round(image.rows * r));
            cv::Mat resizedTemp;
            cv::resize(image, resizedTemp, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
            int dw = targetShape.width - newUnpadW;
            int dh = targetShape.height - newUnpadH;
            int top = dh / 2, bottom = dh - top, left = dw / 2, right = dw - left;
            cv::copyMakeBorder(resizedTemp, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
        } else {
            if (image.size() == targetShape) outImage = image.clone();
            else cv::resize(image, outImage, targetShape, 0, 0, cv::INTER_LINEAR);
        }
    }

    inline void drawClassificationResult(cv::Mat &image, const ClassificationResult &result,
                                         const cv::Point& position = cv::Point(10, 10),
                                         const cv::Scalar& textColor = cv::Scalar(0, 255, 0),
                                         double fontScaleMultiplier = 0.0008,
                                         const cv::Scalar& bgColor = cv::Scalar(0,0,0) ) {
        if (image.empty() || result.classId == -1) return;
        std::ostringstream ss; ss << result.className << ": " << std::fixed << std::setprecision(2) << result.confidence * 100 << "%";
        std::string text = ss.str();
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = std::min(image.rows, image.cols) * fontScaleMultiplier; if (fontScale < 0.4) fontScale = 0.4;
        const int thickness = std::max(1, static_cast<int>(fontScale * 1.8)); int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline); baseline += thickness;
        cv::Point textPosition = position; if (textPosition.x < 0) textPosition.x = 0; if (textPosition.y < textSize.height) textPosition.y = textSize.height + 2;
        cv::Point backgroundTopLeft(textPosition.x, textPosition.y - textSize.height - baseline / 3);
        cv::Point backgroundBottomRight(textPosition.x + textSize.width, textPosition.y + baseline / 2);
        backgroundTopLeft.x = utils::clamp(backgroundTopLeft.x, 0, image.cols -1); backgroundTopLeft.y = utils::clamp(backgroundTopLeft.y, 0, image.rows -1);
        backgroundBottomRight.x = utils::clamp(backgroundBottomRight.x, 0, image.cols -1); backgroundBottomRight.y = utils::clamp(backgroundBottomRight.y, 0, image.rows -1);
        cv::rectangle(image, backgroundTopLeft, backgroundBottomRight, bgColor, cv::FILLED);
        cv::putText(image, text, cv::Point(textPosition.x, textPosition.y), fontFace, fontScale, textColor, thickness, cv::LINE_AA);
    }
} // namespace utils

class BaseYOLOClassifier {
public:
    BaseYOLOClassifier(const std::string &modelPath, const std::string &labelsPath,
                    bool useGPU = false, const cv::Size& targetInputShape = cv::Size(224, 224));
    ClassificationResult classify(const cv::Mat &image);
    void drawResult(cv::Mat &image, const ClassificationResult &result,
                    const cv::Point& position = cv::Point(10, 10)) const {
        utils::drawClassificationResult(image, result, position);
    }
    cv::Size getInputShape() const { return inputImageShape_; }
    bool isModelInputShapeDynamic() const { return isDynamicInputShape_; }
private:
    Ort::Env env_{nullptr};
    Ort::SessionOptions sessionOptions_{nullptr};
    Ort::Session session_{nullptr};
    bool isDynamicInputShape_{};
    cv::Size inputImageShape_{};
    std::vector<float> inputBuffer_{}; // persistent input buffer to avoid reallocations
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

inline BaseYOLOClassifier::BaseYOLOClassifier(const std::string &modelPath, const std::string &labelsPath,
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
        sessionOptions_.AppendExecutionProvider_CUDA(cudaOption);
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
    auto input_node_name = session_.GetInputNameAllocated(0, allocator);
    inputNodeNameAllocatedStrings_.push_back(std::move(input_node_name));
    inputNames_.push_back(inputNodeNameAllocatedStrings_.back().get());
    Ort::TypeInfo inputTypeInfo = session_.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> modelInputTensorShapeVec = inputTensorInfo.GetShape();
    if (modelInputTensorShapeVec.size() == 4) {
        isDynamicInputShape_ = (modelInputTensorShapeVec[2] == -1 || modelInputTensorShapeVec[3] == -1);
        if (!isDynamicInputShape_) {
            int modelH = static_cast<int>(modelInputTensorShapeVec[2]);
            int modelW = static_cast<int>(modelInputTensorShapeVec[3]);
            if (modelH != inputImageShape_.height || modelW != inputImageShape_.width) {
                std::cout << "Warning: Target preprocessing shape (" << inputImageShape_.height << "x" << inputImageShape_.width
                          << ") differs from model's fixed input shape (" << modelH << "x" << modelW << "). "
                          << "Image will be preprocessed to " << inputImageShape_.height << "x" << inputImageShape_.width << "." << std::endl;
            }
        }
    } else {
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
            for (long long dim : outputTensorShapeVec) if (dim > 1 && numClasses_ == 0) numClasses_ = static_cast<int>(dim);
            if (numClasses_ == 0 && !outputTensorShapeVec.empty()) numClasses_ = static_cast<int>(outputTensorShapeVec.back());
        }
    }
    classNames_ = utils::getClassNames(labelsPath);
}

inline void BaseYOLOClassifier::preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape) {
    ScopedTimer timer("Preprocessing (Ultralytics-style)");
    if (image.empty()) throw std::runtime_error("Input image to preprocess is empty.");
    cv::Mat processedImage; utils::preprocessImageToTensor(image, processedImage, inputImageShape_, cv::Scalar(0, 0, 0), true, "resize");
    cv::Mat rgbImageMat; cv::cvtColor(processedImage, rgbImageMat, cv::COLOR_BGR2RGB);
    cv::Mat floatRgbImage; rgbImageMat.convertTo(floatRgbImage, CV_32F, 1.0/255.0);
    inputTensorShape = {1, 3, static_cast<int64_t>(floatRgbImage.rows), static_cast<int64_t>(floatRgbImage.cols)};
    const int h = static_cast<int>(inputTensorShape[2]);
    const int w = static_cast<int>(inputTensorShape[3]);
    const size_t tensorSize = static_cast<size_t>(1) * 3 * h * w;
    inputBuffer_.resize(tensorSize);
    std::vector<cv::Mat> channels(3);
    cv::split(floatRgbImage, channels);
    for (int c = 0; c < 3; ++c) {
        const cv::Mat &plane = channels[c];
        std::memcpy(inputBuffer_.data() + c * (h * w), plane.ptr<float>(), static_cast<size_t>(h * w) * sizeof(float));
    }
    blob = inputBuffer_.data();
}

inline ClassificationResult BaseYOLOClassifier::postprocess(const std::vector<Ort::Value> &outputTensors) {
    ScopedTimer timer("Postprocessing");
    if (outputTensors.empty()) return {};
    const float* rawOutput = outputTensors[0].GetTensorData<float>(); if (!rawOutput) return {};
    const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t numScores = utils::vectorProduct(outputShape);
    int currentNumClasses = numClasses_ > 0 ? numClasses_ : static_cast<int>(classNames_.size()); if (currentNumClasses <= 0) return {};
    int bestClassId = -1; float maxScore = -std::numeric_limits<float>::infinity(); std::vector<float> scores(currentNumClasses);
    if (outputShape.size() == 2 && outputShape[0] == 1) {
        for (int i = 0; i < currentNumClasses && i < static_cast<int>(outputShape[1]); ++i) { scores[i] = rawOutput[i]; if (scores[i] > maxScore) { maxScore = scores[i]; bestClassId = i; } }
    } else {
        for (int i = 0; i < currentNumClasses && i < static_cast<int>(numScores); ++i) { scores[i] = rawOutput[i]; if (scores[i] > maxScore) { maxScore = scores[i]; bestClassId = i; } }
    }
    if (bestClassId == -1) return {};
    float sumExp = 0.0f; std::vector<float> probabilities(currentNumClasses);
    for (int i = 0; i < currentNumClasses; ++i) { probabilities[i] = std::exp(scores[i] - maxScore); sumExp += probabilities[i]; }
    float confidence = sumExp > 0 ? probabilities[bestClassId] / sumExp : 0.0f;
    std::string className = (bestClassId >= 0 && static_cast<size_t>(bestClassId) < classNames_.size()) ? classNames_[bestClassId] : ("ClassID_" + std::to_string(bestClassId));
    return ClassificationResult(bestClassId, confidence, className);
}

inline ClassificationResult BaseYOLOClassifier::classify(const cv::Mat& image) {
    ScopedTimer timer("Overall classification task"); if (image.empty()) return {};
    float* blobPtr = nullptr; std::vector<int64_t> currentInputTensorShape; preprocess(image, blobPtr, currentInputTensorShape);
    size_t inputTensorSize = utils::vectorProduct(currentInputTensorShape);
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, blobPtr, inputTensorSize, currentInputTensorShape.data(), currentInputTensorShape.size());
    std::vector<Ort::Value> outputTensors = session_.Run(Ort::RunOptions{nullptr}, inputNames_.data(), &inputTensor, numInputNodes_, outputNames_.data(), numOutputNodes_);
    if (outputTensors.empty()) return {};
    return postprocess(outputTensors);
}

// Thin wrappers for versioned classifiers
class YOLO11Classifier : public BaseYOLOClassifier {
public:
    using BaseYOLOClassifier::BaseYOLOClassifier;
};

class YOLO12Classifier : public BaseYOLOClassifier {
public:
    using BaseYOLOClassifier::BaseYOLOClassifier;
};

enum class YOLOClassVersion { V11, V12 };

class YOLOClassifier {
public:
    YOLOClassifier(const std::string &modelPath,
                   const std::string &labelsPath,
                   bool useGPU = false,
                   YOLOClassVersion version = YOLOClassVersion::V11)
    {
        if (version == YOLOClassVersion::V11) {
            impl_.template emplace<YOLO11Classifier>(modelPath, labelsPath, useGPU);
        } else {
            impl_.template emplace<YOLO12Classifier>(modelPath, labelsPath, useGPU);
        }
    }
    ClassificationResult classify(const cv::Mat &image) {
        if (auto *p = std::get_if<YOLO11Classifier>(&impl_)) return p->classify(image);
        if (auto *q = std::get_if<YOLO12Classifier>(&impl_)) return q->classify(image);
        return {};
    }
    void drawResult(cv::Mat &image, const ClassificationResult &result,
                    const cv::Point &position = cv::Point(10, 10)) const {
        if (auto *p = std::get_if<YOLO11Classifier>(&impl_)) { p->drawResult(image, result, position); return; }
        if (auto *q = std::get_if<YOLO12Classifier>(&impl_)) { q->drawResult(image, result, position); return; }
    }
    cv::Size getInputShape() const {
        if (auto *p = std::get_if<YOLO11Classifier>(&impl_)) return p->getInputShape();
        if (auto *q = std::get_if<YOLO12Classifier>(&impl_)) return q->getInputShape();
        return cv::Size();
    }
    bool isModelInputShapeDynamic() const {
        if (auto *p = std::get_if<YOLO11Classifier>(&impl_)) return p->isModelInputShapeDynamic();
        if (auto *q = std::get_if<YOLO12Classifier>(&impl_)) return q->isModelInputShapeDynamic();
        return true;
    }
private:
    std::variant<std::monostate, YOLO11Classifier, YOLO12Classifier> impl_;
};
