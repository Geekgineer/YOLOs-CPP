#pragma once

// ============================================================================
// YOLO ONNX Session Base
// ============================================================================
// Common ONNX Runtime session setup and management for all YOLO detectors.
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "yolos/core/version.hpp"
#include "yolos/core/utils.hpp"

namespace yolos {

// ============================================================================
// OrtSessionBase - Common ONNX Runtime session management
// ============================================================================

/// @brief Base class for ONNX Runtime session management
/// Handles model loading, session configuration, and common inference setup
class OrtSessionBase {
public:
    /// @brief Constructor - loads and initializes the ONNX model
    /// @param modelPath Path to the ONNX model file
    /// @param useGPU Whether to use GPU (CUDA) for inference
    /// @param numThreads Number of intra-op threads (0 = auto)
    OrtSessionBase(const std::string& modelPath, bool useGPU = false, int numThreads = 0)
        : env_(ORT_LOGGING_LEVEL_WARNING, "YOLOS") {
        
        initSession(modelPath, useGPU, numThreads);
    }

    virtual ~OrtSessionBase() = default;

    // Prevent copying
    OrtSessionBase(const OrtSessionBase&) = delete;
    OrtSessionBase& operator=(const OrtSessionBase&) = delete;

    // Allow moving
    OrtSessionBase(OrtSessionBase&&) = default;
    OrtSessionBase& operator=(OrtSessionBase&&) = default;

    /// @brief Get the input image shape expected by the model
    [[nodiscard]] cv::Size getInputShape() const noexcept { return inputShape_; }

    /// @brief Check if input shape is dynamic
    [[nodiscard]] bool isDynamicInputShape() const noexcept { return isDynamicInputShape_; }

    /// @brief Check if batch size is dynamic
    [[nodiscard]] bool isDynamicBatchSize() const noexcept { return isDynamicBatchSize_; }

    /// @brief Get the device being used for inference
    [[nodiscard]] const std::string& getDevice() const noexcept { return device_; }

    /// @brief Get the number of input nodes
    [[nodiscard]] size_t getNumInputNodes() const noexcept { return numInputNodes_; }

    /// @brief Get the number of output nodes
    [[nodiscard]] size_t getNumOutputNodes() const noexcept { return numOutputNodes_; }

protected:
    Ort::Env env_{nullptr};
    Ort::SessionOptions sessionOptions_{nullptr};
    Ort::Session session_{nullptr};

    // Input/output node names
    std::vector<Ort::AllocatedStringPtr> inputNameAllocs_;
    std::vector<const char*> inputNames_;
    std::vector<Ort::AllocatedStringPtr> outputNameAllocs_;
    std::vector<const char*> outputNames_;

    size_t numInputNodes_{0};
    size_t numOutputNodes_{0};

    cv::Size inputShape_;
    bool isDynamicInputShape_{false};
    bool isDynamicBatchSize_{false};
    std::string device_{"cpu"};

    /// @brief Run inference with the given input tensor
    /// @param inputTensor Input tensor
    /// @return Vector of output tensors
    std::vector<Ort::Value> runInference(Ort::Value& inputTensor) {
        return session_.Run(
            Ort::RunOptions{nullptr},
            inputNames_.data(),
            &inputTensor,
            numInputNodes_,
            outputNames_.data(),
            numOutputNodes_
        );
    }

    /// @brief Create an input tensor from a blob
    /// @param blob Pointer to the input data
    /// @param inputTensorShape Shape of the input tensor
    /// @return ONNX Runtime input tensor
    Ort::Value createInputTensor(float* blob, const std::vector<int64_t>& inputTensorShape) {
        static Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        size_t inputTensorSize = utils::vectorProduct(inputTensorShape);
        
        return Ort::Value::CreateTensor<float>(
            memoryInfo,
            blob,
            inputTensorSize,
            inputTensorShape.data(),
            inputTensorShape.size()
        );
    }

private:
    void initSession(const std::string& modelPath, bool useGPU, int numThreads) {
        sessionOptions_ = Ort::SessionOptions();

        // Set thread count
        int threads = (numThreads > 0) ? numThreads : std::min(6, static_cast<int>(std::thread::hardware_concurrency()));
        sessionOptions_.SetIntraOpNumThreads(threads);
        sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Configure execution provider
        std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
        auto cudaIt = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");

        if (useGPU && cudaIt != availableProviders.end()) {
            OrtCUDAProviderOptions cudaOptions{};
            sessionOptions_.AppendExecutionProvider_CUDA(cudaOptions);
            device_ = "gpu";
            std::cout << "[INFO] Inference device: GPU (CUDA)" << std::endl;
        } else {
            if (useGPU) {
                std::cout << "[WARNING] GPU requested but CUDA not available. Falling back to CPU." << std::endl;
            }
            device_ = "cpu";
            std::cout << "[INFO] Inference device: CPU" << std::endl;
        }

        // Load model
#ifdef _WIN32
        std::wstring wModelPath(modelPath.begin(), modelPath.end());
        session_ = Ort::Session(env_, wModelPath.c_str(), sessionOptions_);
#else
        session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);
#endif

        // Get node counts
        numInputNodes_ = session_.GetInputCount();
        numOutputNodes_ = session_.GetOutputCount();

        Ort::AllocatorWithDefaultOptions allocator;

        // Get input node names
        for (size_t i = 0; i < numInputNodes_; ++i) {
            auto inputName = session_.GetInputNameAllocated(i, allocator);
            inputNameAllocs_.push_back(std::move(inputName));
            inputNames_.push_back(inputNameAllocs_.back().get());
        }

        // Get output node names
        for (size_t i = 0; i < numOutputNodes_; ++i) {
            auto outputName = session_.GetOutputNameAllocated(i, allocator);
            outputNameAllocs_.push_back(std::move(outputName));
            outputNames_.push_back(outputNameAllocs_.back().get());
        }

        // Get input shape
        Ort::TypeInfo inputTypeInfo = session_.GetInputTypeInfo(0);
        std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();

        if (inputTensorShape.size() >= 4) {
            isDynamicBatchSize_ = (inputTensorShape[0] == -1);
            isDynamicInputShape_ = (inputTensorShape[2] == -1 || inputTensorShape[3] == -1);

            int height = (inputTensorShape[2] == -1) ? 640 : static_cast<int>(inputTensorShape[2]);
            int width = (inputTensorShape[3] == -1) ? 640 : static_cast<int>(inputTensorShape[3]);
            inputShape_ = cv::Size(width, height);
        } else {
            throw std::runtime_error("Invalid input tensor shape. Expected 4D tensor [N, C, H, W].");
        }

        std::cout << "[INFO] Model loaded: " << modelPath << std::endl;
        std::cout << "[INFO] Input shape: " << inputShape_.width << "x" << inputShape_.height
                  << (isDynamicInputShape_ ? " (dynamic)" : "") << std::endl;
        std::cout << "[INFO] Inputs: " << numInputNodes_ << ", Outputs: " << numOutputNodes_ << std::endl;
    }
};

} // namespace yolos
