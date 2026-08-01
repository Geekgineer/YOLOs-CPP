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
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "yolos/core/onnx_metadata.hpp"
#include "yolos/core/preprocessing.hpp"
#include "yolos/core/utils.hpp"
#include "yolos/core/version.hpp"

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

        configureSessionOptions(useGPU, numThreads);

#ifdef _WIN32
        std::wstring wModelPath(modelPath.begin(), modelPath.end());
        session_ = Ort::Session(env_, wModelPath.c_str(), sessionOptions_);
#else
        session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);
#endif

        introspectSession(modelPath);
    }

    /// @brief Constructor - initializes the ONNX model from an in-memory buffer
    /// @param modelData Pointer to the serialized ONNX model bytes
    /// @param modelSize Size of the buffer in bytes
    /// @param useGPU Whether to use GPU (CUDA) for inference
    /// @param numThreads Number of intra-op threads (0 = auto)
    /// @note ONNX Runtime copies the buffer while creating the session, so the
    ///       caller may free @p modelData as soon as the constructor returns.
    ///       Useful for encrypted stores, network streams and embedded resources.
    OrtSessionBase(const void* modelData, size_t modelSize, bool useGPU = false, int numThreads = 0)
        : env_(ORT_LOGGING_LEVEL_WARNING, "YOLOS") {

        if (modelData == nullptr || modelSize == 0) {
            throw std::invalid_argument("Model buffer is empty (modelData == nullptr or modelSize == 0).");
        }

        configureSessionOptions(useGPU, numThreads);

        session_ = Ort::Session(env_, modelData, modelSize, sessionOptions_);

        introspectSession("<memory buffer, " + std::to_string(modelSize) + " bytes>");
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

    /// @brief Fixed batch size baked into the model, or -1 when the batch dim is dynamic
    [[nodiscard]] int getModelBatchSize() const noexcept { return modelBatchSize_; }

    /// @brief Whether a single ONNX call can process exactly @p count images
    /// Dynamic-batch models accept any count; fixed-batch models only their own.
    [[nodiscard]] bool supportsBatchSize(size_t count) const noexcept {
        if (count == 0) return false;
        if (isDynamicBatchSize_) return true;
        return modelBatchSize_ == static_cast<int>(count);
    }

    /// @brief Get the device being used for inference
    [[nodiscard]] const std::string& getDevice() const noexcept { return device_; }

    /// @brief Get the number of input nodes
    [[nodiscard]] size_t getNumInputNodes() const noexcept { return numInputNodes_; }

    /// @brief Get the number of output nodes
    [[nodiscard]] size_t getNumOutputNodes() const noexcept { return numOutputNodes_; }

    /// @brief Class names from ONNX custom metadata `names` (Ultralytics), if present.
    [[nodiscard]] const std::vector<std::string>& getExportedClassNamesFromMetadata() const noexcept {
        return exportedClassNamesFromMetadata_;
    }

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

    int inputChannels_{3};
    cv::Size inputShape_;
    bool isDynamicInputShape_{false};
    bool isDynamicBatchSize_{false};
    int modelBatchSize_{1};
    std::string device_{"cpu"};

    /// Ultralytics-exported `names` dict parsed from ONNX metadata (empty if missing).
    std::vector<std::string> exportedClassNamesFromMetadata_;

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

    /// @brief Letterbox a batch of images and run a single batched inference
    /// @param images Input BGR images (all letterboxed to the same target size)
    /// @param blob Scratch buffer reused across calls for the N*C*H*W input
    /// @param[out] letterboxSize Shared letterbox size used for every image
    /// @return Output tensors with a leading batch dimension of images.size()
    /// @note Batching forces one common letterbox target (the model input shape),
    ///       so dynamic-input-shape models do not get per-image stride alignment
    ///       here the way the single-image paths do.
    std::vector<Ort::Value> runBatchInference(const std::vector<cv::Mat>& images,
                                              std::vector<float>& blob,
                                              cv::Size& letterboxSize) {
        letterboxSize = inputShape_;
        preprocessing::letterBoxToBatchBlob(images, blob, inputChannels_, letterboxSize);

        const std::vector<int64_t> inputTensorShape = {
            static_cast<int64_t>(images.size()),
            static_cast<int64_t>(inputChannels_),
            letterboxSize.height,
            letterboxSize.width
        };

        Ort::Value inputTensor = createInputTensor(blob.data(), inputTensorShape);
        return runInference(inputTensor);
    }

    /// @brief Check that every model output is a float tensor
    /// The batch-slicing helper below only understands float tensors.
    [[nodiscard]] bool allOutputsAreFloat() const {
        for (size_t i = 0; i < numOutputNodes_; ++i) {
            Ort::TypeInfo info = session_.GetOutputTypeInfo(i);
            if (info.GetONNXType() != ONNX_TYPE_TENSOR ||
                info.GetTensorTypeAndShapeInfo().GetElementType() !=
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                return false;
            }
        }
        return true;
    }

    /// @brief Build zero-copy single-image views into batched output tensors
    /// @param batchOutputs Output tensors produced by a batched run
    /// @param batchIndex Index of the image to view
    /// @param batchSize Batch size the run was made with
    /// @return Tensors shaped [1, ...] aliasing @p batchOutputs (no data copied)
    /// @note The returned views borrow @p batchOutputs' memory, so they must not
    ///       outlive it. Slicing lets the existing single-image postprocessing
    ///       (including subclass overrides) run unchanged on each batch element.
    static std::vector<Ort::Value> sliceOutputBatch(std::vector<Ort::Value>& batchOutputs,
                                                    int64_t batchIndex,
                                                    int64_t batchSize) {
        static Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<Ort::Value> views;
        views.reserve(batchOutputs.size());

        for (auto& tensor : batchOutputs) {
            std::vector<int64_t> shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
            float* data = tensor.GetTensorMutableData<float>();

            // Outputs whose leading dimension is not the batch cannot be sliced;
            // share them whole so postprocessing still sees the full tensor.
            if (shape.empty() || shape[0] != batchSize) {
                views.push_back(Ort::Value::CreateTensor<float>(
                    memoryInfo, data, utils::vectorProduct(shape), shape.data(), shape.size()));
                continue;
            }

            std::vector<int64_t> sliceShape = shape;
            sliceShape[0] = 1;
            const size_t sliceElems = utils::vectorProduct(sliceShape);

            views.push_back(Ort::Value::CreateTensor<float>(
                memoryInfo,
                data + static_cast<size_t>(batchIndex) * sliceElems,
                sliceElems,
                sliceShape.data(),
                sliceShape.size()
            ));
        }

        return views;
    }

private:
    void configureSessionOptions(bool useGPU, int numThreads) {
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
    }

    /// @brief Read node names, shapes and metadata from an already-created session
    /// @param modelLabel Human-readable model identifier used only for logging
    void introspectSession(const std::string& modelLabel) {
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
            isDynamicBatchSize_ = (inputTensorShape[0] <= 0);
            modelBatchSize_ = isDynamicBatchSize_ ? -1 : static_cast<int>(inputTensorShape[0]);
            isDynamicInputShape_ = (inputTensorShape[2] == -1 || inputTensorShape[3] == -1);

            inputChannels_ = (inputTensorShape[1] == -1) ? 3 : static_cast<int>(inputTensorShape[1]);
            int height = (inputTensorShape[2] == -1) ? 640 : static_cast<int>(inputTensorShape[2]);
            int width = (inputTensorShape[3] == -1) ? 640 : static_cast<int>(inputTensorShape[3]);
            inputShape_ = cv::Size(width, height);
        } else {
            throw std::runtime_error("Invalid input tensor shape. Expected 4D tensor [N, C, H, W].");
        }

        std::cout << "[INFO] Model loaded: " << modelLabel << std::endl;
        std::cout << "[INFO] Input shape: " << inputShape_.width << "x" << inputShape_.height
                  << (isDynamicInputShape_ ? " (dynamic)" : "") << std::endl;
        std::cout << "[INFO] Batch size: "
                  << (isDynamicBatchSize_ ? std::string("dynamic") : std::to_string(modelBatchSize_))
                  << std::endl;
        std::cout << "[INFO] Inputs: " << numInputNodes_ << ", Outputs: " << numOutputNodes_ << std::endl;

        try {
            exportedClassNamesFromMetadata_ = onnxmeta::tryGetExportedClassNames(session_);
        } catch (...) {
            exportedClassNamesFromMetadata_.clear();
        }
    }
};

} // namespace yolos
