#pragma once

// ============================================================================
// YOLO Monocular Depth Estimation
// ============================================================================
// Per-pixel metric depth estimation using YOLO26-depth models.
//
// The exported ONNX graph already contains the clamp, exp, log-affine
// calibration and 4x upsample from Ultralytics' Depth head, so output0 is
// dense metric depth in meters at the model input resolution. Postprocessing
// is only: crop the letterbox padding, rescale to the original image size.
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <opencv2/opencv.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "yolos/core/drawing.hpp"
#include "yolos/core/preprocessing.hpp"
#include "yolos/core/session_base.hpp"
#include "yolos/core/types.hpp"
#include "yolos/core/utils.hpp"

namespace yolos {
namespace depth {

// ============================================================================
// YOLODepthEstimator
// ============================================================================

/// @brief Monocular metric depth estimator for YOLO26-depth models
///
/// Depth has no classes, no confidence and no IoU: the model emits one dense
/// map, so this class intentionally exposes none of those parameters.
class YOLODepthEstimator : public OrtSessionBase {
public:
    /// @brief Constructor
    /// @param modelPath Path to the ONNX model file
    /// @param useGPU Whether to use GPU (CUDA) for inference
    /// @throws std::runtime_error if the model does not look like a depth export
    explicit YOLODepthEstimator(const std::string& modelPath, bool useGPU = false)
        : OrtSessionBase(modelPath, useGPU) {

        validateDepthOutput();

        // Pre-allocate inference buffer
        buffer_.ensureCapacity(inputShape_.height, inputShape_.width, inputChannels_);
    }

    virtual ~YOLODepthEstimator() = default;

    /// @brief Estimate per-pixel metric depth
    /// @param image Input image (BGR format)
    /// @return CV_32FC1 depth in meters, sized to @p image; empty if @p image is empty
    /// @note Values are metric meters straight from the model. They are never
    ///       normalized or rescaled here; use drawDepth() / drawing::drawDepthMap()
    ///       to visualize.
    cv::Mat estimate(const cv::Mat& image) {
        if (image.empty()) {
            return cv::Mat();
        }

        cv::Size actualSize;
        preprocessing::letterBoxToBlob(image, buffer_, inputChannels_, inputShape_,
                                       actualSize, isDynamicInputShape_);

        const std::vector<int64_t> inputTensorShape = {
            1, static_cast<int64_t>(inputChannels_), actualSize.height, actualSize.width
        };
        Ort::Value inputTensor = createInputTensor(buffer_.blob.data(), inputTensorShape);

        std::vector<Ort::Value> outputTensors = runInference(inputTensor);

        return postprocess(image.size(), outputTensors);
    }

    /// @brief Blend a colorized depth map over an image
    /// @param image Image to draw on, modified in place
    /// @param depth Depth map from estimate()
    /// @param alpha Blend factor for the heatmap
    /// @param cmap Colormap to apply
    /// @param mode Disparity or metric normalization
    /// @note Forwards to drawing::drawDepthMap(). For video, prefer calling that
    ///       directly with explicit vmin/vmax: per-frame percentile normalization
    ///       makes the overlay flicker.
    void drawDepth(cv::Mat& image,
                   const cv::Mat& depth,
                   float alpha = 0.6f,
                   drawing::DepthColormap cmap = drawing::DepthColormap::Jet,
                   drawing::DepthNorm mode = drawing::DepthNorm::Disparity) const {
        drawing::drawDepthMap(image, depth, alpha, cmap, mode);
    }

protected:
    // Pre-allocated buffer for inference (avoids per-frame allocations)
    mutable preprocessing::InferenceBuffer buffer_;

    /// @brief Crop letterbox padding and rescale the dense map to the original size
    /// @param originalSize Size of the un-letterboxed input image
    /// @param outputTensors Tensors returned by runInference()
    /// @return CV_32FC1 depth in meters at @p originalSize
    cv::Mat postprocess(const cv::Size& originalSize,
                        const std::vector<Ort::Value>& outputTensors) {
        if (outputTensors.empty()) {
            return cv::Mat();
        }

        const std::vector<int64_t> shape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4 || shape[1] != 1) {
            throw std::runtime_error(
                "Unexpected depth output shape. Expected [1, 1, H, W].");
        }

        const int mapH = static_cast<int>(shape[2]);
        const int mapW = static_cast<int>(shape[3]);

        // Wrap the tensor without copying; cv::resize makes the only copy.
        const cv::Mat mapView(mapH, mapW, CV_32FC1,
                              const_cast<float*>(outputTensors[0].GetTensorData<float>()));

        cv::Mat result;
        preprocessing::cropLetterboxAndResize(mapView, result, originalSize);
        return result;
    }

private:
    /// @brief Fail fast when handed a model that is not a depth export
    void validateDepthOutput() {
        if (numOutputNodes_ != 1) {
            throw std::runtime_error(
                "Expected 1 output node for a depth model, got " + std::to_string(numOutputNodes_) +
                ". A YOLO26-depth export has a single output of shape [1, 1, H, W].");
        }

        const std::vector<int64_t> shape =
            session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        // Dimension 1 may be -1 on a dynamic export; anything else must be exactly 1.
        const bool rankOk = shape.size() == 4;
        const bool channelOk = rankOk && (shape[1] == 1 || shape[1] == -1);
        if (!rankOk || !channelOk) {
            std::string got = "[";
            for (size_t i = 0; i < shape.size(); ++i) {
                got += std::to_string(shape[i]);
                if (i + 1 < shape.size()) got += ", ";
            }
            got += "]";
            throw std::runtime_error(
                "Model output shape " + got + " is not a depth map. A YOLO26-depth export "
                "has a single output of shape [1, 1, H, W].");
        }
    }
};

// ============================================================================
// Factory Function
// ============================================================================

/// @brief Create a depth estimator
/// @param modelPath Path to the ONNX model
/// @param useGPU Whether to use GPU
/// @return Unique pointer to estimator
inline std::unique_ptr<YOLODepthEstimator> createDepthEstimator(const std::string& modelPath,
                                                                bool useGPU = false) {
    return std::make_unique<YOLODepthEstimator>(modelPath, useGPU);
}

} // namespace depth
} // namespace yolos
