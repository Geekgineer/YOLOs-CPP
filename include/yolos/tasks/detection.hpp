#pragma once

// ============================================================================
// YOLO Object Detection
// ============================================================================
// Object detection using YOLO models with support for multiple versions
// (v7, v8, v10, v11, NAS) through runtime auto-detection or explicit selection.
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <cfloat>

#include "yolos/core/types.hpp"
#include "yolos/core/version.hpp"
#include "yolos/core/utils.hpp"
#include "yolos/core/preprocessing.hpp"
#include "yolos/core/nms.hpp"
#include "yolos/core/drawing.hpp"
#include "yolos/core/session_base.hpp"

namespace yolos {
namespace det {

// ============================================================================
// Detection Result Structure
// ============================================================================

/// @brief Detection result containing bounding box, confidence, and class ID
struct Detection {
    BoundingBox box;    ///< Axis-aligned bounding box
    float conf{0.0f};   ///< Confidence score
    int classId{-1};    ///< Class ID

    Detection() = default;
    Detection(const BoundingBox& box_, float conf_, int classId_)
        : box(box_), conf(conf_), classId(classId_) {}
};

// ============================================================================
// YOLODetector Base Class
// ============================================================================

/// @brief Base YOLO detector with runtime version auto-detection
class YOLODetector : public OrtSessionBase {
public:
    /// @brief Constructor
    /// @param modelPath Path to the ONNX model file
    /// @param labelsPath Path to the class names file
    /// @param useGPU Whether to use GPU for inference
    /// @param version YOLO version (Auto for runtime detection)
    YOLODetector(const std::string& modelPath,
                 const std::string& labelsPath,
                 bool useGPU = false,
                 YOLOVersion version = YOLOVersion::Auto)
        : OrtSessionBase(modelPath, useGPU),
          version_(version) {
        classNames_ = utils::getClassNames(labelsPath);
        classColors_ = drawing::generateColors(classNames_);
        
        // Pre-allocate inference buffer
        buffer_.ensureCapacity(inputShape_.height, inputShape_.width, 3);
    }

    virtual ~YOLODetector() = default;

    /// @brief Run detection on an image (optimized with buffer reuse)
    /// @param image Input image (BGR format)
    /// @param confThreshold Confidence threshold
    /// @param iouThreshold IoU threshold for NMS
    /// @return Vector of detections
    virtual std::vector<Detection> detect(const cv::Mat& image,
                                          float confThreshold = 0.4f,
                                          float iouThreshold = 0.45f) {
        // Optimized preprocessing with buffer reuse
        cv::Size actualSize;
        preprocessing::letterBoxToBlob(image, buffer_, inputShape_, actualSize, isDynamicInputShape_);
        
        // Create input tensor (uses pre-allocated blob)
        std::vector<int64_t> inputTensorShape = {1, 3, actualSize.height, actualSize.width};
        Ort::Value inputTensor = createInputTensor(buffer_.blob.data(), inputTensorShape);

        // Run inference
        std::vector<Ort::Value> outputTensors = runInference(inputTensor);

        // Determine version if auto
        YOLOVersion effectiveVersion = version_;
        if (effectiveVersion == YOLOVersion::Auto) {
            effectiveVersion = detectVersion(outputTensors);
        }

        // Postprocess based on version
        return postprocess(image.size(), actualSize, outputTensors, effectiveVersion, confThreshold, iouThreshold);
    }

    /// @brief Draw detections on an image
    /// @param image Image to draw on
    /// @param detections Vector of detections
    void drawDetections(cv::Mat& image, const std::vector<Detection>& detections) const {
        for (const auto& det : detections) {
            if (det.classId >= 0 && static_cast<size_t>(det.classId) < classNames_.size()) {
                std::string label = classNames_[det.classId] + ": " +
                                   std::to_string(static_cast<int>(det.conf * 100)) + "%";
                const cv::Scalar& color = classColors_[det.classId % classColors_.size()];
                drawing::drawBoundingBox(image, det.box, label, color);
            }
        }
    }

    /// @brief Draw detections with semi-transparent mask fill
    void drawDetectionsWithMask(cv::Mat& image, const std::vector<Detection>& detections, float alpha = 0.4f) const {
        for (const auto& det : detections) {
            if (det.classId >= 0 && static_cast<size_t>(det.classId) < classNames_.size()) {
                std::string label = classNames_[det.classId] + ": " +
                                   std::to_string(static_cast<int>(det.conf * 100)) + "%";
                const cv::Scalar& color = classColors_[det.classId % classColors_.size()];
                drawing::drawBoundingBoxWithMask(image, det.box, label, color, alpha);
            }
        }
    }

    /// @brief Get class names
    [[nodiscard]] const std::vector<std::string>& getClassNames() const { return classNames_; }

    /// @brief Get class colors
    [[nodiscard]] const std::vector<cv::Scalar>& getClassColors() const { return classColors_; }

protected:
    YOLOVersion version_{YOLOVersion::Auto};
    std::vector<std::string> classNames_;
    std::vector<cv::Scalar> classColors_;
    
    // Pre-allocated buffer for inference (avoids per-frame allocations)
    mutable preprocessing::InferenceBuffer buffer_;

    /// @brief Detect YOLO version from output tensors
    YOLOVersion detectVersion(const std::vector<Ort::Value>& outputTensors) {
        const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        return version::detectFromOutputShape(outputShape, outputTensors.size());
    }

    /// @brief Postprocess based on detected version
    virtual std::vector<Detection> postprocess(const cv::Size& originalSize,
                                               const cv::Size& resizedShape,
                                               const std::vector<Ort::Value>& outputTensors,
                                               YOLOVersion version,
                                               float confThreshold,
                                               float iouThreshold) {
        switch (version) {
            case YOLOVersion::V7:
                return postprocessV7(originalSize, resizedShape, outputTensors, confThreshold, iouThreshold);
            case YOLOVersion::V10:
            case YOLOVersion::V26:
                return postprocessV10(originalSize, resizedShape, outputTensors, confThreshold, iouThreshold);
            case YOLOVersion::NAS:
                return postprocessNAS(originalSize, resizedShape, outputTensors, confThreshold, iouThreshold);
            default:
                return postprocessStandard(originalSize, resizedShape, outputTensors, confThreshold, iouThreshold);
        }
    }

    /// @brief Standard postprocess for YOLOv8/v11 format [batch, features, boxes]
    /// Optimized: single box storage with batched NMS
    virtual std::vector<Detection> postprocessStandard(const cv::Size& originalSize,
                                                       const cv::Size& resizedShape,
                                                       const std::vector<Ort::Value>& outputTensors,
                                                       float confThreshold,
                                                       float iouThreshold) {
        std::vector<Detection> detections;
        const float* rawOutput = outputTensors[0].GetTensorData<float>();
        const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        const size_t numFeatures = outputShape[1];
        const size_t numDetections = outputShape[2];

        if (numDetections == 0) return detections;

        const int numClasses = static_cast<int>(numFeatures) - 4;
        if (numClasses <= 0) return detections;

        // Pre-compute scale and padding once
        float scale, padX, padY;
        preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
        const float invScale = 1.0f / scale;

        std::vector<BoundingBox> boxes;
        std::vector<float> confs;
        std::vector<int> classIds;
        boxes.reserve(256);  // Reasonable initial capacity
        confs.reserve(256);
        classIds.reserve(256);

        for (size_t d = 0; d < numDetections; ++d) {
            // Quick reject: check if any class score could exceed threshold
            // by checking box coordinates are valid first
            const float centerX = rawOutput[0 * numDetections + d];
            const float centerY = rawOutput[1 * numDetections + d];
            const float width = rawOutput[2 * numDetections + d];
            const float height = rawOutput[3 * numDetections + d];

            // Find max class score
            int classId = 0;
            float maxScore = rawOutput[4 * numDetections + d];
            for (int c = 1; c < numClasses; ++c) {
                const float score = rawOutput[(4 + c) * numDetections + d];
                if (score > maxScore) {
                    maxScore = score;
                    classId = c;
                }
            }

            if (maxScore > confThreshold) {
                // Convert center to corner and descale in one step
                const float left = (centerX - width * 0.5f - padX) * invScale;
                const float top = (centerY - height * 0.5f - padY) * invScale;
                const float w = width * invScale;
                const float h = height * invScale;

                // Clip to image bounds
                BoundingBox box;
                box.x = utils::clamp(static_cast<int>(left), 0, originalSize.width - 1);
                box.y = utils::clamp(static_cast<int>(top), 0, originalSize.height - 1);
                box.width = utils::clamp(static_cast<int>(w), 1, originalSize.width - box.x);
                box.height = utils::clamp(static_cast<int>(h), 1, originalSize.height - box.y);

                boxes.push_back(box);
                confs.push_back(maxScore);
                classIds.push_back(classId);
            }
        }

        // Batched NMS (handles class offsets internally)
        std::vector<int> indices;
        nms::NMSBoxesBatched(boxes, confs, classIds, confThreshold, iouThreshold, indices);

        detections.reserve(indices.size());
        for (int idx : indices) {
            detections.emplace_back(boxes[idx], confs[idx], classIds[idx]);
        }

        return detections;
    }

    /// @brief Postprocess for YOLOv7 format [batch, boxes, features]
    virtual std::vector<Detection> postprocessV7(const cv::Size& originalSize,
                                                 const cv::Size& resizedShape,
                                                 const std::vector<Ort::Value>& outputTensors,
                                                 float confThreshold,
                                                 float iouThreshold) {
        std::vector<Detection> detections;
        const float* rawOutput = outputTensors[0].GetTensorData<float>();
        const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        const size_t numDetections = outputShape[1];
        const size_t numFeatures = outputShape[2];

        if (numDetections == 0) return detections;

        const int numClasses = static_cast<int>(numFeatures) - 5;
        if (numClasses <= 0) return detections;

        // Pre-compute scale and padding
        float scale, padX, padY;
        preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
        const float invScale = 1.0f / scale;

        std::vector<BoundingBox> boxes;
        std::vector<float> confs;
        std::vector<int> classIds;
        boxes.reserve(256);
        confs.reserve(256);
        classIds.reserve(256);

        for (size_t d = 0; d < numDetections; ++d) {
            const float objConf = rawOutput[d * numFeatures + 4];
            if (objConf <= confThreshold) continue;

            const float centerX = rawOutput[d * numFeatures + 0];
            const float centerY = rawOutput[d * numFeatures + 1];
            const float width = rawOutput[d * numFeatures + 2];
            const float height = rawOutput[d * numFeatures + 3];

            int classId = 0;
            float maxScore = rawOutput[d * numFeatures + 5];
            for (int c = 1; c < numClasses; ++c) {
                const float score = rawOutput[d * numFeatures + 5 + c];
                if (score > maxScore) {
                    maxScore = score;
                    classId = c;
                }
            }

            // Convert and descale in one step
            const float left = (centerX - width * 0.5f - padX) * invScale;
            const float top = (centerY - height * 0.5f - padY) * invScale;

            BoundingBox box;
            box.x = utils::clamp(static_cast<int>(left), 0, originalSize.width - 1);
            box.y = utils::clamp(static_cast<int>(top), 0, originalSize.height - 1);
            box.width = utils::clamp(static_cast<int>(width * invScale), 1, originalSize.width - box.x);
            box.height = utils::clamp(static_cast<int>(height * invScale), 1, originalSize.height - box.y);

            boxes.push_back(box);
            confs.push_back(objConf);
            classIds.push_back(classId);
        }

        std::vector<int> indices;
        nms::NMSBoxesBatched(boxes, confs, classIds, confThreshold, iouThreshold, indices);

        detections.reserve(indices.size());
        for (int idx : indices) {
            detections.emplace_back(boxes[idx], confs[idx], classIds[idx]);
        }

        return detections;
    }

    /// @brief Postprocess for YOLOv10 format [batch, boxes, 6] (end-to-end, no NMS needed)
    virtual std::vector<Detection> postprocessV10(const cv::Size& originalSize,
                                                  const cv::Size& resizedShape,
                                                  const std::vector<Ort::Value>& outputTensors,
                                                  float confThreshold,
                                                  float /*iouThreshold*/) {
        std::vector<Detection> detections;
        const float* rawOutput = outputTensors[0].GetTensorData<float>();
        const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        const int numDetections = static_cast<int>(outputShape[1]);

        // Pre-compute scale and padding
        float scale, padX, padY;
        preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
        const float invScale = 1.0f / scale;

        detections.reserve(numDetections);

        for (int i = 0; i < numDetections; ++i) {
            const float confidence = rawOutput[i * 6 + 4];
            if (confidence <= confThreshold) continue;

            const float x1 = (rawOutput[i * 6 + 0] - padX) * invScale;
            const float y1 = (rawOutput[i * 6 + 1] - padY) * invScale;
            const float x2 = (rawOutput[i * 6 + 2] - padX) * invScale;
            const float y2 = (rawOutput[i * 6 + 3] - padY) * invScale;
            const int classId = static_cast<int>(rawOutput[i * 6 + 5]);

            BoundingBox box;
            box.x = utils::clamp(static_cast<int>(x1), 0, originalSize.width - 1);
            box.y = utils::clamp(static_cast<int>(y1), 0, originalSize.height - 1);
            box.width = utils::clamp(static_cast<int>(x2 - x1), 1, originalSize.width - box.x);
            box.height = utils::clamp(static_cast<int>(y2 - y1), 1, originalSize.height - box.y);

            detections.emplace_back(box, confidence, classId);
        }

        return detections;
    }

    /// @brief Postprocess for YOLO-NAS format (two outputs: boxes and scores)
    virtual std::vector<Detection> postprocessNAS(const cv::Size& originalSize,
                                                  const cv::Size& resizedShape,
                                                  const std::vector<Ort::Value>& outputTensors,
                                                  float confThreshold,
                                                  float iouThreshold) {
        std::vector<Detection> detections;

        if (outputTensors.size() < 2) {
            return postprocessStandard(originalSize, resizedShape, outputTensors, confThreshold, iouThreshold);
        }

        const float* boxOutput = outputTensors[0].GetTensorData<float>();
        const float* scoreOutput = outputTensors[1].GetTensorData<float>();
        const std::vector<int64_t> boxShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        const std::vector<int64_t> scoreShape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();

        const int numDetections = static_cast<int>(boxShape[1]);
        const int numClasses = static_cast<int>(scoreShape[2]);

        // Pre-compute scale and padding
        float scale, padX, padY;
        preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
        const float invScale = 1.0f / scale;

        std::vector<BoundingBox> boxes;
        std::vector<float> confs;
        std::vector<int> classIds;
        boxes.reserve(256);
        confs.reserve(256);
        classIds.reserve(256);

        for (int i = 0; i < numDetections; ++i) {
            // Find max class first (allows early continue if below threshold)
            int classId = 0;
            float maxScore = scoreOutput[i * numClasses];
            for (int c = 1; c < numClasses; ++c) {
                const float score = scoreOutput[i * numClasses + c];
                if (score > maxScore) {
                    maxScore = score;
                    classId = c;
                }
            }

            if (maxScore <= confThreshold) continue;

            const float x1 = (boxOutput[i * 4 + 0] - padX) * invScale;
            const float y1 = (boxOutput[i * 4 + 1] - padY) * invScale;
            const float x2 = (boxOutput[i * 4 + 2] - padX) * invScale;
            const float y2 = (boxOutput[i * 4 + 3] - padY) * invScale;

            BoundingBox box;
            box.x = utils::clamp(static_cast<int>(x1), 0, originalSize.width - 1);
            box.y = utils::clamp(static_cast<int>(y1), 0, originalSize.height - 1);
            box.width = utils::clamp(static_cast<int>(x2 - x1), 1, originalSize.width - box.x);
            box.height = utils::clamp(static_cast<int>(y2 - y1), 1, originalSize.height - box.y);

            boxes.push_back(box);
            confs.push_back(maxScore);
            classIds.push_back(classId);
        }

        std::vector<int> indices;
        nms::NMSBoxesBatched(boxes, confs, classIds, confThreshold, iouThreshold, indices);

        detections.reserve(indices.size());
        for (int idx : indices) {
            detections.emplace_back(boxes[idx], confs[idx], classIds[idx]);
        }

        return detections;
    }
};

// ============================================================================
// Version-Specific Detector Subclasses
// ============================================================================

/// @brief YOLOv7 detector (forces V7 postprocessing)
class YOLOv7Detector : public YOLODetector {
public:
    YOLOv7Detector(const std::string& modelPath, const std::string& labelsPath, bool useGPU = false)
        : YOLODetector(modelPath, labelsPath, useGPU, YOLOVersion::V7) {}
};

/// @brief YOLOv8 detector (forces standard postprocessing)
class YOLOv8Detector : public YOLODetector {
public:
    YOLOv8Detector(const std::string& modelPath, const std::string& labelsPath, bool useGPU = false)
        : YOLODetector(modelPath, labelsPath, useGPU, YOLOVersion::V8) {}
};

/// @brief YOLOv10 detector (forces V10 end-to-end postprocessing)
class YOLOv10Detector : public YOLODetector {
public:
    YOLOv10Detector(const std::string& modelPath, const std::string& labelsPath, bool useGPU = false)
        : YOLODetector(modelPath, labelsPath, useGPU, YOLOVersion::V10) {}
};

/// @brief YOLOv11 detector (forces standard postprocessing)
class YOLOv11Detector : public YOLODetector {
public:
    YOLOv11Detector(const std::string& modelPath, const std::string& labelsPath, bool useGPU = false)
        : YOLODetector(modelPath, labelsPath, useGPU, YOLOVersion::V11) {}
};

/// @brief YOLO-NAS detector (forces NAS postprocessing)
class YOLONASDetector : public YOLODetector {
public:
    YOLONASDetector(const std::string& modelPath, const std::string& labelsPath, bool useGPU = false)
        : YOLODetector(modelPath, labelsPath, useGPU, YOLOVersion::NAS) {}
};

/// @brief YOLOv26 detector (forces V26 end-to-end postprocessing)
class YOLO26Detector : public YOLODetector {
public:
    YOLO26Detector(const std::string& modelPath, const std::string& labelsPath, bool useGPU = false)
        : YOLODetector(modelPath, labelsPath, useGPU, YOLOVersion::V26) {}
};

// ============================================================================
// Factory Function
// ============================================================================

/// @brief Create a detector with explicit version selection
/// @param modelPath Path to the ONNX model
/// @param labelsPath Path to the class names file
/// @param version YOLO version (Auto for runtime detection)
/// @param useGPU Whether to use GPU
/// @return Unique pointer to detector
inline std::unique_ptr<YOLODetector> createDetector(const std::string& modelPath,
                                                    const std::string& labelsPath,
                                                    YOLOVersion version = YOLOVersion::Auto,
                                                    bool useGPU = false) {
    switch (version) {
        case YOLOVersion::V7:
            return std::make_unique<YOLOv7Detector>(modelPath, labelsPath, useGPU);
        case YOLOVersion::V8:
            return std::make_unique<YOLOv8Detector>(modelPath, labelsPath, useGPU);
        case YOLOVersion::V10:
            return std::make_unique<YOLOv10Detector>(modelPath, labelsPath, useGPU);
        case YOLOVersion::V11:
            return std::make_unique<YOLOv11Detector>(modelPath, labelsPath, useGPU);
        case YOLOVersion::V26:
            return std::make_unique<YOLO26Detector>(modelPath, labelsPath, useGPU);
        case YOLOVersion::NAS:
            return std::make_unique<YOLONASDetector>(modelPath, labelsPath, useGPU);
        default:
            return std::make_unique<YOLODetector>(modelPath, labelsPath, useGPU, YOLOVersion::Auto);
    }
}

} // namespace det
} // namespace yolos
