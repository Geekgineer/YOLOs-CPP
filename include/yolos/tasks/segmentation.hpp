#pragma once

// ============================================================================
// YOLO Instance Segmentation
// ============================================================================
// Instance segmentation using YOLO models with mask prediction.
// Supports YOLOv8-seg and YOLOv11-seg models.
//
// Authors: 
// YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// 2- Mohamed Samir, www.linkedin.com/in/mohamed-samir-7a730b237/
// ============================================================================

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

#include "yolos/core/types.hpp"
#include "yolos/core/version.hpp"
#include "yolos/core/utils.hpp"
#include "yolos/core/preprocessing.hpp"
#include "yolos/core/nms.hpp"
#include "yolos/core/drawing.hpp"
#include "yolos/core/session_base.hpp"

namespace yolos {
namespace seg {

// ============================================================================
// Segmentation Result Structure
// ============================================================================

/// @brief Segmentation result containing bounding box, confidence, class ID, and mask
struct Segmentation {
    BoundingBox box;       ///< Axis-aligned bounding box
    float conf{0.0f};      ///< Confidence score
    int classId{0};        ///< Class ID
    cv::Mat mask;          ///< Binary mask (CV_8UC1) in original image coordinates

    Segmentation() = default;
    Segmentation(const BoundingBox& box_, float conf_, int classId_, const cv::Mat& mask_)
        : box(box_), conf(conf_), classId(classId_), mask(mask_) {}
};

// ============================================================================
// YOLOSegDetector Class
// ============================================================================

/// @brief YOLO segmentation detector with mask prediction
class YOLOSegDetector : public OrtSessionBase {
public:
    /// @brief Constructor
    /// @param modelPath Path to the ONNX model file
    /// @param labelsPath Path to the class names file
    /// @param useGPU Whether to use GPU for inference
    YOLOSegDetector(const std::string& modelPath,
                    const std::string& labelsPath,
                    bool useGPU = false)
        : OrtSessionBase(modelPath, useGPU) {
        
        // Validate output count for segmentation models
        if (numOutputNodes_ != 2) {
            throw std::runtime_error("Expected 2 output nodes for segmentation model (output0 and output1)");
        }
        
        classNames_ = utils::getClassNames(labelsPath);
        classColors_ = drawing::generateColors(classNames_);
        
        // Pre-allocate inference buffer
        buffer_.ensureCapacity(inputShape_.height, inputShape_.width, 3);
    }

    virtual ~YOLOSegDetector() = default;

    /// @brief Run segmentation on an image (optimized with buffer reuse)
    /// @param image Input image (BGR format)
    /// @param confThreshold Confidence threshold
    /// @param iouThreshold IoU threshold for NMS
    /// @return Vector of segmentation results
    std::vector<Segmentation> segment(const cv::Mat& image,
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

        // Postprocess
        return postprocess(image.size(), actualSize, outputTensors, confThreshold, iouThreshold);
    }

    /// @brief Draw segmentations with boxes and labels on an image
    /// @param image Image to draw on
    /// @param results Vector of segmentation results
    /// @param maskAlpha Mask transparency (0-1)
    void drawSegmentations(cv::Mat& image,
                           const std::vector<Segmentation>& results,
                           float maskAlpha = 0.5f) const {
        for (const auto& seg : results) {
            if (seg.classId < 0 || static_cast<size_t>(seg.classId) >= classNames_.size()) {
                continue;
            }
            
            const cv::Scalar& color = classColors_[seg.classId % classColors_.size()];

            // Draw mask
            if (!seg.mask.empty()) {
                drawing::drawSegmentationMask(image, seg.mask, color, maskAlpha);
            }

            // Draw bounding box and label
            std::string label = classNames_[seg.classId] + ": " +
                               std::to_string(static_cast<int>(seg.conf * 100)) + "%";
            drawing::drawBoundingBox(image, seg.box, label, color);
        }
    }

    /// @brief Draw only segmentation masks (no boxes)
    void drawMasksOnly(cv::Mat& image,
                       const std::vector<Segmentation>& results,
                       float maskAlpha = 0.5f) const {
        for (const auto& seg : results) {
            if (seg.classId < 0 || static_cast<size_t>(seg.classId) >= classNames_.size()) {
                continue;
            }
            const cv::Scalar& color = classColors_[seg.classId % classColors_.size()];
            if (!seg.mask.empty()) {
                drawing::drawSegmentationMask(image, seg.mask, color, maskAlpha);
            }
        }
    }

    /// @brief Get class names
    [[nodiscard]] const std::vector<std::string>& getClassNames() const { return classNames_; }

    /// @brief Get class colors
    [[nodiscard]] const std::vector<cv::Scalar>& getClassColors() const { return classColors_; }

protected:
    std::vector<std::string> classNames_;
    std::vector<cv::Scalar> classColors_;
    static constexpr float MASK_THRESHOLD = 0.5f;
    
    // Pre-allocated buffer for inference (avoids per-frame allocations)
    mutable preprocessing::InferenceBuffer buffer_;

    /// @brief Postprocess segmentation outputs
    std::vector<Segmentation> postprocess(const cv::Size& originalSize,
                                          const cv::Size& letterboxSize,
                                          const std::vector<Ort::Value>& outputTensors,
                                          float confThreshold,
                                          float iouThreshold) {
        std::vector<Segmentation> results;

        if (outputTensors.size() < 2) {
            return results;
        }

        const float* output0 = outputTensors[0].GetTensorData<float>();
        const float* output1 = outputTensors[1].GetTensorData<float>();

        auto shape0 = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        auto shape1 = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape(); // [1, 32, maskH, maskW]

        if (shape1.size() != 4 || shape1[1] != 32) {
            throw std::runtime_error("Unexpected mask output shape. Expected [1, 32, maskH, maskW]");
        }

        // Detect YOLO26-seg format: [1, num_detections, 38] where dim2 == 38
        // vs standard format: [1, num_features, num_detections] where dim1 > dim2
        const bool isV26Format = (shape0.size() == 3 && shape0[2] == 38);

        if (isV26Format) {
            return postprocessV26(originalSize, letterboxSize, output0, output1, shape0, shape1, confThreshold);
        }

        // Standard format: [1, 116, num_detections]
        const size_t numFeatures = shape0[1];
        const size_t numDetections = shape0[2];

        if (numDetections == 0) return results;

        const int numClasses = static_cast<int>(numFeatures) - 4 - 32;
        if (numClasses <= 0) return results;

        const int maskH = static_cast<int>(shape1[2]);
        const int maskW = static_cast<int>(shape1[3]);

        // Load prototype masks
        std::vector<cv::Mat> prototypeMasks;
        prototypeMasks.reserve(32);
        for (int m = 0; m < 32; ++m) {
            cv::Mat proto(maskH, maskW, CV_32F, const_cast<float*>(output1 + m * maskH * maskW));
            prototypeMasks.emplace_back(proto.clone());
        }

        // Pre-compute scale and padding for descaling AFTER NMS
        float gain, padW, padH;
        preprocessing::getScalePad(originalSize, letterboxSize, gain, padW, padH);
        const float invGain = 1.0f / gain;

        // Process detections in LETTERBOX coordinates with FLOAT precision (NMS applied before descaling)
        std::vector<cv::Rect2f> letterboxBoxes;  // Float boxes in letterbox space for NMS
        std::vector<float> confidences;
        std::vector<int> classIds;
        std::vector<std::vector<float>> maskCoeffsList;
        letterboxBoxes.reserve(256);
        confidences.reserve(256);
        classIds.reserve(256);
        maskCoeffsList.reserve(256);

        for (size_t i = 0; i < numDetections; ++i) {
            const float xc = output0[0 * numDetections + i];
            const float yc = output0[1 * numDetections + i];
            const float w = output0[2 * numDetections + i];
            const float h = output0[3 * numDetections + i];

            // Find max class score
            int classId = 0;
            float maxConf = output0[4 * numDetections + i];
            for (int c = 1; c < numClasses; ++c) {
                const float conf = output0[(4 + c) * numDetections + i];
                if (conf > maxConf) {
                    maxConf = conf;
                    classId = c;
                }
            }

            if (maxConf < confThreshold) continue;

            // Store box in LETTERBOX coordinates as FLOAT (for precise NMS)
            cv::Rect2f box(xc - w * 0.5f, yc - h * 0.5f, w, h);

            letterboxBoxes.push_back(box);
            confidences.push_back(maxConf);
            classIds.push_back(classId);

            std::vector<float> maskCoeffs(32);
            for (int m = 0; m < 32; ++m) {
                maskCoeffs[m] = output0[(4 + numClasses + m) * numDetections + i];
            }
            maskCoeffsList.emplace_back(std::move(maskCoeffs));
        }

        if (letterboxBoxes.empty()) return results;

        // Apply class-aware (batched) NMS on LETTERBOX coordinates with FLOAT precision
        std::vector<int> nmsIndices;
        nms::NMSBoxesFBatched(letterboxBoxes, confidences, classIds, confThreshold, iouThreshold, nmsIndices);

        if (nmsIndices.empty()) return results;

        float maskScaleX = static_cast<float>(maskW) / letterboxSize.width;
        float maskScaleY = static_cast<float>(maskH) / letterboxSize.height;

        results.reserve(nmsIndices.size());

        for (int idx : nmsIndices) {
            Segmentation seg;
            
            // NOW descale box coordinates from letterbox to original
            const cv::Rect2f& lbBox = letterboxBoxes[idx];
            const float left = (lbBox.x - padW) * invGain;
            const float top = (lbBox.y - padH) * invGain;
            const float scaledW = lbBox.width * invGain;
            const float scaledH = lbBox.height * invGain;
            
            seg.box.x = utils::clamp(static_cast<int>(left), 0, originalSize.width - 1);
            seg.box.y = utils::clamp(static_cast<int>(top), 0, originalSize.height - 1);
            seg.box.width = utils::clamp(static_cast<int>(scaledW), 1, originalSize.width - seg.box.x);
            seg.box.height = utils::clamp(static_cast<int>(scaledH), 1, originalSize.height - seg.box.y);
            seg.conf = confidences[idx];
            seg.classId = classIds[idx];

            // Compute mask from prototype masks and coefficients
            cv::Mat finalMask = cv::Mat::zeros(maskH, maskW, CV_32F);
            for (int m = 0; m < 32; ++m) {
                finalMask += maskCoeffsList[idx][m] * prototypeMasks[m];
            }

            // Apply sigmoid activation
            cv::exp(-finalMask, finalMask);
            finalMask = 1.0 / (1.0 + finalMask);

            // Crop to letterbox area
            int x1 = static_cast<int>(std::round((padW - 0.1f) * maskScaleX));
            int y1 = static_cast<int>(std::round((padH - 0.1f) * maskScaleY));
            int x2 = static_cast<int>(std::round((letterboxSize.width - padW + 0.1f) * maskScaleX));
            int y2 = static_cast<int>(std::round((letterboxSize.height - padH + 0.1f) * maskScaleY));

            x1 = std::max(0, std::min(x1, maskW - 1));
            y1 = std::max(0, std::min(y1, maskH - 1));
            x2 = std::max(x1, std::min(x2, maskW));
            y2 = std::max(y1, std::min(y2, maskH));

            if (x2 <= x1 || y2 <= y1) continue;

            cv::Mat croppedMask = finalMask(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();

            // Resize to original image size
            cv::Mat resizedMask;
            cv::resize(croppedMask, resizedMask, originalSize, 0, 0, cv::INTER_LINEAR);

            // Threshold and convert to binary
            cv::Mat binaryMask;
            cv::threshold(resizedMask, binaryMask, MASK_THRESHOLD, 255.0, cv::THRESH_BINARY);
            binaryMask.convertTo(binaryMask, CV_8U);

            // Crop to bounding box
            cv::Mat finalBinaryMask = cv::Mat::zeros(originalSize, CV_8U);
            cv::Rect roi(seg.box.x, seg.box.y, seg.box.width, seg.box.height);
            roi &= cv::Rect(0, 0, binaryMask.cols, binaryMask.rows);
            if (roi.area() > 0) {
                binaryMask(roi).copyTo(finalBinaryMask(roi));
            }

            seg.mask = finalBinaryMask;
            results.push_back(seg);
        }

        return results;
    }

    /// @brief Postprocess YOLO26-seg format outputs (end-to-end, no NMS needed)
    /// Output0 shape: [1, num_detections, 38] where 38 = 4 (x1,y1,x2,y2) + 1 (conf) + 1 (class_id) + 32 (mask_coeffs)
    std::vector<Segmentation> postprocessV26(const cv::Size& originalSize,
                                              const cv::Size& letterboxSize,
                                              const float* output0,
                                              const float* output1,
                                              const std::vector<int64_t>& shape0,
                                              const std::vector<int64_t>& shape1,
                                              float confThreshold) {
        std::vector<Segmentation> results;

        const size_t numDetections = shape0[1];
        const size_t numFeaturesPerDet = shape0[2]; // Should be 38

        if (numDetections == 0 || numFeaturesPerDet != 38) return results;

        const int maskH = static_cast<int>(shape1[2]);
        const int maskW = static_cast<int>(shape1[3]);

        // Load prototype masks
        std::vector<cv::Mat> prototypeMasks;
        prototypeMasks.reserve(32);
        for (int m = 0; m < 32; ++m) {
            cv::Mat proto(maskH, maskW, CV_32F, const_cast<float*>(output1 + m * maskH * maskW));
            prototypeMasks.emplace_back(proto.clone());
        }

        // Pre-compute scale and padding
        float gain, padW, padH;
        preprocessing::getScalePad(originalSize, letterboxSize, gain, padW, padH);
        const float invGain = 1.0f / gain;

        float maskScaleX = static_cast<float>(maskW) / letterboxSize.width;
        float maskScaleY = static_cast<float>(maskH) / letterboxSize.height;

        results.reserve(numDetections);

        for (size_t i = 0; i < numDetections; ++i) {
            const float* det = output0 + i * numFeaturesPerDet;

            // V26 format: [x1, y1, x2, y2, conf, class_id, mask_coeffs(32)]
            const float x1_lb = det[0];
            const float y1_lb = det[1];
            const float x2_lb = det[2];
            const float y2_lb = det[3];
            const float conf = det[4];
            const int classId = static_cast<int>(det[5]);

            if (conf < confThreshold) continue;

            // Descale from letterbox to original coordinates
            const float x1 = (x1_lb - padW) * invGain;
            const float y1 = (y1_lb - padH) * invGain;
            const float x2 = (x2_lb - padW) * invGain;
            const float y2 = (y2_lb - padH) * invGain;

            Segmentation seg;
            seg.box.x = utils::clamp(static_cast<int>(x1), 0, originalSize.width - 1);
            seg.box.y = utils::clamp(static_cast<int>(y1), 0, originalSize.height - 1);
            seg.box.width = utils::clamp(static_cast<int>(x2 - x1), 1, originalSize.width - seg.box.x);
            seg.box.height = utils::clamp(static_cast<int>(y2 - y1), 1, originalSize.height - seg.box.y);
            seg.conf = conf;
            seg.classId = classId;

            // Extract mask coefficients
            std::vector<float> maskCoeffs(32);
            for (int m = 0; m < 32; ++m) {
                maskCoeffs[m] = det[6 + m];
            }

            // Compute mask from prototype masks and coefficients
            cv::Mat finalMask = cv::Mat::zeros(maskH, maskW, CV_32F);
            for (int m = 0; m < 32; ++m) {
                finalMask += maskCoeffs[m] * prototypeMasks[m];
            }

            // Apply sigmoid activation
            cv::exp(-finalMask, finalMask);
            finalMask = 1.0 / (1.0 + finalMask);

            // Crop to letterbox area
            int mx1 = static_cast<int>(std::round((padW - 0.1f) * maskScaleX));
            int my1 = static_cast<int>(std::round((padH - 0.1f) * maskScaleY));
            int mx2 = static_cast<int>(std::round((letterboxSize.width - padW + 0.1f) * maskScaleX));
            int my2 = static_cast<int>(std::round((letterboxSize.height - padH + 0.1f) * maskScaleY));

            mx1 = std::max(0, std::min(mx1, maskW - 1));
            my1 = std::max(0, std::min(my1, maskH - 1));
            mx2 = std::max(mx1, std::min(mx2, maskW));
            my2 = std::max(my1, std::min(my2, maskH));

            if (mx2 <= mx1 || my2 <= my1) continue;

            cv::Mat croppedMask = finalMask(cv::Rect(mx1, my1, mx2 - mx1, my2 - my1)).clone();

            // Resize to original image size
            cv::Mat resizedMask;
            cv::resize(croppedMask, resizedMask, originalSize, 0, 0, cv::INTER_LINEAR);

            // Threshold and convert to binary
            cv::Mat binaryMask;
            cv::threshold(resizedMask, binaryMask, MASK_THRESHOLD, 255.0, cv::THRESH_BINARY);
            binaryMask.convertTo(binaryMask, CV_8U);

            // Crop to bounding box
            cv::Mat finalBinaryMask = cv::Mat::zeros(originalSize, CV_8U);
            cv::Rect roi(seg.box.x, seg.box.y, seg.box.width, seg.box.height);
            roi &= cv::Rect(0, 0, binaryMask.cols, binaryMask.rows);
            if (roi.area() > 0) {
                binaryMask(roi).copyTo(finalBinaryMask(roi));
            }

            seg.mask = finalBinaryMask;
            results.push_back(seg);
        }

        return results;
    }
};

} // namespace seg
} // namespace yolos
