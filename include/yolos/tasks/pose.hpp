#pragma once

// ============================================================================
// YOLO Pose Estimation
// ============================================================================
// Human pose estimation using YOLO models with keypoint detection.
// Supports YOLOv8-pose, YOLOv11-pose, and YOLO26-pose models.
//
// Authors: 
// YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// 2- Mohamed Samir, www.linkedin.com/in/mohamed-samir-7a730b237/
// ============================================================================

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <cmath>

#include "yolos/core/types.hpp"
#include "yolos/core/version.hpp"
#include "yolos/core/utils.hpp"
#include "yolos/core/preprocessing.hpp"
#include "yolos/core/nms.hpp"
#include "yolos/core/drawing.hpp"
#include "yolos/core/session_base.hpp"

namespace yolos {
namespace pose {

// ============================================================================
// Pose Result Structure
// ============================================================================

/// @brief Pose estimation result containing bounding box, confidence, and keypoints
struct PoseResult {
    BoundingBox box;               ///< Bounding box around the person
    float conf{0.0f};              ///< Detection confidence
    int classId{0};                ///< Class ID (typically 0 for person)
    std::vector<KeyPoint> keypoints; ///< Detected keypoints (17 for COCO format)

    PoseResult() = default;
    PoseResult(const BoundingBox& box_, float conf_, int classId_, const std::vector<KeyPoint>& kpts)
        : box(box_), conf(conf_), classId(classId_), keypoints(kpts) {}
};

// ============================================================================
// YOLOPoseDetector Class
// ============================================================================

/// @brief YOLO pose estimation detector with keypoint detection
class YOLOPoseDetector : public OrtSessionBase {
public:
    /// @brief Constructor
    /// @param modelPath Path to the ONNX model file
    /// @param labelsPath Path to the class names file (optional for pose)
    /// @param useGPU Whether to use GPU for inference
    YOLOPoseDetector(const std::string& modelPath,
                     const std::string& labelsPath = "",
                     bool useGPU = false)
        : OrtSessionBase(modelPath, useGPU) {
        
        if (!labelsPath.empty()) {
            classNames_ = utils::getClassNames(labelsPath);
        } else {
            classNames_ = {"person"};
        }
        classColors_ = drawing::generateColors(classNames_);
        
        // Pre-allocate inference buffer
        buffer_.ensureCapacity(inputShape_.height, inputShape_.width, 3);
    }

    virtual ~YOLOPoseDetector() = default;

    /// @brief Run pose detection on an image (optimized with buffer reuse)
    /// @param image Input image (BGR format)
    /// @param confThreshold Confidence threshold
    /// @param iouThreshold IoU threshold for NMS
    /// @return Vector of pose results
    std::vector<PoseResult> detect(const cv::Mat& image,
                                   float confThreshold = 0.4f,
                                   float iouThreshold = 0.5f) {
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

    /// @brief Draw pose estimations on an image
    /// @param image Image to draw on
    /// @param results Vector of pose results
    /// @param kptRadius Keypoint circle radius
    /// @param kptThreshold Minimum confidence to draw keypoint
    /// @param lineThickness Skeleton line thickness
    void drawPoses(cv::Mat& image,
                   const std::vector<PoseResult>& results,
                   int kptRadius = 4,
                   float kptThreshold = 0.5f,
                   int lineThickness = 2) const {
        
        for (const auto& pose : results) {
            // Draw bounding box
            cv::rectangle(image,
                         cv::Point(pose.box.x, pose.box.y),
                         cv::Point(pose.box.x + pose.box.width, pose.box.y + pose.box.height),
                         cv::Scalar(0, 255, 0), lineThickness);

            // Draw keypoints and skeleton
            drawing::drawPoseSkeleton(image, pose.keypoints, getPoseSkeleton(),
                                     kptRadius, kptThreshold, lineThickness);
        }
    }

    /// @brief Draw only skeletons (no bounding boxes)
    void drawSkeletonsOnly(cv::Mat& image,
                           const std::vector<PoseResult>& results,
                           int kptRadius = 4,
                           float kptThreshold = 0.5f,
                           int lineThickness = 2) const {
        for (const auto& pose : results) {
            drawing::drawPoseSkeleton(image, pose.keypoints, getPoseSkeleton(),
                                     kptRadius, kptThreshold, lineThickness);
        }
    }

    /// @brief Get class names
    [[nodiscard]] const std::vector<std::string>& getClassNames() const { return classNames_; }

    /// @brief Get COCO pose skeleton connections
    [[nodiscard]] static const std::vector<std::pair<int, int>>& getPoseSkeleton() {
        static const std::vector<std::pair<int, int>> skeleton = {
            {0, 1}, {0, 2}, {1, 3}, {2, 4},       // Face
            {3, 5}, {4, 6},                       // Head to shoulders
            {5, 7}, {7, 9}, {6, 8}, {8, 10},      // Arms
            {5, 6}, {5, 11}, {6, 12}, {11, 12},   // Body
            {11, 13}, {13, 15}, {12, 14}, {14, 16} // Legs
        };
        return skeleton;
    }

protected:
    std::vector<std::string> classNames_;
    std::vector<cv::Scalar> classColors_;
    static constexpr int NUM_KEYPOINTS = 17;
    static constexpr int FEATURES_PER_KEYPOINT = 3;
    
    // Pre-allocated buffer for inference
    mutable preprocessing::InferenceBuffer buffer_;

    /// @brief Postprocess pose detection outputs
    std::vector<PoseResult> postprocess(const cv::Size& originalSize,
                                        const cv::Size& resizedShape,
                                        const std::vector<Ort::Value>& outputTensors,
                                        float confThreshold,
                                        float iouThreshold) {
        const float* rawOutput = outputTensors[0].GetTensorData<float>();
        const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        // Detect output format based on shape:
        // YOLOv8/v11: [1, 56, num_detections] - requires NMS
        // YOLO26:     [1, 300, 57] - end-to-end, NMS-free
        const int expectedFeaturesV8 = 4 + 1 + NUM_KEYPOINTS * FEATURES_PER_KEYPOINT;  // 56
        const int expectedFeaturesV26 = 4 + 1 + 1 + NUM_KEYPOINTS * FEATURES_PER_KEYPOINT;  // 57

        if (outputShape.size() == 3 && outputShape[2] == expectedFeaturesV26) {
            // YOLO26 end-to-end format: [1, num_detections, 57]
            return postprocessV26(originalSize, resizedShape, rawOutput, outputShape, confThreshold);
        } else if (outputShape.size() == 3 && outputShape[1] == expectedFeaturesV8) {
            // YOLOv8/v11 format: [1, 56, num_detections]
            return postprocessV8(originalSize, resizedShape, rawOutput, outputShape, confThreshold, iouThreshold);
        } else {
            std::cerr << "[ERROR] Unsupported pose model output shape: [" 
                      << outputShape[0] << ", " << outputShape[1] << ", " << outputShape[2] << "]" << std::endl;
            return {};
        }
    }

    /// @brief Postprocess YOLOv8/v11 pose detection outputs (requires NMS)
    std::vector<PoseResult> postprocessV8(const cv::Size& originalSize,
                                          const cv::Size& resizedShape,
                                          const float* rawOutput,
                                          const std::vector<int64_t>& outputShape,
                                          float confThreshold,
                                          float iouThreshold) {
        std::vector<PoseResult> results;
        
        const size_t numDetections = outputShape[2];

        // Pre-compute scale and padding
        float scale, padX, padY;
        preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
        const float invScale = 1.0f / scale;

        std::vector<BoundingBox> boxes;
        std::vector<float> confidences;
        std::vector<std::vector<KeyPoint>> allKeypoints;
        boxes.reserve(64);
        confidences.reserve(64);
        allKeypoints.reserve(64);

        for (size_t d = 0; d < numDetections; ++d) {
            const float objConfidence = rawOutput[4 * numDetections + d];
            if (objConfidence < confThreshold) continue;

            // Decode bounding box (cx, cy, w, h format)
            const float cx = rawOutput[0 * numDetections + d];
            const float cy = rawOutput[1 * numDetections + d];
            const float w = rawOutput[2 * numDetections + d];
            const float h = rawOutput[3 * numDetections + d];

            // Convert to original image coordinates
            BoundingBox box;
            box.x = utils::clamp(static_cast<int>((cx - w * 0.5f - padX) * invScale), 0, originalSize.width - 1);
            box.y = utils::clamp(static_cast<int>((cy - h * 0.5f - padY) * invScale), 0, originalSize.height - 1);
            box.width = utils::clamp(static_cast<int>(w * invScale), 1, originalSize.width - box.x);
            box.height = utils::clamp(static_cast<int>(h * invScale), 1, originalSize.height - box.y);

            // Extract keypoints
            std::vector<KeyPoint> keypoints;
            keypoints.reserve(NUM_KEYPOINTS);
            for (int k = 0; k < NUM_KEYPOINTS; ++k) {
                const int offset = 5 + k * FEATURES_PER_KEYPOINT;
                KeyPoint kpt;
                kpt.x = (rawOutput[offset * numDetections + d] - padX) * invScale;
                kpt.y = (rawOutput[(offset + 1) * numDetections + d] - padY) * invScale;
                const float rawConf = rawOutput[(offset + 2) * numDetections + d];
                kpt.confidence = 1.0f / (1.0f + std::exp(-rawConf)); // Sigmoid

                // Clip keypoints to image boundaries
                kpt.x = utils::clamp(kpt.x, 0.0f, static_cast<float>(originalSize.width - 1));
                kpt.y = utils::clamp(kpt.y, 0.0f, static_cast<float>(originalSize.height - 1));

                keypoints.push_back(kpt);
            }

            boxes.push_back(box);
            confidences.push_back(objConfidence);
            allKeypoints.push_back(std::move(keypoints));
        }

        if (boxes.empty()) return results;

        // Apply NMS
        std::vector<int> indices;
        nms::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, indices);

        results.reserve(indices.size());
        for (int idx : indices) {
            results.emplace_back(boxes[idx], confidences[idx], 0, allKeypoints[idx]);
        }

        return results;
    }

    /// @brief Postprocess YOLO26 pose detection outputs (end-to-end, NMS-free)
    std::vector<PoseResult> postprocessV26(const cv::Size& originalSize,
                                           const cv::Size& resizedShape,
                                           const float* rawOutput,
                                           const std::vector<int64_t>& outputShape,
                                           float confThreshold) {
        std::vector<PoseResult> results;
        
        const size_t numDetections = outputShape[1];  // 300
        const size_t numFeatures = outputShape[2];    // 57

        // Pre-compute scale and padding
        float scale, padX, padY;
        preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
        const float invScale = 1.0f / scale;

        for (size_t d = 0; d < numDetections; ++d) {
            const size_t base = d * numFeatures;
            
            // YOLO26 format: [x1, y1, x2, y2, conf, class_id, kpt1_x, kpt1_y, kpt1_conf, ...]
            const float x1 = rawOutput[base + 0];
            const float y1 = rawOutput[base + 1];
            const float x2 = rawOutput[base + 2];
            const float y2 = rawOutput[base + 3];
            const float conf = rawOutput[base + 4];
            // const int classId = static_cast<int>(rawOutput[base + 5]);  // Always 0 for person

            if (conf < confThreshold) continue;

            // Convert to original image coordinates
            BoundingBox box;
            box.x = utils::clamp(static_cast<int>((x1 - padX) * invScale), 0, originalSize.width - 1);
            box.y = utils::clamp(static_cast<int>((y1 - padY) * invScale), 0, originalSize.height - 1);
            const int x2_scaled = utils::clamp(static_cast<int>((x2 - padX) * invScale), 0, originalSize.width - 1);
            const int y2_scaled = utils::clamp(static_cast<int>((y2 - padY) * invScale), 0, originalSize.height - 1);
            box.width = std::max(1, x2_scaled - box.x);
            box.height = std::max(1, y2_scaled - box.y);

            // Extract keypoints (starting at index 6)
            std::vector<KeyPoint> keypoints;
            keypoints.reserve(NUM_KEYPOINTS);
            for (int k = 0; k < NUM_KEYPOINTS; ++k) {
                const size_t kptBase = base + 6 + k * FEATURES_PER_KEYPOINT;
                KeyPoint kpt;
                kpt.x = (rawOutput[kptBase + 0] - padX) * invScale;
                kpt.y = (rawOutput[kptBase + 1] - padY) * invScale;
                kpt.confidence = rawOutput[kptBase + 2];  // Already sigmoid-applied in end-to-end

                // Clip keypoints to image boundaries
                kpt.x = utils::clamp(kpt.x, 0.0f, static_cast<float>(originalSize.width - 1));
                kpt.y = utils::clamp(kpt.y, 0.0f, static_cast<float>(originalSize.height - 1));

                keypoints.push_back(kpt);
            }

            results.emplace_back(box, conf, 0, std::move(keypoints));
        }

        return results;
    }
};

} // namespace pose
} // namespace yolos
