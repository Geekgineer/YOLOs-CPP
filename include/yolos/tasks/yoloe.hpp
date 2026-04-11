#pragma once

// ============================================================================
// YOLOE: Real-Time Seeing Anything
// ============================================================================
// Open-vocabulary detection and instance segmentation using YOLOE models.
// Text-prompt exports: class list is fixed at ONNX export (set_classes before export).
// Prompt-free (-pf): large fixed vocabulary; labels file must match that export.
// Visual prompt inference is Ultralytics Python-first; export afterward for C++ deployment.
//
// YOLOE is architecturally identical to its base YOLO backbone:
//   - yoloe-11x-seg  → YOLO11 backbone → V11 output format (standard)
//   - yoloe-26x-seg  → YOLO26 backbone → V26 output format (end-to-end, NMS-free)
//
// Key capabilities over standard YOLO:
//   - Dynamic class vocabulary: set classes via vector<string> or at export time
//   - Agnostic NMS (default): class-unaware suppression for large vocabularies
//   - setClasses() API: switch vocabulary without reloading the model
//
// Export workflow (Python):
//   model = YOLOE("yoloe-26s-seg.pt")
//   model.set_classes(["person", "car", "bus"])
//   model.export(format="onnx", nms=False)
//
// Authors:
//   YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

#include <opencv2/opencv.hpp>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "yolos/tasks/detection.hpp"
#include "yolos/tasks/segmentation.hpp"
#include "yolos/core/drawing.hpp"
#include "yolos/core/utils.hpp"

namespace yolos {
namespace yoloe {

namespace detail {

/// Inline class names must match the ONNX export (same count as model.set_classes() in Python).
inline void validateInlineClassesMatchOnnxExport(const std::vector<std::string>& user,
                                                 const std::vector<std::string>& exportedFromMetadata) {
    if (exportedFromMetadata.empty()) {
        return;
    }
    if (user.size() != exportedFromMetadata.size()) {
        std::ostringstream os;
        os << "YOLOE: passed " << user.size() << " class name(s), but this ONNX was exported with "
           << exportedFromMetadata.size() << ".\n"
           << "Exported vocabulary (from ONNX metadata): ";
        for (size_t i = 0; i < exportedFromMetadata.size(); ++i) {
            if (i) os << ", ";
            os << "'" << exportedFromMetadata[i] << "'";
        }
        os << "\nUse the same list and order as in Python `model.set_classes([...])` before export, "
              "or create a new ONNX (see scripts/export_yoloe_classes.py).";
        throw std::invalid_argument(os.str());
    }
}

} // namespace detail

// ============================================================================
// YOLOEDetector
// ============================================================================

/// @brief Open-vocabulary YOLO detector.
///
/// Wraps YOLODetector with:
///   - Inline class names for text-prompt ONNX (must match Python export)
///   - Agnostic NMS enabled by default (recommended for large vocabularies)
///   - setClasses() to relabel fixed output channels (same cardinality as export)
///
/// Compatible with any YOLOE-11x or YOLOE-26x detection model exported to ONNX.
class YOLOEDetector : public det::YOLODetector {
public:
    /// @brief Construct from inline class names (text-prompt or few-class mode)
    /// @param modelPath Path to the ONNX model (exported after set_classes())
    /// @param classNames Class names that were used during model.set_classes() in Python
    /// @param useGPU Whether to use CUDA GPU for inference
    /// @param agnosticNms Use class-agnostic NMS (recommended; suppresses across classes)
    YOLOEDetector(const std::string& modelPath,
                  const std::vector<std::string>& classNames,
                  bool useGPU = false,
                  bool agnosticNms = true)
        : det::YOLODetector(modelPath, "", useGPU, YOLOVersion::Auto)
    {
        if (classNames.empty()) {
            throw std::invalid_argument("YOLOEDetector: classNames must not be empty");
        }
        detail::validateInlineClassesMatchOnnxExport(classNames, getExportedClassNamesFromMetadata());
        classNames_  = classNames;
        classColors_ = drawing::generateColors(classNames_);
        agnosticNms_ = agnosticNms;
    }

    /// @brief Construct from a labels file (prompt-free / large fixed vocabulary)
    /// @param modelPath Path to the ONNX model (prompt-free, e.g. yoloe-26s-seg-pf.pt exported)
    /// @param labelsPath One class name per line; line count must match the PF ONNX export
    /// @param useGPU Whether to use CUDA GPU for inference
    /// @param agnosticNms Use class-agnostic NMS (recommended; suppresses across classes)
    YOLOEDetector(const std::string& modelPath,
                  const std::string& labelsPath,
                  bool useGPU = false,
                  bool agnosticNms = true)
        : det::YOLODetector(modelPath, labelsPath, useGPU, YOLOVersion::Auto)
    {
        agnosticNms_ = agnosticNms;
    }

    virtual ~YOLOEDetector() = default;

    /// @brief Relabel fixed output channels without reloading the model.
    ///
    /// Same count and order as export; not for adding new concepts without a new ONNX.
    /// @param classNames Names aligned to the model's output channels
    void setClasses(const std::vector<std::string>& classNames) {
        if (classNames.empty()) {
            throw std::invalid_argument("YOLOEDetector::setClasses: classNames must not be empty");
        }
        detail::validateInlineClassesMatchOnnxExport(classNames, getExportedClassNamesFromMetadata());
        classNames_  = classNames;
        classColors_ = drawing::generateColors(classNames_);
    }
};

// ============================================================================
// YOLOESegDetector
// ============================================================================

/// @brief Open-vocabulary YOLO instance segmentation detector.
///
/// Wraps YOLOSegDetector with:
///   - Inline class names for text-prompt ONNX (must match Python export)
///   - Agnostic NMS enabled by default (recommended for large vocabularies)
///   - setClasses() to relabel fixed output channels (same cardinality as export)
///
/// Compatible with any YOLOE-11x-seg or YOLOE-26x-seg model exported to ONNX.
/// Segmentation masks are returned in results[i].mask (CV_8UC1, original image size).
class YOLOESegDetector : public seg::YOLOSegDetector {
public:
    /// @brief Construct from inline class names (text-prompt or few-class mode)
    /// @param modelPath Path to the ONNX model (exported after set_classes())
    /// @param classNames Class names that were used during model.set_classes() in Python
    /// @param useGPU Whether to use CUDA GPU for inference
    /// @param agnosticNms Use class-agnostic NMS (recommended; suppresses across classes)
    YOLOESegDetector(const std::string& modelPath,
                     const std::vector<std::string>& classNames,
                     bool useGPU = false,
                     bool agnosticNms = true)
        : seg::YOLOSegDetector(modelPath, "", useGPU)
    {
        if (classNames.empty()) {
            throw std::invalid_argument("YOLOESegDetector: classNames must not be empty");
        }
        detail::validateInlineClassesMatchOnnxExport(classNames, getExportedClassNamesFromMetadata());
        classNames_  = classNames;
        classColors_ = drawing::generateColors(classNames_);
        agnosticNms_ = agnosticNms;
    }

    /// @brief Construct from a labels file (prompt-free / large fixed vocabulary)
    /// @param modelPath Path to the ONNX model (prompt-free, e.g. yoloe-26s-seg-pf.pt exported)
    /// @param labelsPath One class name per line; line count must match the PF ONNX export
    /// @param useGPU Whether to use CUDA GPU for inference
    /// @param agnosticNms Use class-agnostic NMS (recommended; suppresses across classes)
    YOLOESegDetector(const std::string& modelPath,
                     const std::string& labelsPath,
                     bool useGPU = false,
                     bool agnosticNms = true)
        : seg::YOLOSegDetector(modelPath, labelsPath, useGPU)
    {
        agnosticNms_ = agnosticNms;
    }

    virtual ~YOLOESegDetector() = default;

    /// @brief Relabel fixed output channels without reloading the model.
    ///
    /// Same count and order as export; not for adding new concepts without a new ONNX.
    /// @param classNames Names aligned to the model's output channels
    void setClasses(const std::vector<std::string>& classNames) {
        if (classNames.empty()) {
            throw std::invalid_argument("YOLOESegDetector::setClasses: classNames must not be empty");
        }
        detail::validateInlineClassesMatchOnnxExport(classNames, getExportedClassNamesFromMetadata());
        classNames_  = classNames;
        classColors_ = drawing::generateColors(classNames_);
    }
};

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Create a YOLOE open-vocabulary detector from inline class names.
/// @param modelPath Path to the ONNX model
/// @param classNames Class names used when exporting the model
/// @param useGPU Whether to use CUDA GPU
/// @param agnosticNms Use class-agnostic NMS (default true)
/// @return Unique pointer to YOLOEDetector
inline std::unique_ptr<YOLOEDetector> createYOLOEDetector(
    const std::string& modelPath,
    const std::vector<std::string>& classNames,
    bool useGPU = false,
    bool agnosticNms = true)
{
    return std::make_unique<YOLOEDetector>(modelPath, classNames, useGPU, agnosticNms);
}

/// @brief Create a YOLOE open-vocabulary detector from a labels file.
/// @param modelPath Path to the ONNX model
/// @param labelsPath One name per line (prompt-free: count must match the ONNX)
/// @param useGPU Whether to use CUDA GPU
/// @param agnosticNms Use class-agnostic NMS (default true)
/// @return Unique pointer to YOLOEDetector
inline std::unique_ptr<YOLOEDetector> createYOLOEDetector(
    const std::string& modelPath,
    const std::string& labelsPath,
    bool useGPU = false,
    bool agnosticNms = true)
{
    return std::make_unique<YOLOEDetector>(modelPath, labelsPath, useGPU, agnosticNms);
}

/// @brief Create a YOLOE open-vocabulary segmentation detector from inline class names.
/// @param modelPath Path to the ONNX model
/// @param classNames Class names used when exporting the model
/// @param useGPU Whether to use CUDA GPU
/// @param agnosticNms Use class-agnostic NMS (default true)
/// @return Unique pointer to YOLOESegDetector
inline std::unique_ptr<YOLOESegDetector> createYOLOESegDetector(
    const std::string& modelPath,
    const std::vector<std::string>& classNames,
    bool useGPU = false,
    bool agnosticNms = true)
{
    return std::make_unique<YOLOESegDetector>(modelPath, classNames, useGPU, agnosticNms);
}

/// @brief Create a YOLOE open-vocabulary segmentation detector from a labels file.
/// @param modelPath Path to the ONNX model
/// @param labelsPath One name per line (prompt-free: count must match the ONNX)
/// @param useGPU Whether to use CUDA GPU
/// @param agnosticNms Use class-agnostic NMS (default true)
/// @return Unique pointer to YOLOESegDetector
inline std::unique_ptr<YOLOESegDetector> createYOLOESegDetector(
    const std::string& modelPath,
    const std::string& labelsPath,
    bool useGPU = false,
    bool agnosticNms = true)
{
    return std::make_unique<YOLOESegDetector>(modelPath, labelsPath, useGPU, agnosticNms);
}

} // namespace yoloe
} // namespace yolos
