#pragma once

// ============================================================================
// YOLOs-CPP - Unified YOLO Inference Library
// ============================================================================
// Master include header for all YOLO tasks.
//
// Usage:
//   #include "yolos/yolos.hpp"  // Include all tasks
//   or include specific tasks:
//   #include "yolos/tasks/detection.hpp"
//   #include "yolos/tasks/segmentation.hpp"
//   #include "yolos/tasks/pose.hpp"
//   #include "yolos/tasks/obb.hpp"
//   #include "yolos/tasks/classification.hpp"
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

// Core components
#include "yolos/core/types.hpp"
#include "yolos/core/version.hpp"
#include "yolos/core/utils.hpp"
#include "yolos/core/preprocessing.hpp"
#include "yolos/core/nms.hpp"
#include "yolos/core/drawing.hpp"
#include "yolos/core/session_base.hpp"

// Task-specific implementations
#include "yolos/tasks/detection.hpp"
#include "yolos/tasks/segmentation.hpp"
#include "yolos/tasks/pose.hpp"
#include "yolos/tasks/obb.hpp"
#include "yolos/tasks/classification.hpp"

// ============================================================================
// Namespace Aliases for Convenience
// ============================================================================
namespace yolos {

// Detection task aliases
using Detection = det::Detection;
using YOLODetector = det::YOLODetector;
using YOLO26Detector = det::YOLO26Detector;

// Segmentation task aliases
using Segmentation = seg::Segmentation;
using YOLOSegDetector = seg::YOLOSegDetector;

// Pose estimation task aliases
using PoseResult = pose::PoseResult;
using YOLOPoseDetector = pose::YOLOPoseDetector;

// OBB detection task aliases
using OBBResult = obb::OBBResult;
using YOLOOBBDetector = obb::YOLOOBBDetector;

// Classification task aliases
using ClassificationResult = cls::ClassificationResult;
using YOLOClassifier = cls::YOLOClassifier;
using YOLO26Classifier = cls::YOLO26Classifier;

} // namespace yolos
