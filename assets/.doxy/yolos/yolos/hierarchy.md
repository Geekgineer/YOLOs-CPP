
# Class Hierarchy

This inheritance list is sorted roughly, but not completely, alphabetically:


* **class** [**yolos::OrtSessionBase**](classyolos_1_1OrtSessionBase.md) _Base class for ONNX Runtime session management Handles model loading, session configuration, and common inference setup._     
    * **class** [**yolos::det::YOLODetector**](classyolos_1_1det_1_1YOLODetector.md) _Base YOLO detector with runtime version auto-detection._     
        * **class** [**yolos::det::YOLO26Detector**](classyolos_1_1det_1_1YOLO26Detector.md) _YOLOv26 detector (forces V26 end-to-end postprocessing)_ 
        * **class** [**yolos::det::YOLONASDetector**](classyolos_1_1det_1_1YOLONASDetector.md) _YOLO-NAS detector (forces NAS postprocessing)_ 
        * **class** [**yolos::det::YOLOv10Detector**](classyolos_1_1det_1_1YOLOv10Detector.md) _YOLOv10 detector (forces V10 end-to-end postprocessing)_ 
        * **class** [**yolos::det::YOLOv11Detector**](classyolos_1_1det_1_1YOLOv11Detector.md) _YOLOv11 detector (forces standard postprocessing)_ 
        * **class** [**yolos::det::YOLOv7Detector**](classyolos_1_1det_1_1YOLOv7Detector.md) _YOLOv7 detector (forces V7 postprocessing)_ 
        * **class** [**yolos::det::YOLOv8Detector**](classyolos_1_1det_1_1YOLOv8Detector.md) _YOLOv8 detector (forces standard postprocessing)_ 
    * **class** [**yolos::obb::YOLOOBBDetector**](classyolos_1_1obb_1_1YOLOOBBDetector.md) _YOLO oriented bounding box detector for rotated object detection._ 
    * **class** [**yolos::pose::YOLOPoseDetector**](classyolos_1_1pose_1_1YOLOPoseDetector.md) _YOLO pose estimation detector with keypoint detection._ 
    * **class** [**yolos::seg::YOLOSegDetector**](classyolos_1_1seg_1_1YOLOSegDetector.md) _YOLO segmentation detector with mask prediction._ 
* **class** [**yolos::cls::YOLOClassifier**](classyolos_1_1cls_1_1YOLOClassifier.md) _YOLO classifier for image classification._     
    * **class** [**yolos::cls::YOLO11Classifier**](classyolos_1_1cls_1_1YOLO11Classifier.md) _YOLOv11 classifier._ 
    * **class** [**yolos::cls::YOLO12Classifier**](classyolos_1_1cls_1_1YOLO12Classifier.md) _YOLOv12 classifier._ 
    * **class** [**yolos::cls::YOLO26Classifier**](classyolos_1_1cls_1_1YOLO26Classifier.md) _YOLO26 classifier._ 
* **struct** [**yolos::BoundingBox**](structyolos_1_1BoundingBox.md) 
* **struct** [**yolos::KeyPoint**](structyolos_1_1KeyPoint.md) 
* **struct** [**yolos::OrientedBoundingBox**](structyolos_1_1OrientedBoundingBox.md) 
* **struct** [**yolos::cls::ClassificationResult**](structyolos_1_1cls_1_1ClassificationResult.md) _Classification result containing class ID, confidence, and class name._ 
* **struct** [**yolos::det::Detection**](structyolos_1_1det_1_1Detection.md) [_**Detection**_](structyolos_1_1det_1_1Detection.md) _result containing bounding box, confidence, and class ID._
* **struct** [**yolos::obb::OBBResult**](structyolos_1_1obb_1_1OBBResult.md) _OBB detection result containing oriented bounding box, confidence, and class ID._ 
* **struct** [**yolos::pose::PoseResult**](structyolos_1_1pose_1_1PoseResult.md) _Pose estimation result containing bounding box, confidence, and keypoints._ 
* **struct** [**yolos::preprocessing::InferenceBuffer**](structyolos_1_1preprocessing_1_1InferenceBuffer.md) _Pre-allocated inference buffer to avoid per-frame allocations._ 
* **struct** [**yolos::seg::Segmentation**](structyolos_1_1seg_1_1Segmentation.md) [_**Segmentation**_](structyolos_1_1seg_1_1Segmentation.md) _result containing bounding box, confidence, class ID, and mask._

