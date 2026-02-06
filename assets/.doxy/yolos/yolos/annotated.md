
# Class List


Here are the classes, structs, unions and interfaces with brief descriptions:

* **namespace** [**yolos**](namespaceyolos.md)     
    * **struct** [**BoundingBox**](structyolos_1_1BoundingBox.md)     
    * **struct** [**KeyPoint**](structyolos_1_1KeyPoint.md)     
    * **struct** [**OrientedBoundingBox**](structyolos_1_1OrientedBoundingBox.md)     
    * **class** [**OrtSessionBase**](classyolos_1_1OrtSessionBase.md) _Base class for ONNX Runtime session management Handles model loading, session configuration, and common inference setup._     
    * **namespace** [**cls**](namespaceyolos_1_1cls.md)     
        * **struct** [**ClassificationResult**](structyolos_1_1cls_1_1ClassificationResult.md) _Classification result containing class ID, confidence, and class name._     
        * **class** [**YOLO11Classifier**](classyolos_1_1cls_1_1YOLO11Classifier.md) _YOLOv11 classifier._     
        * **class** [**YOLO12Classifier**](classyolos_1_1cls_1_1YOLO12Classifier.md) _YOLOv12 classifier._     
        * **class** [**YOLO26Classifier**](classyolos_1_1cls_1_1YOLO26Classifier.md) _YOLO26 classifier._     
        * **class** [**YOLOClassifier**](classyolos_1_1cls_1_1YOLOClassifier.md) _YOLO classifier for image classification._     
    * **namespace** [**det**](namespaceyolos_1_1det.md)     
        * **struct** [**Detection**](structyolos_1_1det_1_1Detection.md) [_**Detection**_](structyolos_1_1det_1_1Detection.md) _result containing bounding box, confidence, and class ID._    
        * **class** [**YOLO26Detector**](classyolos_1_1det_1_1YOLO26Detector.md) _YOLOv26 detector (forces V26 end-to-end postprocessing)_     
        * **class** [**YOLODetector**](classyolos_1_1det_1_1YOLODetector.md) _Base YOLO detector with runtime version auto-detection._     
        * **class** [**YOLONASDetector**](classyolos_1_1det_1_1YOLONASDetector.md) _YOLO-NAS detector (forces NAS postprocessing)_     
        * **class** [**YOLOv10Detector**](classyolos_1_1det_1_1YOLOv10Detector.md) _YOLOv10 detector (forces V10 end-to-end postprocessing)_     
        * **class** [**YOLOv11Detector**](classyolos_1_1det_1_1YOLOv11Detector.md) _YOLOv11 detector (forces standard postprocessing)_     
        * **class** [**YOLOv7Detector**](classyolos_1_1det_1_1YOLOv7Detector.md) _YOLOv7 detector (forces V7 postprocessing)_     
        * **class** [**YOLOv8Detector**](classyolos_1_1det_1_1YOLOv8Detector.md) _YOLOv8 detector (forces standard postprocessing)_     
    * **namespace** [**drawing**](namespaceyolos_1_1drawing.md)     
    * **namespace** [**nms**](namespaceyolos_1_1nms.md)     
    * **namespace** [**obb**](namespaceyolos_1_1obb.md)     
        * **struct** [**OBBResult**](structyolos_1_1obb_1_1OBBResult.md) _OBB detection result containing oriented bounding box, confidence, and class ID._     
        * **class** [**YOLOOBBDetector**](classyolos_1_1obb_1_1YOLOOBBDetector.md) _YOLO oriented bounding box detector for rotated object detection._     
    * **namespace** [**pose**](namespaceyolos_1_1pose.md)     
        * **struct** [**PoseResult**](structyolos_1_1pose_1_1PoseResult.md) _Pose estimation result containing bounding box, confidence, and keypoints._     
        * **class** [**YOLOPoseDetector**](classyolos_1_1pose_1_1YOLOPoseDetector.md) _YOLO pose estimation detector with keypoint detection._     
    * **namespace** [**preprocessing**](namespaceyolos_1_1preprocessing.md)     
        * **struct** [**InferenceBuffer**](structyolos_1_1preprocessing_1_1InferenceBuffer.md) _Pre-allocated inference buffer to avoid per-frame allocations._     
    * **namespace** [**seg**](namespaceyolos_1_1seg.md)     
        * **struct** [**Segmentation**](structyolos_1_1seg_1_1Segmentation.md) [_**Segmentation**_](structyolos_1_1seg_1_1Segmentation.md) _result containing bounding box, confidence, class ID, and mask._    
        * **class** [**YOLOSegDetector**](classyolos_1_1seg_1_1YOLOSegDetector.md) _YOLO segmentation detector with mask prediction._     
    * **namespace** [**utils**](namespaceyolos_1_1utils.md)     
    * **namespace** [**version**](namespaceyolos_1_1version.md)     

