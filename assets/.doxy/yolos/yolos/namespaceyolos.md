

# Namespace yolos



[**Namespace List**](namespaces.md) **>** [**yolos**](namespaceyolos.md)


















## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**cls**](namespaceyolos_1_1cls.md) <br> |
| namespace | [**det**](namespaceyolos_1_1det.md) <br> |
| namespace | [**drawing**](namespaceyolos_1_1drawing.md) <br> |
| namespace | [**nms**](namespaceyolos_1_1nms.md) <br> |
| namespace | [**obb**](namespaceyolos_1_1obb.md) <br> |
| namespace | [**pose**](namespaceyolos_1_1pose.md) <br> |
| namespace | [**preprocessing**](namespaceyolos_1_1preprocessing.md) <br> |
| namespace | [**seg**](namespaceyolos_1_1seg.md) <br> |
| namespace | [**utils**](namespaceyolos_1_1utils.md) <br> |
| namespace | [**version**](namespaceyolos_1_1version.md) <br> |


## Classes

| Type | Name |
| ---: | :--- |
| struct | [**BoundingBox**](structyolos_1_1BoundingBox.md) <br> |
| struct | [**KeyPoint**](structyolos_1_1KeyPoint.md) <br> |
| struct | [**OrientedBoundingBox**](structyolos_1_1OrientedBoundingBox.md) <br> |
| class | [**OrtSessionBase**](classyolos_1_1OrtSessionBase.md) <br>_Base class for ONNX Runtime session management Handles model loading, session configuration, and common inference setup._  |


## Public Types

| Type | Name |
| ---: | :--- |
| typedef [**cls::ClassificationResult**](structyolos_1_1cls_1_1ClassificationResult.md) | [**ClassificationResult**](#typedef-classificationresult)  <br> |
| typedef [**det::Detection**](structyolos_1_1det_1_1Detection.md) | [**Detection**](#typedef-detection)  <br> |
| typedef [**obb::OBBResult**](structyolos_1_1obb_1_1OBBResult.md) | [**OBBResult**](#typedef-obbresult)  <br> |
| typedef [**pose::PoseResult**](structyolos_1_1pose_1_1PoseResult.md) | [**PoseResult**](#typedef-poseresult)  <br> |
| typedef [**seg::Segmentation**](structyolos_1_1seg_1_1Segmentation.md) | [**Segmentation**](#typedef-segmentation)  <br> |
| typedef [**cls::YOLO26Classifier**](classyolos_1_1cls_1_1YOLO26Classifier.md) | [**YOLO26Classifier**](#typedef-yolo26classifier)  <br> |
| typedef [**det::YOLO26Detector**](classyolos_1_1det_1_1YOLO26Detector.md) | [**YOLO26Detector**](#typedef-yolo26detector)  <br> |
| typedef [**cls::YOLOClassifier**](classyolos_1_1cls_1_1YOLOClassifier.md) | [**YOLOClassifier**](#typedef-yoloclassifier)  <br> |
| typedef [**det::YOLODetector**](classyolos_1_1det_1_1YOLODetector.md) | [**YOLODetector**](#typedef-yolodetector)  <br> |
| typedef [**obb::YOLOOBBDetector**](classyolos_1_1obb_1_1YOLOOBBDetector.md) | [**YOLOOBBDetector**](#typedef-yoloobbdetector)  <br> |
| typedef [**pose::YOLOPoseDetector**](classyolos_1_1pose_1_1YOLOPoseDetector.md) | [**YOLOPoseDetector**](#typedef-yoloposedetector)  <br> |
| typedef [**seg::YOLOSegDetector**](classyolos_1_1seg_1_1YOLOSegDetector.md) | [**YOLOSegDetector**](#typedef-yolosegdetector)  <br> |
| enum  | [**YOLOVersion**](#enum-yoloversion)  <br> |




















## Public Functions

| Type | Name |
| ---: | :--- |
|  const std::vector&lt; std::pair&lt; int, int &gt; &gt; & | [**getPoseSkeleton**](#function-getposeskeleton) () <br> |




























## Public Types Documentation




### typedef ClassificationResult 

```C++
using yolos::ClassificationResult = typedef cls::ClassificationResult;
```




<hr>



### typedef Detection 

```C++
using yolos::Detection = typedef det::Detection;
```




<hr>



### typedef OBBResult 

```C++
using yolos::OBBResult = typedef obb::OBBResult;
```




<hr>



### typedef PoseResult 

```C++
using yolos::PoseResult = typedef pose::PoseResult;
```




<hr>



### typedef Segmentation 

```C++
using yolos::Segmentation = typedef seg::Segmentation;
```




<hr>



### typedef YOLO26Classifier 

```C++
using yolos::YOLO26Classifier = typedef cls::YOLO26Classifier;
```




<hr>



### typedef YOLO26Detector 

```C++
using yolos::YOLO26Detector = typedef det::YOLO26Detector;
```




<hr>



### typedef YOLOClassifier 

```C++
using yolos::YOLOClassifier = typedef cls::YOLOClassifier;
```




<hr>



### typedef YOLODetector 

```C++
using yolos::YOLODetector = typedef det::YOLODetector;
```




<hr>



### typedef YOLOOBBDetector 

```C++
using yolos::YOLOOBBDetector = typedef obb::YOLOOBBDetector;
```




<hr>



### typedef YOLOPoseDetector 

```C++
using yolos::YOLOPoseDetector = typedef pose::YOLOPoseDetector;
```




<hr>



### typedef YOLOSegDetector 

```C++
using yolos::YOLOSegDetector = typedef seg::YOLOSegDetector;
```




<hr>



### enum YOLOVersion 

```C++
enum yolos::YOLOVersion {
    Auto,
    V7,
    V8,
    V10,
    V11,
    V12,
    V26,
    NAS
};
```




<hr>
## Public Functions Documentation




### function getPoseSkeleton 

```C++
inline const std::vector< std::pair< int, int > > & yolos::getPoseSkeleton () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/core/drawing.hpp`

