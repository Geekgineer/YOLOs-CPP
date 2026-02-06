

# Namespace yolos::det



[**Namespace List**](namespaces.md) **>** [**yolos**](namespaceyolos.md) **>** [**det**](namespaceyolos_1_1det.md)




















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**Detection**](structyolos_1_1det_1_1Detection.md) <br>[_**Detection**_](structyolos_1_1det_1_1Detection.md) _result containing bounding box, confidence, and class ID._ |
| class | [**YOLO26Detector**](classyolos_1_1det_1_1YOLO26Detector.md) <br>_YOLOv26 detector (forces V26 end-to-end postprocessing)_  |
| class | [**YOLODetector**](classyolos_1_1det_1_1YOLODetector.md) <br>_Base YOLO detector with runtime version auto-detection._  |
| class | [**YOLONASDetector**](classyolos_1_1det_1_1YOLONASDetector.md) <br>_YOLO-NAS detector (forces NAS postprocessing)_  |
| class | [**YOLOv10Detector**](classyolos_1_1det_1_1YOLOv10Detector.md) <br>_YOLOv10 detector (forces V10 end-to-end postprocessing)_  |
| class | [**YOLOv11Detector**](classyolos_1_1det_1_1YOLOv11Detector.md) <br>_YOLOv11 detector (forces standard postprocessing)_  |
| class | [**YOLOv7Detector**](classyolos_1_1det_1_1YOLOv7Detector.md) <br>_YOLOv7 detector (forces V7 postprocessing)_  |
| class | [**YOLOv8Detector**](classyolos_1_1det_1_1YOLOv8Detector.md) <br>_YOLOv8 detector (forces standard postprocessing)_  |






















## Public Functions

| Type | Name |
| ---: | :--- |
|  std::unique\_ptr&lt; [**YOLODetector**](classyolos_1_1det_1_1YOLODetector.md) &gt; | [**createDetector**](#function-createdetector) (const std::string & modelPath, const std::string & labelsPath, [**YOLOVersion**](namespaceyolos.md#enum-yoloversion) version=YOLOVersion::Auto, bool useGPU=false) <br>_Create a detector with explicit version selection._  |




























## Public Functions Documentation




### function createDetector 

_Create a detector with explicit version selection._ 
```C++
inline std::unique_ptr< YOLODetector > yolos::det::createDetector (
    const std::string & modelPath,
    const std::string & labelsPath,
    YOLOVersion version=YOLOVersion::Auto,
    bool useGPU=false
) 
```





**Parameters:**


* `modelPath` Path to the ONNX model 
* `labelsPath` Path to the class names file 
* `version` YOLO version (Auto for runtime detection) 
* `useGPU` Whether to use GPU 



**Returns:**

Unique pointer to detector 





        

<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/tasks/detection.hpp`

