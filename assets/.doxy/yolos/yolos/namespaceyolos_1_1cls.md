

# Namespace yolos::cls



[**Namespace List**](namespaces.md) **>** [**yolos**](namespaceyolos.md) **>** [**cls**](namespaceyolos_1_1cls.md)




















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**ClassificationResult**](structyolos_1_1cls_1_1ClassificationResult.md) <br>_Classification result containing class ID, confidence, and class name._  |
| class | [**YOLO11Classifier**](classyolos_1_1cls_1_1YOLO11Classifier.md) <br>_YOLOv11 classifier._  |
| class | [**YOLO12Classifier**](classyolos_1_1cls_1_1YOLO12Classifier.md) <br>_YOLOv12 classifier._  |
| class | [**YOLO26Classifier**](classyolos_1_1cls_1_1YOLO26Classifier.md) <br>_YOLO26 classifier._  |
| class | [**YOLOClassifier**](classyolos_1_1cls_1_1YOLOClassifier.md) <br>_YOLO classifier for image classification._  |






















## Public Functions

| Type | Name |
| ---: | :--- |
|  std::unique\_ptr&lt; [**YOLOClassifier**](classyolos_1_1cls_1_1YOLOClassifier.md) &gt; | [**createClassifier**](#function-createclassifier) (const std::string & modelPath, const std::string & labelsPath, [**YOLOVersion**](namespaceyolos.md#enum-yoloversion) version=YOLOVersion::V11, bool useGPU=false) <br>_Create a classifier with explicit version selection._  |
|  void | [**drawClassificationResult**](#function-drawclassificationresult) (cv::Mat & image, const [**ClassificationResult**](structyolos_1_1cls_1_1ClassificationResult.md) & result, const cv::Point & position=cv::Point(10, 30), const cv::Scalar & textColor=cv::Scalar(0, 255, 0), const cv::Scalar & bgColor=cv::Scalar(0, 0, 0)) <br>_Draw classification result on an image._  |




























## Public Functions Documentation




### function createClassifier 

_Create a classifier with explicit version selection._ 
```C++
inline std::unique_ptr< YOLOClassifier > yolos::cls::createClassifier (
    const std::string & modelPath,
    const std::string & labelsPath,
    YOLOVersion version=YOLOVersion::V11,
    bool useGPU=false
) 
```





**Parameters:**


* `modelPath` Path to the ONNX model 
* `labelsPath` Path to the class names file 
* `version` YOLO version 
* `useGPU` Whether to use GPU 



**Returns:**

Unique pointer to classifier 





        

<hr>



### function drawClassificationResult 

_Draw classification result on an image._ 
```C++
inline void yolos::cls::drawClassificationResult (
    cv::Mat & image,
    const ClassificationResult & result,
    const cv::Point & position=cv::Point(10, 30),
    const cv::Scalar & textColor=cv::Scalar(0, 255, 0),
    const cv::Scalar & bgColor=cv::Scalar(0, 0, 0)
) 
```





**Parameters:**


* `image` Image to draw on 
* `result` Classification result 
* `position` Position for the text 
* `textColor` Text color 
* `bgColor` Background color 




        

<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/tasks/classification.hpp`

