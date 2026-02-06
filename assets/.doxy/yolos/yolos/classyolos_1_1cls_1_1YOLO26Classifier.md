

# Class yolos::cls::YOLO26Classifier



[**ClassList**](annotated.md) **>** [**yolos**](namespaceyolos.md) **>** [**cls**](namespaceyolos_1_1cls.md) **>** [**YOLO26Classifier**](classyolos_1_1cls_1_1YOLO26Classifier.md)



_YOLO26 classifier._ 

* `#include <classification.hpp>`



Inherits the following classes: [yolos::cls::YOLOClassifier](classyolos_1_1cls_1_1YOLOClassifier.md)






















































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**YOLO26Classifier**](#function-yolo26classifier) (const std::string & modelPath, const std::string & labelsPath, bool useGPU=false) <br> |


## Public Functions inherited from yolos::cls::YOLOClassifier

See [yolos::cls::YOLOClassifier](classyolos_1_1cls_1_1YOLOClassifier.md)

| Type | Name |
| ---: | :--- |
|   | [**YOLOClassifier**](classyolos_1_1cls_1_1YOLOClassifier.md#function-yoloclassifier) (const std::string & modelPath, const std::string & labelsPath, bool useGPU=false, const cv::Size & targetInputShape=cv::Size(224, 224)) <br>_Constructor._  |
|  [**ClassificationResult**](structyolos_1_1cls_1_1ClassificationResult.md) | [**classify**](classyolos_1_1cls_1_1YOLOClassifier.md#function-classify) (const cv::Mat & image) <br>_Run classification on an image._  |
|  void | [**drawResult**](classyolos_1_1cls_1_1YOLOClassifier.md#function-drawresult) (cv::Mat & image, const [**ClassificationResult**](structyolos_1_1cls_1_1ClassificationResult.md) & result, const cv::Point & position=cv::Point(10, 30)) const<br>_Draw classification result on an image._  |
|  const std::vector&lt; std::string &gt; & | [**getClassNames**](classyolos_1_1cls_1_1YOLOClassifier.md#function-getclassnames) () const<br>_Get class names._  |
|  cv::Size | [**getInputShape**](classyolos_1_1cls_1_1YOLOClassifier.md#function-getinputshape) () const<br>_Get input shape._  |
|  bool | [**isDynamicInputShape**](classyolos_1_1cls_1_1YOLOClassifier.md#function-isdynamicinputshape) () const<br>_Check if input shape is dynamic._  |
| virtual  | [**~YOLOClassifier**](classyolos_1_1cls_1_1YOLOClassifier.md#function-yoloclassifier) () = default<br> |
















## Protected Attributes inherited from yolos::cls::YOLOClassifier

See [yolos::cls::YOLOClassifier](classyolos_1_1cls_1_1YOLOClassifier.md)

| Type | Name |
| ---: | :--- |
|  std::vector&lt; std::string &gt; | [**classNames\_**](classyolos_1_1cls_1_1YOLOClassifier.md#variable-classnames_)  <br> |
|  Ort::Env | [**env\_**](classyolos_1_1cls_1_1YOLOClassifier.md#variable-env_)   = `{nullptr}`<br> |
|  std::vector&lt; float &gt; | [**inputBuffer\_**](classyolos_1_1cls_1_1YOLOClassifier.md#variable-inputbuffer_)  <br> |
|  cv::Size | [**inputImageShape\_**](classyolos_1_1cls_1_1YOLOClassifier.md#variable-inputimageshape_)  <br> |
|  std::vector&lt; Ort::AllocatedStringPtr &gt; | [**inputNameAllocs\_**](classyolos_1_1cls_1_1YOLOClassifier.md#variable-inputnameallocs_)  <br> |
|  std::vector&lt; const char \* &gt; | [**inputNames\_**](classyolos_1_1cls_1_1YOLOClassifier.md#variable-inputnames_)  <br> |
|  bool | [**isDynamicInputShape\_**](classyolos_1_1cls_1_1YOLOClassifier.md#variable-isdynamicinputshape_)   = `{false}`<br> |
|  int | [**numClasses\_**](classyolos_1_1cls_1_1YOLOClassifier.md#variable-numclasses_)   = `{0}`<br> |
|  size\_t | [**numInputNodes\_**](classyolos_1_1cls_1_1YOLOClassifier.md#variable-numinputnodes_)   = `{0}`<br> |
|  size\_t | [**numOutputNodes\_**](classyolos_1_1cls_1_1YOLOClassifier.md#variable-numoutputnodes_)   = `{0}`<br> |
|  std::vector&lt; Ort::AllocatedStringPtr &gt; | [**outputNameAllocs\_**](classyolos_1_1cls_1_1YOLOClassifier.md#variable-outputnameallocs_)  <br> |
|  std::vector&lt; const char \* &gt; | [**outputNames\_**](classyolos_1_1cls_1_1YOLOClassifier.md#variable-outputnames_)  <br> |
|  Ort::SessionOptions | [**sessionOptions\_**](classyolos_1_1cls_1_1YOLOClassifier.md#variable-sessionoptions_)   = `{nullptr}`<br> |
|  Ort::Session | [**session\_**](classyolos_1_1cls_1_1YOLOClassifier.md#variable-session_)   = `{nullptr}`<br> |
































## Protected Functions inherited from yolos::cls::YOLOClassifier

See [yolos::cls::YOLOClassifier](classyolos_1_1cls_1_1YOLOClassifier.md)

| Type | Name |
| ---: | :--- |
|  void | [**initSession**](classyolos_1_1cls_1_1YOLOClassifier.md#function-initsession) (const std::string & modelPath, bool useGPU) <br> |
|  [**ClassificationResult**](structyolos_1_1cls_1_1ClassificationResult.md) | [**postprocess**](classyolos_1_1cls_1_1YOLOClassifier.md#function-postprocess) (const std::vector&lt; Ort::Value &gt; & outputTensors) <br>_Postprocess classification output._  |
|  void | [**preprocess**](classyolos_1_1cls_1_1YOLOClassifier.md#function-preprocess) (const cv::Mat & image, std::vector&lt; int64\_t &gt; & inputTensorShape) <br>_Preprocess image for classification (Ultralytics-style)_  |






## Public Functions Documentation




### function YOLO26Classifier 

```C++
inline yolos::cls::YOLO26Classifier::YOLO26Classifier (
    const std::string & modelPath,
    const std::string & labelsPath,
    bool useGPU=false
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/tasks/classification.hpp`

