

# Class yolos::cls::YOLOClassifier



[**ClassList**](annotated.md) **>** [**yolos**](namespaceyolos.md) **>** [**cls**](namespaceyolos_1_1cls.md) **>** [**YOLOClassifier**](classyolos_1_1cls_1_1YOLOClassifier.md)



_YOLO classifier for image classification._ 

* `#include <classification.hpp>`





Inherited by the following classes: [yolos::cls::YOLO11Classifier](classyolos_1_1cls_1_1YOLO11Classifier.md),  [yolos::cls::YOLO12Classifier](classyolos_1_1cls_1_1YOLO12Classifier.md),  [yolos::cls::YOLO26Classifier](classyolos_1_1cls_1_1YOLO26Classifier.md)
































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**YOLOClassifier**](#function-yoloclassifier) (const std::string & modelPath, const std::string & labelsPath, bool useGPU=false, const cv::Size & targetInputShape=cv::Size(224, 224)) <br>_Constructor._  |
|  [**ClassificationResult**](structyolos_1_1cls_1_1ClassificationResult.md) | [**classify**](#function-classify) (const cv::Mat & image) <br>_Run classification on an image._  |
|  void | [**drawResult**](#function-drawresult) (cv::Mat & image, const [**ClassificationResult**](structyolos_1_1cls_1_1ClassificationResult.md) & result, const cv::Point & position=cv::Point(10, 30)) const<br>_Draw classification result on an image._  |
|  const std::vector&lt; std::string &gt; & | [**getClassNames**](#function-getclassnames) () const<br>_Get class names._  |
|  cv::Size | [**getInputShape**](#function-getinputshape) () const<br>_Get input shape._  |
|  bool | [**isDynamicInputShape**](#function-isdynamicinputshape) () const<br>_Check if input shape is dynamic._  |
| virtual  | [**~YOLOClassifier**](#function-yoloclassifier) () = default<br> |








## Protected Attributes

| Type | Name |
| ---: | :--- |
|  std::vector&lt; std::string &gt; | [**classNames\_**](#variable-classnames_)  <br> |
|  Ort::Env | [**env\_**](#variable-env_)   = `{nullptr}`<br> |
|  std::vector&lt; float &gt; | [**inputBuffer\_**](#variable-inputbuffer_)  <br> |
|  cv::Size | [**inputImageShape\_**](#variable-inputimageshape_)  <br> |
|  std::vector&lt; Ort::AllocatedStringPtr &gt; | [**inputNameAllocs\_**](#variable-inputnameallocs_)  <br> |
|  std::vector&lt; const char \* &gt; | [**inputNames\_**](#variable-inputnames_)  <br> |
|  bool | [**isDynamicInputShape\_**](#variable-isdynamicinputshape_)   = `{false}`<br> |
|  int | [**numClasses\_**](#variable-numclasses_)   = `{0}`<br> |
|  size\_t | [**numInputNodes\_**](#variable-numinputnodes_)   = `{0}`<br> |
|  size\_t | [**numOutputNodes\_**](#variable-numoutputnodes_)   = `{0}`<br> |
|  std::vector&lt; Ort::AllocatedStringPtr &gt; | [**outputNameAllocs\_**](#variable-outputnameallocs_)  <br> |
|  std::vector&lt; const char \* &gt; | [**outputNames\_**](#variable-outputnames_)  <br> |
|  Ort::SessionOptions | [**sessionOptions\_**](#variable-sessionoptions_)   = `{nullptr}`<br> |
|  Ort::Session | [**session\_**](#variable-session_)   = `{nullptr}`<br> |
















## Protected Functions

| Type | Name |
| ---: | :--- |
|  void | [**initSession**](#function-initsession) (const std::string & modelPath, bool useGPU) <br> |
|  [**ClassificationResult**](structyolos_1_1cls_1_1ClassificationResult.md) | [**postprocess**](#function-postprocess) (const std::vector&lt; Ort::Value &gt; & outputTensors) <br>_Postprocess classification output._  |
|  void | [**preprocess**](#function-preprocess) (const cv::Mat & image, std::vector&lt; int64\_t &gt; & inputTensorShape) <br>_Preprocess image for classification (Ultralytics-style)_  |




## Public Functions Documentation




### function YOLOClassifier 

_Constructor._ 
```C++
inline yolos::cls::YOLOClassifier::YOLOClassifier (
    const std::string & modelPath,
    const std::string & labelsPath,
    bool useGPU=false,
    const cv::Size & targetInputShape=cv::Size(224, 224)
) 
```





**Parameters:**


* `modelPath` Path to the ONNX model file 
* `labelsPath` Path to the class names file 
* `useGPU` Whether to use GPU for inference 
* `targetInputShape` Target input shape for preprocessing 




        

<hr>



### function classify 

_Run classification on an image._ 
```C++
inline ClassificationResult yolos::cls::YOLOClassifier::classify (
    const cv::Mat & image
) 
```





**Parameters:**


* `image` Input image (BGR format) 



**Returns:**

Classification result 





        

<hr>



### function drawResult 

_Draw classification result on an image._ 
```C++
inline void yolos::cls::YOLOClassifier::drawResult (
    cv::Mat & image,
    const ClassificationResult & result,
    const cv::Point & position=cv::Point(10, 30)
) const
```




<hr>



### function getClassNames 

_Get class names._ 
```C++
inline const std::vector< std::string > & yolos::cls::YOLOClassifier::getClassNames () const
```




<hr>



### function getInputShape 

_Get input shape._ 
```C++
inline cv::Size yolos::cls::YOLOClassifier::getInputShape () const
```




<hr>



### function isDynamicInputShape 

_Check if input shape is dynamic._ 
```C++
inline bool yolos::cls::YOLOClassifier::isDynamicInputShape () const
```




<hr>



### function ~YOLOClassifier 

```C++
virtual yolos::cls::YOLOClassifier::~YOLOClassifier () = default
```




<hr>
## Protected Attributes Documentation




### variable classNames\_ 

```C++
std::vector<std::string> yolos::cls::YOLOClassifier::classNames_;
```




<hr>



### variable env\_ 

```C++
Ort::Env yolos::cls::YOLOClassifier::env_;
```




<hr>



### variable inputBuffer\_ 

```C++
std::vector<float> yolos::cls::YOLOClassifier::inputBuffer_;
```




<hr>



### variable inputImageShape\_ 

```C++
cv::Size yolos::cls::YOLOClassifier::inputImageShape_;
```




<hr>



### variable inputNameAllocs\_ 

```C++
std::vector<Ort::AllocatedStringPtr> yolos::cls::YOLOClassifier::inputNameAllocs_;
```




<hr>



### variable inputNames\_ 

```C++
std::vector<const char*> yolos::cls::YOLOClassifier::inputNames_;
```




<hr>



### variable isDynamicInputShape\_ 

```C++
bool yolos::cls::YOLOClassifier::isDynamicInputShape_;
```




<hr>



### variable numClasses\_ 

```C++
int yolos::cls::YOLOClassifier::numClasses_;
```




<hr>



### variable numInputNodes\_ 

```C++
size_t yolos::cls::YOLOClassifier::numInputNodes_;
```




<hr>



### variable numOutputNodes\_ 

```C++
size_t yolos::cls::YOLOClassifier::numOutputNodes_;
```




<hr>



### variable outputNameAllocs\_ 

```C++
std::vector<Ort::AllocatedStringPtr> yolos::cls::YOLOClassifier::outputNameAllocs_;
```




<hr>



### variable outputNames\_ 

```C++
std::vector<const char*> yolos::cls::YOLOClassifier::outputNames_;
```




<hr>



### variable sessionOptions\_ 

```C++
Ort::SessionOptions yolos::cls::YOLOClassifier::sessionOptions_;
```




<hr>



### variable session\_ 

```C++
Ort::Session yolos::cls::YOLOClassifier::session_;
```




<hr>
## Protected Functions Documentation




### function initSession 

```C++
inline void yolos::cls::YOLOClassifier::initSession (
    const std::string & modelPath,
    bool useGPU
) 
```




<hr>



### function postprocess 

_Postprocess classification output._ 
```C++
inline ClassificationResult yolos::cls::YOLOClassifier::postprocess (
    const std::vector< Ort::Value > & outputTensors
) 
```




<hr>



### function preprocess 

_Preprocess image for classification (Ultralytics-style)_ 
```C++
inline void yolos::cls::YOLOClassifier::preprocess (
    const cv::Mat & image,
    std::vector< int64_t > & inputTensorShape
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/tasks/classification.hpp`

