

# Class yolos::obb::YOLOOBBDetector



[**ClassList**](annotated.md) **>** [**yolos**](namespaceyolos.md) **>** [**obb**](namespaceyolos_1_1obb.md) **>** [**YOLOOBBDetector**](classyolos_1_1obb_1_1YOLOOBBDetector.md)



_YOLO oriented bounding box detector for rotated object detection._ 

* `#include <obb.hpp>`



Inherits the following classes: [yolos::OrtSessionBase](classyolos_1_1OrtSessionBase.md)






















































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**YOLOOBBDetector**](#function-yoloobbdetector) (const std::string & modelPath, const std::string & labelsPath, bool useGPU=false) <br>_Constructor._  |
|  std::vector&lt; [**OBBResult**](structyolos_1_1obb_1_1OBBResult.md) &gt; | [**detect**](#function-detect) (const cv::Mat & image, float confThreshold=0.25f, float iouThreshold=0.45f, int maxDet=300) <br>_Run OBB detection on an image (optimized with buffer reuse)_  |
|  void | [**drawDetections**](#function-drawdetections) (cv::Mat & image, const std::vector&lt; [**OBBResult**](structyolos_1_1obb_1_1OBBResult.md) &gt; & results, int thickness=2) const<br>_Draw OBB detections on an image._  |
|  const std::vector&lt; cv::Scalar &gt; & | [**getClassColors**](#function-getclasscolors) () const<br>_Get class colors._  |
|  const std::vector&lt; std::string &gt; & | [**getClassNames**](#function-getclassnames) () const<br>_Get class names._  |
| virtual  | [**~YOLOOBBDetector**](#function-yoloobbdetector) () = default<br> |


## Public Functions inherited from yolos::OrtSessionBase

See [yolos::OrtSessionBase](classyolos_1_1OrtSessionBase.md)

| Type | Name |
| ---: | :--- |
|   | [**OrtSessionBase**](classyolos_1_1OrtSessionBase.md#function-ortsessionbase-13) (const std::string & modelPath, bool useGPU=false, int numThreads=0) <br>_Constructor - loads and initializes the ONNX model._  |
|   | [**OrtSessionBase**](classyolos_1_1OrtSessionBase.md#function-ortsessionbase-23) (const [**OrtSessionBase**](classyolos_1_1OrtSessionBase.md) &) = delete<br> |
|   | [**OrtSessionBase**](classyolos_1_1OrtSessionBase.md#function-ortsessionbase-33) ([**OrtSessionBase**](classyolos_1_1OrtSessionBase.md) &&) = default<br> |
|  const std::string & | [**getDevice**](classyolos_1_1OrtSessionBase.md#function-getdevice) () noexcept const<br>_Get the device being used for inference._  |
|  cv::Size | [**getInputShape**](classyolos_1_1OrtSessionBase.md#function-getinputshape) () noexcept const<br>_Get the input image shape expected by the model._  |
|  size\_t | [**getNumInputNodes**](classyolos_1_1OrtSessionBase.md#function-getnuminputnodes) () noexcept const<br>_Get the number of input nodes._  |
|  size\_t | [**getNumOutputNodes**](classyolos_1_1OrtSessionBase.md#function-getnumoutputnodes) () noexcept const<br>_Get the number of output nodes._  |
|  bool | [**isDynamicBatchSize**](classyolos_1_1OrtSessionBase.md#function-isdynamicbatchsize) () noexcept const<br>_Check if batch size is dynamic._  |
|  bool | [**isDynamicInputShape**](classyolos_1_1OrtSessionBase.md#function-isdynamicinputshape) () noexcept const<br>_Check if input shape is dynamic._  |
|  [**OrtSessionBase**](classyolos_1_1OrtSessionBase.md) & | [**operator=**](classyolos_1_1OrtSessionBase.md#function-operator) (const [**OrtSessionBase**](classyolos_1_1OrtSessionBase.md) &) = delete<br> |
|  [**OrtSessionBase**](classyolos_1_1OrtSessionBase.md) & | [**operator=**](classyolos_1_1OrtSessionBase.md#function-operator_1) ([**OrtSessionBase**](classyolos_1_1OrtSessionBase.md) &&) = default<br> |
| virtual  | [**~OrtSessionBase**](classyolos_1_1OrtSessionBase.md#function-ortsessionbase) () = default<br> |














## Protected Attributes

| Type | Name |
| ---: | :--- |
|  [**preprocessing::InferenceBuffer**](structyolos_1_1preprocessing_1_1InferenceBuffer.md) | [**buffer\_**](#variable-buffer_)  <br> |
|  std::vector&lt; cv::Scalar &gt; | [**classColors\_**](#variable-classcolors_)  <br> |
|  std::vector&lt; std::string &gt; | [**classNames\_**](#variable-classnames_)  <br> |


## Protected Attributes inherited from yolos::OrtSessionBase

See [yolos::OrtSessionBase](classyolos_1_1OrtSessionBase.md)

| Type | Name |
| ---: | :--- |
|  std::string | [**device\_**](classyolos_1_1OrtSessionBase.md#variable-device_)   = `{"cpu"}`<br> |
|  Ort::Env | [**env\_**](classyolos_1_1OrtSessionBase.md#variable-env_)   = `{nullptr}`<br> |
|  std::vector&lt; Ort::AllocatedStringPtr &gt; | [**inputNameAllocs\_**](classyolos_1_1OrtSessionBase.md#variable-inputnameallocs_)  <br> |
|  std::vector&lt; const char \* &gt; | [**inputNames\_**](classyolos_1_1OrtSessionBase.md#variable-inputnames_)  <br> |
|  cv::Size | [**inputShape\_**](classyolos_1_1OrtSessionBase.md#variable-inputshape_)  <br> |
|  bool | [**isDynamicBatchSize\_**](classyolos_1_1OrtSessionBase.md#variable-isdynamicbatchsize_)   = `{false}`<br> |
|  bool | [**isDynamicInputShape\_**](classyolos_1_1OrtSessionBase.md#variable-isdynamicinputshape_)   = `{false}`<br> |
|  size\_t | [**numInputNodes\_**](classyolos_1_1OrtSessionBase.md#variable-numinputnodes_)   = `{0}`<br> |
|  size\_t | [**numOutputNodes\_**](classyolos_1_1OrtSessionBase.md#variable-numoutputnodes_)   = `{0}`<br> |
|  std::vector&lt; Ort::AllocatedStringPtr &gt; | [**outputNameAllocs\_**](classyolos_1_1OrtSessionBase.md#variable-outputnameallocs_)  <br> |
|  std::vector&lt; const char \* &gt; | [**outputNames\_**](classyolos_1_1OrtSessionBase.md#variable-outputnames_)  <br> |
|  Ort::SessionOptions | [**sessionOptions\_**](classyolos_1_1OrtSessionBase.md#variable-sessionoptions_)   = `{nullptr}`<br> |
|  Ort::Session | [**session\_**](classyolos_1_1OrtSessionBase.md#variable-session_)   = `{nullptr}`<br> |






























## Protected Functions

| Type | Name |
| ---: | :--- |
|  std::vector&lt; [**OBBResult**](structyolos_1_1obb_1_1OBBResult.md) &gt; | [**postprocess**](#function-postprocess) (const cv::Size & originalSize, const cv::Size & resizedShape, const std::vector&lt; Ort::Value &gt; & outputTensors, float confThreshold, float iouThreshold, int maxDet) <br>_Postprocess OBB detection outputs._  |
|  std::vector&lt; [**OBBResult**](structyolos_1_1obb_1_1OBBResult.md) &gt; | [**postprocessV26**](#function-postprocessv26) (const cv::Size & originalSize, const cv::Size & resizedShape, const float \* rawOutput, const std::vector&lt; int64\_t &gt; & outputShape, float confThreshold, int maxDet) <br>_Postprocess YOLO26 OBB detection outputs (end-to-end, NMS-free)_  |
|  std::vector&lt; [**OBBResult**](structyolos_1_1obb_1_1OBBResult.md) &gt; | [**postprocessV8**](#function-postprocessv8) (const cv::Size & originalSize, const cv::Size & resizedShape, const float \* rawOutput, const std::vector&lt; int64\_t &gt; & outputShape, float confThreshold, float iouThreshold, int maxDet) <br>_Postprocess YOLOv8/v11 OBB detection outputs (requires NMS)_  |


## Protected Functions inherited from yolos::OrtSessionBase

See [yolos::OrtSessionBase](classyolos_1_1OrtSessionBase.md)

| Type | Name |
| ---: | :--- |
|  Ort::Value | [**createInputTensor**](classyolos_1_1OrtSessionBase.md#function-createinputtensor) (float \* blob, const std::vector&lt; int64\_t &gt; & inputTensorShape) <br>_Create an input tensor from a blob._  |
|  std::vector&lt; Ort::Value &gt; | [**runInference**](classyolos_1_1OrtSessionBase.md#function-runinference) (Ort::Value & inputTensor) <br>_Run inference with the given input tensor._  |






## Public Functions Documentation




### function YOLOOBBDetector 

_Constructor._ 
```C++
inline yolos::obb::YOLOOBBDetector::YOLOOBBDetector (
    const std::string & modelPath,
    const std::string & labelsPath,
    bool useGPU=false
) 
```





**Parameters:**


* `modelPath` Path to the ONNX model file 
* `labelsPath` Path to the class names file 
* `useGPU` Whether to use GPU for inference 




        

<hr>



### function detect 

_Run OBB detection on an image (optimized with buffer reuse)_ 
```C++
inline std::vector< OBBResult > yolos::obb::YOLOOBBDetector::detect (
    const cv::Mat & image,
    float confThreshold=0.25f,
    float iouThreshold=0.45f,
    int maxDet=300
) 
```





**Parameters:**


* `image` Input image (BGR format) 
* `confThreshold` Confidence threshold 
* `iouThreshold` IoU threshold for NMS 
* `maxDet` Maximum number of detections to return 



**Returns:**

Vector of OBB detection results 





        

<hr>



### function drawDetections 

_Draw OBB detections on an image._ 
```C++
inline void yolos::obb::YOLOOBBDetector::drawDetections (
    cv::Mat & image,
    const std::vector< OBBResult > & results,
    int thickness=2
) const
```





**Parameters:**


* `image` Image to draw on 
* `results` Vector of OBB detection results 
* `thickness` Line thickness 




        

<hr>



### function getClassColors 

_Get class colors._ 
```C++
inline const std::vector< cv::Scalar > & yolos::obb::YOLOOBBDetector::getClassColors () const
```




<hr>



### function getClassNames 

_Get class names._ 
```C++
inline const std::vector< std::string > & yolos::obb::YOLOOBBDetector::getClassNames () const
```




<hr>



### function ~YOLOOBBDetector 

```C++
virtual yolos::obb::YOLOOBBDetector::~YOLOOBBDetector () = default
```




<hr>
## Protected Attributes Documentation




### variable buffer\_ 

```C++
preprocessing::InferenceBuffer yolos::obb::YOLOOBBDetector::buffer_;
```




<hr>



### variable classColors\_ 

```C++
std::vector<cv::Scalar> yolos::obb::YOLOOBBDetector::classColors_;
```




<hr>



### variable classNames\_ 

```C++
std::vector<std::string> yolos::obb::YOLOOBBDetector::classNames_;
```




<hr>
## Protected Functions Documentation




### function postprocess 

_Postprocess OBB detection outputs._ 
```C++
inline std::vector< OBBResult > yolos::obb::YOLOOBBDetector::postprocess (
    const cv::Size & originalSize,
    const cv::Size & resizedShape,
    const std::vector< Ort::Value > & outputTensors,
    float confThreshold,
    float iouThreshold,
    int maxDet
) 
```




<hr>



### function postprocessV26 

_Postprocess YOLO26 OBB detection outputs (end-to-end, NMS-free)_ 
```C++
inline std::vector< OBBResult > yolos::obb::YOLOOBBDetector::postprocessV26 (
    const cv::Size & originalSize,
    const cv::Size & resizedShape,
    const float * rawOutput,
    const std::vector< int64_t > & outputShape,
    float confThreshold,
    int maxDet
) 
```




<hr>



### function postprocessV8 

_Postprocess YOLOv8/v11 OBB detection outputs (requires NMS)_ 
```C++
inline std::vector< OBBResult > yolos::obb::YOLOOBBDetector::postprocessV8 (
    const cv::Size & originalSize,
    const cv::Size & resizedShape,
    const float * rawOutput,
    const std::vector< int64_t > & outputShape,
    float confThreshold,
    float iouThreshold,
    int maxDet
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/tasks/obb.hpp`

