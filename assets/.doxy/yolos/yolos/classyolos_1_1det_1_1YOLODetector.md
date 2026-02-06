

# Class yolos::det::YOLODetector



[**ClassList**](annotated.md) **>** [**yolos**](namespaceyolos.md) **>** [**det**](namespaceyolos_1_1det.md) **>** [**YOLODetector**](classyolos_1_1det_1_1YOLODetector.md)



_Base YOLO detector with runtime version auto-detection._ 

* `#include <detection.hpp>`



Inherits the following classes: [yolos::OrtSessionBase](classyolos_1_1OrtSessionBase.md)


Inherited by the following classes: [yolos::det::YOLO26Detector](classyolos_1_1det_1_1YOLO26Detector.md),  [yolos::det::YOLONASDetector](classyolos_1_1det_1_1YOLONASDetector.md),  [yolos::det::YOLOv10Detector](classyolos_1_1det_1_1YOLOv10Detector.md),  [yolos::det::YOLOv11Detector](classyolos_1_1det_1_1YOLOv11Detector.md),  [yolos::det::YOLOv7Detector](classyolos_1_1det_1_1YOLOv7Detector.md),  [yolos::det::YOLOv8Detector](classyolos_1_1det_1_1YOLOv8Detector.md)




















































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**YOLODetector**](#function-yolodetector) (const std::string & modelPath, const std::string & labelsPath, bool useGPU=false, [**YOLOVersion**](namespaceyolos.md#enum-yoloversion) version=YOLOVersion::Auto) <br>_Constructor._  |
| virtual std::vector&lt; [**Detection**](structyolos_1_1det_1_1Detection.md) &gt; | [**detect**](#function-detect) (const cv::Mat & image, float confThreshold=0.4f, float iouThreshold=0.45f) <br>_Run detection on an image (optimized with buffer reuse)_  |
|  void | [**drawDetections**](#function-drawdetections) (cv::Mat & image, const std::vector&lt; [**Detection**](structyolos_1_1det_1_1Detection.md) &gt; & detections) const<br>_Draw detections on an image._  |
|  void | [**drawDetectionsWithMask**](#function-drawdetectionswithmask) (cv::Mat & image, const std::vector&lt; [**Detection**](structyolos_1_1det_1_1Detection.md) &gt; & detections, float alpha=0.4f) const<br>_Draw detections with semi-transparent mask fill._  |
|  const std::vector&lt; cv::Scalar &gt; & | [**getClassColors**](#function-getclasscolors) () const<br>_Get class colors._  |
|  const std::vector&lt; std::string &gt; & | [**getClassNames**](#function-getclassnames) () const<br>_Get class names._  |
| virtual  | [**~YOLODetector**](#function-yolodetector) () = default<br> |


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
|  [**YOLOVersion**](namespaceyolos.md#enum-yoloversion) | [**version\_**](#variable-version_)   = `{YOLOVersion::Auto}`<br> |


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
|  [**YOLOVersion**](namespaceyolos.md#enum-yoloversion) | [**detectVersion**](#function-detectversion) (const std::vector&lt; Ort::Value &gt; & outputTensors) <br>_Detect YOLO version from output tensors._  |
| virtual std::vector&lt; [**Detection**](structyolos_1_1det_1_1Detection.md) &gt; | [**postprocess**](#function-postprocess) (const cv::Size & originalSize, const cv::Size & resizedShape, const std::vector&lt; Ort::Value &gt; & outputTensors, [**YOLOVersion**](namespaceyolos.md#enum-yoloversion) version, float confThreshold, float iouThreshold) <br>_Postprocess based on detected version._  |
| virtual std::vector&lt; [**Detection**](structyolos_1_1det_1_1Detection.md) &gt; | [**postprocessNAS**](#function-postprocessnas) (const cv::Size & originalSize, const cv::Size & resizedShape, const std::vector&lt; Ort::Value &gt; & outputTensors, float confThreshold, float iouThreshold) <br>_Postprocess for YOLO-NAS format (two outputs: boxes and scores)_  |
| virtual std::vector&lt; [**Detection**](structyolos_1_1det_1_1Detection.md) &gt; | [**postprocessStandard**](#function-postprocessstandard) (const cv::Size & originalSize, const cv::Size & resizedShape, const std::vector&lt; Ort::Value &gt; & outputTensors, float confThreshold, float iouThreshold) <br>_Standard postprocess for YOLOv8/v11 format [batch, features, boxes] Optimized: single box storage with batched NMS._  |
| virtual std::vector&lt; [**Detection**](structyolos_1_1det_1_1Detection.md) &gt; | [**postprocessV10**](#function-postprocessv10) (const cv::Size & originalSize, const cv::Size & resizedShape, const std::vector&lt; Ort::Value &gt; & outputTensors, float confThreshold, float) <br>_Postprocess for YOLOv10 format [batch, boxes, 6] (end-to-end, no NMS needed)_  |
| virtual std::vector&lt; [**Detection**](structyolos_1_1det_1_1Detection.md) &gt; | [**postprocessV7**](#function-postprocessv7) (const cv::Size & originalSize, const cv::Size & resizedShape, const std::vector&lt; Ort::Value &gt; & outputTensors, float confThreshold, float iouThreshold) <br>_Postprocess for YOLOv7 format [batch, boxes, features]._  |


## Protected Functions inherited from yolos::OrtSessionBase

See [yolos::OrtSessionBase](classyolos_1_1OrtSessionBase.md)

| Type | Name |
| ---: | :--- |
|  Ort::Value | [**createInputTensor**](classyolos_1_1OrtSessionBase.md#function-createinputtensor) (float \* blob, const std::vector&lt; int64\_t &gt; & inputTensorShape) <br>_Create an input tensor from a blob._  |
|  std::vector&lt; Ort::Value &gt; | [**runInference**](classyolos_1_1OrtSessionBase.md#function-runinference) (Ort::Value & inputTensor) <br>_Run inference with the given input tensor._  |






## Public Functions Documentation




### function YOLODetector 

_Constructor._ 
```C++
inline yolos::det::YOLODetector::YOLODetector (
    const std::string & modelPath,
    const std::string & labelsPath,
    bool useGPU=false,
    YOLOVersion version=YOLOVersion::Auto
) 
```





**Parameters:**


* `modelPath` Path to the ONNX model file 
* `labelsPath` Path to the class names file 
* `useGPU` Whether to use GPU for inference 
* `version` YOLO version (Auto for runtime detection) 




        

<hr>



### function detect 

_Run detection on an image (optimized with buffer reuse)_ 
```C++
inline virtual std::vector< Detection > yolos::det::YOLODetector::detect (
    const cv::Mat & image,
    float confThreshold=0.4f,
    float iouThreshold=0.45f
) 
```





**Parameters:**


* `image` Input image (BGR format) 
* `confThreshold` Confidence threshold 
* `iouThreshold` IoU threshold for NMS 



**Returns:**

Vector of detections 





        

<hr>



### function drawDetections 

_Draw detections on an image._ 
```C++
inline void yolos::det::YOLODetector::drawDetections (
    cv::Mat & image,
    const std::vector< Detection > & detections
) const
```





**Parameters:**


* `image` Image to draw on 
* `detections` Vector of detections 




        

<hr>



### function drawDetectionsWithMask 

_Draw detections with semi-transparent mask fill._ 
```C++
inline void yolos::det::YOLODetector::drawDetectionsWithMask (
    cv::Mat & image,
    const std::vector< Detection > & detections,
    float alpha=0.4f
) const
```




<hr>



### function getClassColors 

_Get class colors._ 
```C++
inline const std::vector< cv::Scalar > & yolos::det::YOLODetector::getClassColors () const
```




<hr>



### function getClassNames 

_Get class names._ 
```C++
inline const std::vector< std::string > & yolos::det::YOLODetector::getClassNames () const
```




<hr>



### function ~YOLODetector 

```C++
virtual yolos::det::YOLODetector::~YOLODetector () = default
```




<hr>
## Protected Attributes Documentation




### variable buffer\_ 

```C++
preprocessing::InferenceBuffer yolos::det::YOLODetector::buffer_;
```




<hr>



### variable classColors\_ 

```C++
std::vector<cv::Scalar> yolos::det::YOLODetector::classColors_;
```




<hr>



### variable classNames\_ 

```C++
std::vector<std::string> yolos::det::YOLODetector::classNames_;
```




<hr>



### variable version\_ 

```C++
YOLOVersion yolos::det::YOLODetector::version_;
```




<hr>
## Protected Functions Documentation




### function detectVersion 

_Detect YOLO version from output tensors._ 
```C++
inline YOLOVersion yolos::det::YOLODetector::detectVersion (
    const std::vector< Ort::Value > & outputTensors
) 
```




<hr>



### function postprocess 

_Postprocess based on detected version._ 
```C++
inline virtual std::vector< Detection > yolos::det::YOLODetector::postprocess (
    const cv::Size & originalSize,
    const cv::Size & resizedShape,
    const std::vector< Ort::Value > & outputTensors,
    YOLOVersion version,
    float confThreshold,
    float iouThreshold
) 
```




<hr>



### function postprocessNAS 

_Postprocess for YOLO-NAS format (two outputs: boxes and scores)_ 
```C++
inline virtual std::vector< Detection > yolos::det::YOLODetector::postprocessNAS (
    const cv::Size & originalSize,
    const cv::Size & resizedShape,
    const std::vector< Ort::Value > & outputTensors,
    float confThreshold,
    float iouThreshold
) 
```




<hr>



### function postprocessStandard 

_Standard postprocess for YOLOv8/v11 format [batch, features, boxes] Optimized: single box storage with batched NMS._ 
```C++
inline virtual std::vector< Detection > yolos::det::YOLODetector::postprocessStandard (
    const cv::Size & originalSize,
    const cv::Size & resizedShape,
    const std::vector< Ort::Value > & outputTensors,
    float confThreshold,
    float iouThreshold
) 
```




<hr>



### function postprocessV10 

_Postprocess for YOLOv10 format [batch, boxes, 6] (end-to-end, no NMS needed)_ 
```C++
inline virtual std::vector< Detection > yolos::det::YOLODetector::postprocessV10 (
    const cv::Size & originalSize,
    const cv::Size & resizedShape,
    const std::vector< Ort::Value > & outputTensors,
    float confThreshold,
    float
) 
```




<hr>



### function postprocessV7 

_Postprocess for YOLOv7 format [batch, boxes, features]._ 
```C++
inline virtual std::vector< Detection > yolos::det::YOLODetector::postprocessV7 (
    const cv::Size & originalSize,
    const cv::Size & resizedShape,
    const std::vector< Ort::Value > & outputTensors,
    float confThreshold,
    float iouThreshold
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/tasks/detection.hpp`

