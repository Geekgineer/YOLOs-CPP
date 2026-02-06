

# Class yolos::seg::YOLOSegDetector



[**ClassList**](annotated.md) **>** [**yolos**](namespaceyolos.md) **>** [**seg**](namespaceyolos_1_1seg.md) **>** [**YOLOSegDetector**](classyolos_1_1seg_1_1YOLOSegDetector.md)



_YOLO segmentation detector with mask prediction._ 

* `#include <segmentation.hpp>`



Inherits the following classes: [yolos::OrtSessionBase](classyolos_1_1OrtSessionBase.md)






















































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**YOLOSegDetector**](#function-yolosegdetector) (const std::string & modelPath, const std::string & labelsPath, bool useGPU=false) <br>_Constructor._  |
|  void | [**drawMasksOnly**](#function-drawmasksonly) (cv::Mat & image, const std::vector&lt; [**Segmentation**](structyolos_1_1seg_1_1Segmentation.md) &gt; & results, float maskAlpha=0.5f) const<br>_Draw only segmentation masks (no boxes)_  |
|  void | [**drawSegmentations**](#function-drawsegmentations) (cv::Mat & image, const std::vector&lt; [**Segmentation**](structyolos_1_1seg_1_1Segmentation.md) &gt; & results, float maskAlpha=0.5f) const<br>_Draw segmentations with boxes and labels on an image._  |
|  const std::vector&lt; cv::Scalar &gt; & | [**getClassColors**](#function-getclasscolors) () const<br>_Get class colors._  |
|  const std::vector&lt; std::string &gt; & | [**getClassNames**](#function-getclassnames) () const<br>_Get class names._  |
|  std::vector&lt; [**Segmentation**](structyolos_1_1seg_1_1Segmentation.md) &gt; | [**segment**](#function-segment) (const cv::Mat & image, float confThreshold=0.4f, float iouThreshold=0.45f) <br>_Run segmentation on an image (optimized with buffer reuse)_  |
| virtual  | [**~YOLOSegDetector**](#function-yolosegdetector) () = default<br> |


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


## Protected Static Attributes

| Type | Name |
| ---: | :--- |
|  constexpr float | [**MASK\_THRESHOLD**](#variable-mask_threshold)   = `0.5f`<br> |




























## Protected Functions

| Type | Name |
| ---: | :--- |
|  std::vector&lt; [**Segmentation**](structyolos_1_1seg_1_1Segmentation.md) &gt; | [**postprocess**](#function-postprocess) (const cv::Size & originalSize, const cv::Size & letterboxSize, const std::vector&lt; Ort::Value &gt; & outputTensors, float confThreshold, float iouThreshold) <br>_Postprocess segmentation outputs._  |
|  std::vector&lt; [**Segmentation**](structyolos_1_1seg_1_1Segmentation.md) &gt; | [**postprocessV26**](#function-postprocessv26) (const cv::Size & originalSize, const cv::Size & letterboxSize, const float \* output0, const float \* output1, const std::vector&lt; int64\_t &gt; & shape0, const std::vector&lt; int64\_t &gt; & shape1, float confThreshold) <br>_Postprocess YOLO26-seg format outputs (end-to-end, no NMS needed) Output0 shape: [1, num\_detections, 38] where 38 = 4 (x1,y1,x2,y2) + 1 (conf) + 1 (class\_id) + 32 (mask\_coeffs)_  |


## Protected Functions inherited from yolos::OrtSessionBase

See [yolos::OrtSessionBase](classyolos_1_1OrtSessionBase.md)

| Type | Name |
| ---: | :--- |
|  Ort::Value | [**createInputTensor**](classyolos_1_1OrtSessionBase.md#function-createinputtensor) (float \* blob, const std::vector&lt; int64\_t &gt; & inputTensorShape) <br>_Create an input tensor from a blob._  |
|  std::vector&lt; Ort::Value &gt; | [**runInference**](classyolos_1_1OrtSessionBase.md#function-runinference) (Ort::Value & inputTensor) <br>_Run inference with the given input tensor._  |






## Public Functions Documentation




### function YOLOSegDetector 

_Constructor._ 
```C++
inline yolos::seg::YOLOSegDetector::YOLOSegDetector (
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



### function drawMasksOnly 

_Draw only segmentation masks (no boxes)_ 
```C++
inline void yolos::seg::YOLOSegDetector::drawMasksOnly (
    cv::Mat & image,
    const std::vector< Segmentation > & results,
    float maskAlpha=0.5f
) const
```




<hr>



### function drawSegmentations 

_Draw segmentations with boxes and labels on an image._ 
```C++
inline void yolos::seg::YOLOSegDetector::drawSegmentations (
    cv::Mat & image,
    const std::vector< Segmentation > & results,
    float maskAlpha=0.5f
) const
```





**Parameters:**


* `image` Image to draw on 
* `results` Vector of segmentation results 
* `maskAlpha` Mask transparency (0-1) 




        

<hr>



### function getClassColors 

_Get class colors._ 
```C++
inline const std::vector< cv::Scalar > & yolos::seg::YOLOSegDetector::getClassColors () const
```




<hr>



### function getClassNames 

_Get class names._ 
```C++
inline const std::vector< std::string > & yolos::seg::YOLOSegDetector::getClassNames () const
```




<hr>



### function segment 

_Run segmentation on an image (optimized with buffer reuse)_ 
```C++
inline std::vector< Segmentation > yolos::seg::YOLOSegDetector::segment (
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

Vector of segmentation results 





        

<hr>



### function ~YOLOSegDetector 

```C++
virtual yolos::seg::YOLOSegDetector::~YOLOSegDetector () = default
```




<hr>
## Protected Attributes Documentation




### variable buffer\_ 

```C++
preprocessing::InferenceBuffer yolos::seg::YOLOSegDetector::buffer_;
```




<hr>



### variable classColors\_ 

```C++
std::vector<cv::Scalar> yolos::seg::YOLOSegDetector::classColors_;
```




<hr>



### variable classNames\_ 

```C++
std::vector<std::string> yolos::seg::YOLOSegDetector::classNames_;
```




<hr>
## Protected Static Attributes Documentation




### variable MASK\_THRESHOLD 

```C++
constexpr float yolos::seg::YOLOSegDetector::MASK_THRESHOLD;
```




<hr>
## Protected Functions Documentation




### function postprocess 

_Postprocess segmentation outputs._ 
```C++
inline std::vector< Segmentation > yolos::seg::YOLOSegDetector::postprocess (
    const cv::Size & originalSize,
    const cv::Size & letterboxSize,
    const std::vector< Ort::Value > & outputTensors,
    float confThreshold,
    float iouThreshold
) 
```




<hr>



### function postprocessV26 

_Postprocess YOLO26-seg format outputs (end-to-end, no NMS needed) Output0 shape: [1, num\_detections, 38] where 38 = 4 (x1,y1,x2,y2) + 1 (conf) + 1 (class\_id) + 32 (mask\_coeffs)_ 
```C++
inline std::vector< Segmentation > yolos::seg::YOLOSegDetector::postprocessV26 (
    const cv::Size & originalSize,
    const cv::Size & letterboxSize,
    const float * output0,
    const float * output1,
    const std::vector< int64_t > & shape0,
    const std::vector< int64_t > & shape1,
    float confThreshold
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/tasks/segmentation.hpp`

