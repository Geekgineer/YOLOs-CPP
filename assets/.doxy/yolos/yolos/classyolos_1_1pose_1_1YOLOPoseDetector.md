

# Class yolos::pose::YOLOPoseDetector



[**ClassList**](annotated.md) **>** [**yolos**](namespaceyolos.md) **>** [**pose**](namespaceyolos_1_1pose.md) **>** [**YOLOPoseDetector**](classyolos_1_1pose_1_1YOLOPoseDetector.md)



_YOLO pose estimation detector with keypoint detection._ 

* `#include <pose.hpp>`



Inherits the following classes: [yolos::OrtSessionBase](classyolos_1_1OrtSessionBase.md)






















































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**YOLOPoseDetector**](#function-yoloposedetector) (const std::string & modelPath, const std::string & labelsPath="", bool useGPU=false) <br>_Constructor._  |
|  std::vector&lt; [**PoseResult**](structyolos_1_1pose_1_1PoseResult.md) &gt; | [**detect**](#function-detect) (const cv::Mat & image, float confThreshold=0.4f, float iouThreshold=0.5f) <br>_Run pose detection on an image (optimized with buffer reuse)_  |
|  void | [**drawPoses**](#function-drawposes) (cv::Mat & image, const std::vector&lt; [**PoseResult**](structyolos_1_1pose_1_1PoseResult.md) &gt; & results, int kptRadius=4, float kptThreshold=0.5f, int lineThickness=2) const<br>_Draw pose estimations on an image._  |
|  void | [**drawSkeletonsOnly**](#function-drawskeletonsonly) (cv::Mat & image, const std::vector&lt; [**PoseResult**](structyolos_1_1pose_1_1PoseResult.md) &gt; & results, int kptRadius=4, float kptThreshold=0.5f, int lineThickness=2) const<br>_Draw only skeletons (no bounding boxes)_  |
|  const std::vector&lt; std::string &gt; & | [**getClassNames**](#function-getclassnames) () const<br>_Get class names._  |
| virtual  | [**~YOLOPoseDetector**](#function-yoloposedetector) () = default<br> |


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


## Public Static Functions

| Type | Name |
| ---: | :--- |
|  const std::vector&lt; std::pair&lt; int, int &gt; &gt; & | [**getPoseSkeleton**](#function-getposeskeleton) () <br>_Get COCO pose skeleton connections._  |












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
|  constexpr int | [**FEATURES\_PER\_KEYPOINT**](#variable-features_per_keypoint)   = `3`<br> |
|  constexpr int | [**NUM\_KEYPOINTS**](#variable-num_keypoints)   = `17`<br> |




























## Protected Functions

| Type | Name |
| ---: | :--- |
|  std::vector&lt; [**PoseResult**](structyolos_1_1pose_1_1PoseResult.md) &gt; | [**postprocess**](#function-postprocess) (const cv::Size & originalSize, const cv::Size & resizedShape, const std::vector&lt; Ort::Value &gt; & outputTensors, float confThreshold, float iouThreshold) <br>_Postprocess pose detection outputs._  |
|  std::vector&lt; [**PoseResult**](structyolos_1_1pose_1_1PoseResult.md) &gt; | [**postprocessV26**](#function-postprocessv26) (const cv::Size & originalSize, const cv::Size & resizedShape, const float \* rawOutput, const std::vector&lt; int64\_t &gt; & outputShape, float confThreshold) <br>_Postprocess YOLO26 pose detection outputs (end-to-end, NMS-free)_  |
|  std::vector&lt; [**PoseResult**](structyolos_1_1pose_1_1PoseResult.md) &gt; | [**postprocessV8**](#function-postprocessv8) (const cv::Size & originalSize, const cv::Size & resizedShape, const float \* rawOutput, const std::vector&lt; int64\_t &gt; & outputShape, float confThreshold, float iouThreshold) <br>_Postprocess YOLOv8/v11 pose detection outputs (requires NMS)_  |


## Protected Functions inherited from yolos::OrtSessionBase

See [yolos::OrtSessionBase](classyolos_1_1OrtSessionBase.md)

| Type | Name |
| ---: | :--- |
|  Ort::Value | [**createInputTensor**](classyolos_1_1OrtSessionBase.md#function-createinputtensor) (float \* blob, const std::vector&lt; int64\_t &gt; & inputTensorShape) <br>_Create an input tensor from a blob._  |
|  std::vector&lt; Ort::Value &gt; | [**runInference**](classyolos_1_1OrtSessionBase.md#function-runinference) (Ort::Value & inputTensor) <br>_Run inference with the given input tensor._  |






## Public Functions Documentation




### function YOLOPoseDetector 

_Constructor._ 
```C++
inline yolos::pose::YOLOPoseDetector::YOLOPoseDetector (
    const std::string & modelPath,
    const std::string & labelsPath="",
    bool useGPU=false
) 
```





**Parameters:**


* `modelPath` Path to the ONNX model file 
* `labelsPath` Path to the class names file (optional for pose) 
* `useGPU` Whether to use GPU for inference 




        

<hr>



### function detect 

_Run pose detection on an image (optimized with buffer reuse)_ 
```C++
inline std::vector< PoseResult > yolos::pose::YOLOPoseDetector::detect (
    const cv::Mat & image,
    float confThreshold=0.4f,
    float iouThreshold=0.5f
) 
```





**Parameters:**


* `image` Input image (BGR format) 
* `confThreshold` Confidence threshold 
* `iouThreshold` IoU threshold for NMS 



**Returns:**

Vector of pose results 





        

<hr>



### function drawPoses 

_Draw pose estimations on an image._ 
```C++
inline void yolos::pose::YOLOPoseDetector::drawPoses (
    cv::Mat & image,
    const std::vector< PoseResult > & results,
    int kptRadius=4,
    float kptThreshold=0.5f,
    int lineThickness=2
) const
```





**Parameters:**


* `image` Image to draw on 
* `results` Vector of pose results 
* `kptRadius` Keypoint circle radius 
* `kptThreshold` Minimum confidence to draw keypoint 
* `lineThickness` Skeleton line thickness 




        

<hr>



### function drawSkeletonsOnly 

_Draw only skeletons (no bounding boxes)_ 
```C++
inline void yolos::pose::YOLOPoseDetector::drawSkeletonsOnly (
    cv::Mat & image,
    const std::vector< PoseResult > & results,
    int kptRadius=4,
    float kptThreshold=0.5f,
    int lineThickness=2
) const
```




<hr>



### function getClassNames 

_Get class names._ 
```C++
inline const std::vector< std::string > & yolos::pose::YOLOPoseDetector::getClassNames () const
```




<hr>



### function ~YOLOPoseDetector 

```C++
virtual yolos::pose::YOLOPoseDetector::~YOLOPoseDetector () = default
```




<hr>
## Public Static Functions Documentation




### function getPoseSkeleton 

_Get COCO pose skeleton connections._ 
```C++
static inline const std::vector< std::pair< int, int > > & yolos::pose::YOLOPoseDetector::getPoseSkeleton () 
```




<hr>
## Protected Attributes Documentation




### variable buffer\_ 

```C++
preprocessing::InferenceBuffer yolos::pose::YOLOPoseDetector::buffer_;
```




<hr>



### variable classColors\_ 

```C++
std::vector<cv::Scalar> yolos::pose::YOLOPoseDetector::classColors_;
```




<hr>



### variable classNames\_ 

```C++
std::vector<std::string> yolos::pose::YOLOPoseDetector::classNames_;
```




<hr>
## Protected Static Attributes Documentation




### variable FEATURES\_PER\_KEYPOINT 

```C++
constexpr int yolos::pose::YOLOPoseDetector::FEATURES_PER_KEYPOINT;
```




<hr>



### variable NUM\_KEYPOINTS 

```C++
constexpr int yolos::pose::YOLOPoseDetector::NUM_KEYPOINTS;
```




<hr>
## Protected Functions Documentation




### function postprocess 

_Postprocess pose detection outputs._ 
```C++
inline std::vector< PoseResult > yolos::pose::YOLOPoseDetector::postprocess (
    const cv::Size & originalSize,
    const cv::Size & resizedShape,
    const std::vector< Ort::Value > & outputTensors,
    float confThreshold,
    float iouThreshold
) 
```




<hr>



### function postprocessV26 

_Postprocess YOLO26 pose detection outputs (end-to-end, NMS-free)_ 
```C++
inline std::vector< PoseResult > yolos::pose::YOLOPoseDetector::postprocessV26 (
    const cv::Size & originalSize,
    const cv::Size & resizedShape,
    const float * rawOutput,
    const std::vector< int64_t > & outputShape,
    float confThreshold
) 
```




<hr>



### function postprocessV8 

_Postprocess YOLOv8/v11 pose detection outputs (requires NMS)_ 
```C++
inline std::vector< PoseResult > yolos::pose::YOLOPoseDetector::postprocessV8 (
    const cv::Size & originalSize,
    const cv::Size & resizedShape,
    const float * rawOutput,
    const std::vector< int64_t > & outputShape,
    float confThreshold,
    float iouThreshold
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/tasks/pose.hpp`

