

# Class yolos::det::YOLOv8Detector



[**ClassList**](annotated.md) **>** [**yolos**](namespaceyolos.md) **>** [**det**](namespaceyolos_1_1det.md) **>** [**YOLOv8Detector**](classyolos_1_1det_1_1YOLOv8Detector.md)



_YOLOv8 detector (forces standard postprocessing)_ 

* `#include <detection.hpp>`



Inherits the following classes: [yolos::det::YOLODetector](classyolos_1_1det_1_1YOLODetector.md)










































































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**YOLOv8Detector**](#function-yolov8detector) (const std::string & modelPath, const std::string & labelsPath, bool useGPU=false) <br> |


## Public Functions inherited from yolos::det::YOLODetector

See [yolos::det::YOLODetector](classyolos_1_1det_1_1YOLODetector.md)

| Type | Name |
| ---: | :--- |
|   | [**YOLODetector**](classyolos_1_1det_1_1YOLODetector.md#function-yolodetector) (const std::string & modelPath, const std::string & labelsPath, bool useGPU=false, [**YOLOVersion**](namespaceyolos.md#enum-yoloversion) version=YOLOVersion::Auto) <br>_Constructor._  |
| virtual std::vector&lt; [**Detection**](structyolos_1_1det_1_1Detection.md) &gt; | [**detect**](classyolos_1_1det_1_1YOLODetector.md#function-detect) (const cv::Mat & image, float confThreshold=0.4f, float iouThreshold=0.45f) <br>_Run detection on an image (optimized with buffer reuse)_  |
|  void | [**drawDetections**](classyolos_1_1det_1_1YOLODetector.md#function-drawdetections) (cv::Mat & image, const std::vector&lt; [**Detection**](structyolos_1_1det_1_1Detection.md) &gt; & detections) const<br>_Draw detections on an image._  |
|  void | [**drawDetectionsWithMask**](classyolos_1_1det_1_1YOLODetector.md#function-drawdetectionswithmask) (cv::Mat & image, const std::vector&lt; [**Detection**](structyolos_1_1det_1_1Detection.md) &gt; & detections, float alpha=0.4f) const<br>_Draw detections with semi-transparent mask fill._  |
|  const std::vector&lt; cv::Scalar &gt; & | [**getClassColors**](classyolos_1_1det_1_1YOLODetector.md#function-getclasscolors) () const<br>_Get class colors._  |
|  const std::vector&lt; std::string &gt; & | [**getClassNames**](classyolos_1_1det_1_1YOLODetector.md#function-getclassnames) () const<br>_Get class names._  |
| virtual  | [**~YOLODetector**](classyolos_1_1det_1_1YOLODetector.md#function-yolodetector) () = default<br> |


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






















## Protected Attributes inherited from yolos::det::YOLODetector

See [yolos::det::YOLODetector](classyolos_1_1det_1_1YOLODetector.md)

| Type | Name |
| ---: | :--- |
|  [**preprocessing::InferenceBuffer**](structyolos_1_1preprocessing_1_1InferenceBuffer.md) | [**buffer\_**](classyolos_1_1det_1_1YOLODetector.md#variable-buffer_)  <br> |
|  std::vector&lt; cv::Scalar &gt; | [**classColors\_**](classyolos_1_1det_1_1YOLODetector.md#variable-classcolors_)  <br> |
|  std::vector&lt; std::string &gt; | [**classNames\_**](classyolos_1_1det_1_1YOLODetector.md#variable-classnames_)  <br> |
|  [**YOLOVersion**](namespaceyolos.md#enum-yoloversion) | [**version\_**](classyolos_1_1det_1_1YOLODetector.md#variable-version_)   = `{YOLOVersion::Auto}`<br> |


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














































## Protected Functions inherited from yolos::det::YOLODetector

See [yolos::det::YOLODetector](classyolos_1_1det_1_1YOLODetector.md)

| Type | Name |
| ---: | :--- |
|  [**YOLOVersion**](namespaceyolos.md#enum-yoloversion) | [**detectVersion**](classyolos_1_1det_1_1YOLODetector.md#function-detectversion) (const std::vector&lt; Ort::Value &gt; & outputTensors) <br>_Detect YOLO version from output tensors._  |
| virtual std::vector&lt; [**Detection**](structyolos_1_1det_1_1Detection.md) &gt; | [**postprocess**](classyolos_1_1det_1_1YOLODetector.md#function-postprocess) (const cv::Size & originalSize, const cv::Size & resizedShape, const std::vector&lt; Ort::Value &gt; & outputTensors, [**YOLOVersion**](namespaceyolos.md#enum-yoloversion) version, float confThreshold, float iouThreshold) <br>_Postprocess based on detected version._  |
| virtual std::vector&lt; [**Detection**](structyolos_1_1det_1_1Detection.md) &gt; | [**postprocessNAS**](classyolos_1_1det_1_1YOLODetector.md#function-postprocessnas) (const cv::Size & originalSize, const cv::Size & resizedShape, const std::vector&lt; Ort::Value &gt; & outputTensors, float confThreshold, float iouThreshold) <br>_Postprocess for YOLO-NAS format (two outputs: boxes and scores)_  |
| virtual std::vector&lt; [**Detection**](structyolos_1_1det_1_1Detection.md) &gt; | [**postprocessStandard**](classyolos_1_1det_1_1YOLODetector.md#function-postprocessstandard) (const cv::Size & originalSize, const cv::Size & resizedShape, const std::vector&lt; Ort::Value &gt; & outputTensors, float confThreshold, float iouThreshold) <br>_Standard postprocess for YOLOv8/v11 format [batch, features, boxes] Optimized: single box storage with batched NMS._  |
| virtual std::vector&lt; [**Detection**](structyolos_1_1det_1_1Detection.md) &gt; | [**postprocessV10**](classyolos_1_1det_1_1YOLODetector.md#function-postprocessv10) (const cv::Size & originalSize, const cv::Size & resizedShape, const std::vector&lt; Ort::Value &gt; & outputTensors, float confThreshold, float) <br>_Postprocess for YOLOv10 format [batch, boxes, 6] (end-to-end, no NMS needed)_  |
| virtual std::vector&lt; [**Detection**](structyolos_1_1det_1_1Detection.md) &gt; | [**postprocessV7**](classyolos_1_1det_1_1YOLODetector.md#function-postprocessv7) (const cv::Size & originalSize, const cv::Size & resizedShape, const std::vector&lt; Ort::Value &gt; & outputTensors, float confThreshold, float iouThreshold) <br>_Postprocess for YOLOv7 format [batch, boxes, features]._  |


## Protected Functions inherited from yolos::OrtSessionBase

See [yolos::OrtSessionBase](classyolos_1_1OrtSessionBase.md)

| Type | Name |
| ---: | :--- |
|  Ort::Value | [**createInputTensor**](classyolos_1_1OrtSessionBase.md#function-createinputtensor) (float \* blob, const std::vector&lt; int64\_t &gt; & inputTensorShape) <br>_Create an input tensor from a blob._  |
|  std::vector&lt; Ort::Value &gt; | [**runInference**](classyolos_1_1OrtSessionBase.md#function-runinference) (Ort::Value & inputTensor) <br>_Run inference with the given input tensor._  |








## Public Functions Documentation




### function YOLOv8Detector 

```C++
inline yolos::det::YOLOv8Detector::YOLOv8Detector (
    const std::string & modelPath,
    const std::string & labelsPath,
    bool useGPU=false
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/tasks/detection.hpp`

