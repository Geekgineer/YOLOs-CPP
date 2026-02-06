

# Class yolos::OrtSessionBase



[**ClassList**](annotated.md) **>** [**yolos**](namespaceyolos.md) **>** [**OrtSessionBase**](classyolos_1_1OrtSessionBase.md)



_Base class for ONNX Runtime session management Handles model loading, session configuration, and common inference setup._ 

* `#include <session_base.hpp>`





Inherited by the following classes: [yolos::det::YOLODetector](classyolos_1_1det_1_1YOLODetector.md),  [yolos::obb::YOLOOBBDetector](classyolos_1_1obb_1_1YOLOOBBDetector.md),  [yolos::pose::YOLOPoseDetector](classyolos_1_1pose_1_1YOLOPoseDetector.md),  [yolos::seg::YOLOSegDetector](classyolos_1_1seg_1_1YOLOSegDetector.md)
































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**OrtSessionBase**](#function-ortsessionbase-13) (const std::string & modelPath, bool useGPU=false, int numThreads=0) <br>_Constructor - loads and initializes the ONNX model._  |
|   | [**OrtSessionBase**](#function-ortsessionbase-23) (const [**OrtSessionBase**](classyolos_1_1OrtSessionBase.md) &) = delete<br> |
|   | [**OrtSessionBase**](#function-ortsessionbase-33) ([**OrtSessionBase**](classyolos_1_1OrtSessionBase.md) &&) = default<br> |
|  const std::string & | [**getDevice**](#function-getdevice) () noexcept const<br>_Get the device being used for inference._  |
|  cv::Size | [**getInputShape**](#function-getinputshape) () noexcept const<br>_Get the input image shape expected by the model._  |
|  size\_t | [**getNumInputNodes**](#function-getnuminputnodes) () noexcept const<br>_Get the number of input nodes._  |
|  size\_t | [**getNumOutputNodes**](#function-getnumoutputnodes) () noexcept const<br>_Get the number of output nodes._  |
|  bool | [**isDynamicBatchSize**](#function-isdynamicbatchsize) () noexcept const<br>_Check if batch size is dynamic._  |
|  bool | [**isDynamicInputShape**](#function-isdynamicinputshape) () noexcept const<br>_Check if input shape is dynamic._  |
|  [**OrtSessionBase**](classyolos_1_1OrtSessionBase.md) & | [**operator=**](#function-operator) (const [**OrtSessionBase**](classyolos_1_1OrtSessionBase.md) &) = delete<br> |
|  [**OrtSessionBase**](classyolos_1_1OrtSessionBase.md) & | [**operator=**](#function-operator_1) ([**OrtSessionBase**](classyolos_1_1OrtSessionBase.md) &&) = default<br> |
| virtual  | [**~OrtSessionBase**](#function-ortsessionbase) () = default<br> |








## Protected Attributes

| Type | Name |
| ---: | :--- |
|  std::string | [**device\_**](#variable-device_)   = `{"cpu"}`<br> |
|  Ort::Env | [**env\_**](#variable-env_)   = `{nullptr}`<br> |
|  std::vector&lt; Ort::AllocatedStringPtr &gt; | [**inputNameAllocs\_**](#variable-inputnameallocs_)  <br> |
|  std::vector&lt; const char \* &gt; | [**inputNames\_**](#variable-inputnames_)  <br> |
|  cv::Size | [**inputShape\_**](#variable-inputshape_)  <br> |
|  bool | [**isDynamicBatchSize\_**](#variable-isdynamicbatchsize_)   = `{false}`<br> |
|  bool | [**isDynamicInputShape\_**](#variable-isdynamicinputshape_)   = `{false}`<br> |
|  size\_t | [**numInputNodes\_**](#variable-numinputnodes_)   = `{0}`<br> |
|  size\_t | [**numOutputNodes\_**](#variable-numoutputnodes_)   = `{0}`<br> |
|  std::vector&lt; Ort::AllocatedStringPtr &gt; | [**outputNameAllocs\_**](#variable-outputnameallocs_)  <br> |
|  std::vector&lt; const char \* &gt; | [**outputNames\_**](#variable-outputnames_)  <br> |
|  Ort::SessionOptions | [**sessionOptions\_**](#variable-sessionoptions_)   = `{nullptr}`<br> |
|  Ort::Session | [**session\_**](#variable-session_)   = `{nullptr}`<br> |
















## Protected Functions

| Type | Name |
| ---: | :--- |
|  Ort::Value | [**createInputTensor**](#function-createinputtensor) (float \* blob, const std::vector&lt; int64\_t &gt; & inputTensorShape) <br>_Create an input tensor from a blob._  |
|  std::vector&lt; Ort::Value &gt; | [**runInference**](#function-runinference) (Ort::Value & inputTensor) <br>_Run inference with the given input tensor._  |




## Public Functions Documentation




### function OrtSessionBase [1/3]

_Constructor - loads and initializes the ONNX model._ 
```C++
inline yolos::OrtSessionBase::OrtSessionBase (
    const std::string & modelPath,
    bool useGPU=false,
    int numThreads=0
) 
```





**Parameters:**


* `modelPath` Path to the ONNX model file 
* `useGPU` Whether to use GPU (CUDA) for inference 
* `numThreads` Number of intra-op threads (0 = auto) 




        

<hr>



### function OrtSessionBase [2/3]

```C++
yolos::OrtSessionBase::OrtSessionBase (
    const OrtSessionBase &
) = delete
```




<hr>



### function OrtSessionBase [3/3]

```C++
yolos::OrtSessionBase::OrtSessionBase (
    OrtSessionBase &&
) = default
```




<hr>



### function getDevice 

_Get the device being used for inference._ 
```C++
inline const std::string & yolos::OrtSessionBase::getDevice () noexcept const
```




<hr>



### function getInputShape 

_Get the input image shape expected by the model._ 
```C++
inline cv::Size yolos::OrtSessionBase::getInputShape () noexcept const
```




<hr>



### function getNumInputNodes 

_Get the number of input nodes._ 
```C++
inline size_t yolos::OrtSessionBase::getNumInputNodes () noexcept const
```




<hr>



### function getNumOutputNodes 

_Get the number of output nodes._ 
```C++
inline size_t yolos::OrtSessionBase::getNumOutputNodes () noexcept const
```




<hr>



### function isDynamicBatchSize 

_Check if batch size is dynamic._ 
```C++
inline bool yolos::OrtSessionBase::isDynamicBatchSize () noexcept const
```




<hr>



### function isDynamicInputShape 

_Check if input shape is dynamic._ 
```C++
inline bool yolos::OrtSessionBase::isDynamicInputShape () noexcept const
```




<hr>



### function operator= 

```C++
OrtSessionBase & yolos::OrtSessionBase::operator= (
    const OrtSessionBase &
) = delete
```




<hr>



### function operator= 

```C++
OrtSessionBase & yolos::OrtSessionBase::operator= (
    OrtSessionBase &&
) = default
```




<hr>



### function ~OrtSessionBase 

```C++
virtual yolos::OrtSessionBase::~OrtSessionBase () = default
```




<hr>
## Protected Attributes Documentation




### variable device\_ 

```C++
std::string yolos::OrtSessionBase::device_;
```




<hr>



### variable env\_ 

```C++
Ort::Env yolos::OrtSessionBase::env_;
```




<hr>



### variable inputNameAllocs\_ 

```C++
std::vector<Ort::AllocatedStringPtr> yolos::OrtSessionBase::inputNameAllocs_;
```




<hr>



### variable inputNames\_ 

```C++
std::vector<const char*> yolos::OrtSessionBase::inputNames_;
```




<hr>



### variable inputShape\_ 

```C++
cv::Size yolos::OrtSessionBase::inputShape_;
```




<hr>



### variable isDynamicBatchSize\_ 

```C++
bool yolos::OrtSessionBase::isDynamicBatchSize_;
```




<hr>



### variable isDynamicInputShape\_ 

```C++
bool yolos::OrtSessionBase::isDynamicInputShape_;
```




<hr>



### variable numInputNodes\_ 

```C++
size_t yolos::OrtSessionBase::numInputNodes_;
```




<hr>



### variable numOutputNodes\_ 

```C++
size_t yolos::OrtSessionBase::numOutputNodes_;
```




<hr>



### variable outputNameAllocs\_ 

```C++
std::vector<Ort::AllocatedStringPtr> yolos::OrtSessionBase::outputNameAllocs_;
```




<hr>



### variable outputNames\_ 

```C++
std::vector<const char*> yolos::OrtSessionBase::outputNames_;
```




<hr>



### variable sessionOptions\_ 

```C++
Ort::SessionOptions yolos::OrtSessionBase::sessionOptions_;
```




<hr>



### variable session\_ 

```C++
Ort::Session yolos::OrtSessionBase::session_;
```




<hr>
## Protected Functions Documentation




### function createInputTensor 

_Create an input tensor from a blob._ 
```C++
inline Ort::Value yolos::OrtSessionBase::createInputTensor (
    float * blob,
    const std::vector< int64_t > & inputTensorShape
) 
```





**Parameters:**


* `blob` Pointer to the input data 
* `inputTensorShape` Shape of the input tensor 



**Returns:**

ONNX Runtime input tensor 





        

<hr>



### function runInference 

_Run inference with the given input tensor._ 
```C++
inline std::vector< Ort::Value > yolos::OrtSessionBase::runInference (
    Ort::Value & inputTensor
) 
```





**Parameters:**


* `inputTensor` Input tensor 



**Returns:**

Vector of output tensors 





        

<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/core/session_base.hpp`

