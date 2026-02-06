

# Struct yolos::preprocessing::InferenceBuffer



[**ClassList**](annotated.md) **>** [**yolos**](namespaceyolos.md) **>** [**preprocessing**](namespaceyolos_1_1preprocessing.md) **>** [**InferenceBuffer**](structyolos_1_1preprocessing_1_1InferenceBuffer.md)



_Pre-allocated inference buffer to avoid per-frame allocations._ 

* `#include <preprocessing.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  std::vector&lt; float &gt; | [**blob**](#variable-blob)  <br>_CHW format blob for ONNX._  |
|  cv::Size | [**lastInputSize**](#variable-lastinputsize)  <br>_Last input size (for reuse check)_  |
|  cv::Size | [**lastTargetSize**](#variable-lasttargetsize)  <br>_Last target size._  |
|  cv::Mat | [**resized**](#variable-resized)  <br>_Letterboxed image._  |
|  cv::Mat | [**rgbFloat**](#variable-rgbfloat)  <br>_RGB float image._  |
















## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**ensureCapacity**](#function-ensurecapacity) (int height, int width, int channels=3) <br>_Ensure blob has required capacity._  |




























## Public Attributes Documentation




### variable blob 

_CHW format blob for ONNX._ 
```C++
std::vector<float> yolos::preprocessing::InferenceBuffer::blob;
```




<hr>



### variable lastInputSize 

_Last input size (for reuse check)_ 
```C++
cv::Size yolos::preprocessing::InferenceBuffer::lastInputSize;
```




<hr>



### variable lastTargetSize 

_Last target size._ 
```C++
cv::Size yolos::preprocessing::InferenceBuffer::lastTargetSize;
```




<hr>



### variable resized 

_Letterboxed image._ 
```C++
cv::Mat yolos::preprocessing::InferenceBuffer::resized;
```




<hr>



### variable rgbFloat 

_RGB float image._ 
```C++
cv::Mat yolos::preprocessing::InferenceBuffer::rgbFloat;
```




<hr>
## Public Functions Documentation




### function ensureCapacity 

_Ensure blob has required capacity._ 
```C++
inline void yolos::preprocessing::InferenceBuffer::ensureCapacity (
    int height,
    int width,
    int channels=3
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/core/preprocessing.hpp`

