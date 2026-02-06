

# Namespace yolos::version



[**Namespace List**](namespaces.md) **>** [**yolos**](namespaceyolos.md) **>** [**version**](namespaceyolos_1_1version.md)










































## Public Functions

| Type | Name |
| ---: | :--- |
|  [**YOLOVersion**](namespaceyolos.md#enum-yoloversion) | [**detectClassificationVersion**](#function-detectclassificationversion) (const std::vector&lt; int64\_t &gt; & outputShape) <br>_Detect YOLO version for classification model._  |
|  [**YOLOVersion**](namespaceyolos.md#enum-yoloversion) | [**detectFromOutputShape**](#function-detectfromoutputshape) (const std::vector&lt; int64\_t &gt; & outputShape, size\_t numOutputs=1) <br>_Detect YOLO version from detection model output tensor shape._  |
|  bool | [**requiresNMS**](#function-requiresnms) ([**YOLOVersion**](namespaceyolos.md#enum-yoloversion) version) <br>_Check if version requires NMS post-processing._  |
|  std::string | [**toString**](#function-tostring) ([**YOLOVersion**](namespaceyolos.md#enum-yoloversion) version) <br>_Convert YOLOVersion enum to string._  |




























## Public Functions Documentation




### function detectClassificationVersion 

_Detect YOLO version for classification model._ 
```C++
inline YOLOVersion yolos::version::detectClassificationVersion (
    const std::vector< int64_t > & outputShape
) 
```





**Parameters:**


* `outputShape` The shape of the output tensor 



**Returns:**

Detected YOLOVersion (V11 or V12 for classification) 





        

<hr>



### function detectFromOutputShape 

_Detect YOLO version from detection model output tensor shape._ 
```C++
inline YOLOVersion yolos::version::detectFromOutputShape (
    const std::vector< int64_t > & outputShape,
    size_t numOutputs=1
) 
```





**Parameters:**


* `outputShape` The shape of the first output tensor [batch, dim1, dim2, ...] 
* `numOutputs` Number of output tensors from the model 



**Returns:**

Detected YOLOVersion 





        

<hr>



### function requiresNMS 

_Check if version requires NMS post-processing._ 
```C++
inline bool yolos::version::requiresNMS (
    YOLOVersion version
) 
```




<hr>



### function toString 

_Convert YOLOVersion enum to string._ 
```C++
inline std::string yolos::version::toString (
    YOLOVersion version
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/core/version.hpp`

