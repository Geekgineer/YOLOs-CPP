

# Struct yolos::KeyPoint



[**ClassList**](annotated.md) **>** [**yolos**](namespaceyolos.md) **>** [**KeyPoint**](structyolos_1_1KeyPoint.md)





* `#include <types.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  float | [**confidence**](#variable-confidence)   = `{0.0f}`<br>_Confidence score._  |
|  float | [**x**](#variable-x)   = `{0.0f}`<br>_X-coordinate._  |
|  float | [**y**](#variable-y)   = `{0.0f}`<br>_Y-coordinate._  |
















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**KeyPoint**](#function-keypoint-12) () = default<br> |
|   | [**KeyPoint**](#function-keypoint-22) (float x\_, float y\_, float conf\_=0.0f) <br> |




























## Public Attributes Documentation




### variable confidence 

_Confidence score._ 
```C++
float yolos::KeyPoint::confidence;
```




<hr>



### variable x 

_X-coordinate._ 
```C++
float yolos::KeyPoint::x;
```




<hr>



### variable y 

_Y-coordinate._ 
```C++
float yolos::KeyPoint::y;
```




<hr>
## Public Functions Documentation




### function KeyPoint [1/2]

```C++
yolos::KeyPoint::KeyPoint () = default
```




<hr>



### function KeyPoint [2/2]

```C++
inline yolos::KeyPoint::KeyPoint (
    float x_,
    float y_,
    float conf_=0.0f
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/core/types.hpp`

