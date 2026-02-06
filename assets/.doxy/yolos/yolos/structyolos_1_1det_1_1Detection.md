

# Struct yolos::det::Detection



[**ClassList**](annotated.md) **>** [**yolos**](namespaceyolos.md) **>** [**det**](namespaceyolos_1_1det.md) **>** [**Detection**](structyolos_1_1det_1_1Detection.md)



[_**Detection**_](structyolos_1_1det_1_1Detection.md) _result containing bounding box, confidence, and class ID._

* `#include <detection.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**BoundingBox**](structyolos_1_1BoundingBox.md) | [**box**](#variable-box)  <br>_Axis-aligned bounding box._  |
|  int | [**classId**](#variable-classid)   = `{-1}`<br>_Class ID._  |
|  float | [**conf**](#variable-conf)   = `{0.0f}`<br>_Confidence score._  |
















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**Detection**](#function-detection-12) () = default<br> |
|   | [**Detection**](#function-detection-22) (const [**BoundingBox**](structyolos_1_1BoundingBox.md) & box\_, float conf\_, int classId\_) <br> |




























## Public Attributes Documentation




### variable box 

_Axis-aligned bounding box._ 
```C++
BoundingBox yolos::det::Detection::box;
```




<hr>



### variable classId 

_Class ID._ 
```C++
int yolos::det::Detection::classId;
```




<hr>



### variable conf 

_Confidence score._ 
```C++
float yolos::det::Detection::conf;
```




<hr>
## Public Functions Documentation




### function Detection [1/2]

```C++
yolos::det::Detection::Detection () = default
```




<hr>



### function Detection [2/2]

```C++
inline yolos::det::Detection::Detection (
    const BoundingBox & box_,
    float conf_,
    int classId_
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/tasks/detection.hpp`

