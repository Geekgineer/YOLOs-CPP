

# Struct yolos::obb::OBBResult



[**ClassList**](annotated.md) **>** [**yolos**](namespaceyolos.md) **>** [**obb**](namespaceyolos_1_1obb.md) **>** [**OBBResult**](structyolos_1_1obb_1_1OBBResult.md)



_OBB detection result containing oriented bounding box, confidence, and class ID._ 

* `#include <obb.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**OrientedBoundingBox**](structyolos_1_1OrientedBoundingBox.md) | [**box**](#variable-box)  <br>_Oriented bounding box (center-based with angle)_  |
|  int | [**classId**](#variable-classid)   = `{-1}`<br>_Class ID._  |
|  float | [**conf**](#variable-conf)   = `{0.0f}`<br>_Confidence score._  |
















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**OBBResult**](#function-obbresult-12) () = default<br> |
|   | [**OBBResult**](#function-obbresult-22) (const [**OrientedBoundingBox**](structyolos_1_1OrientedBoundingBox.md) & box\_, float conf\_, int classId\_) <br> |




























## Public Attributes Documentation




### variable box 

_Oriented bounding box (center-based with angle)_ 
```C++
OrientedBoundingBox yolos::obb::OBBResult::box;
```




<hr>



### variable classId 

_Class ID._ 
```C++
int yolos::obb::OBBResult::classId;
```




<hr>



### variable conf 

_Confidence score._ 
```C++
float yolos::obb::OBBResult::conf;
```




<hr>
## Public Functions Documentation




### function OBBResult [1/2]

```C++
yolos::obb::OBBResult::OBBResult () = default
```




<hr>



### function OBBResult [2/2]

```C++
inline yolos::obb::OBBResult::OBBResult (
    const OrientedBoundingBox & box_,
    float conf_,
    int classId_
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/tasks/obb.hpp`

