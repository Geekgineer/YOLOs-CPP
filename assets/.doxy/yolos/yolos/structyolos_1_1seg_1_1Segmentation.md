

# Struct yolos::seg::Segmentation



[**ClassList**](annotated.md) **>** [**yolos**](namespaceyolos.md) **>** [**seg**](namespaceyolos_1_1seg.md) **>** [**Segmentation**](structyolos_1_1seg_1_1Segmentation.md)



[_**Segmentation**_](structyolos_1_1seg_1_1Segmentation.md) _result containing bounding box, confidence, class ID, and mask._

* `#include <segmentation.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**BoundingBox**](structyolos_1_1BoundingBox.md) | [**box**](#variable-box)  <br>_Axis-aligned bounding box._  |
|  int | [**classId**](#variable-classid)   = `{0}`<br>_Class ID._  |
|  float | [**conf**](#variable-conf)   = `{0.0f}`<br>_Confidence score._  |
|  cv::Mat | [**mask**](#variable-mask)  <br>_Binary mask (CV\_8UC1) in original image coordinates._  |
















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**Segmentation**](#function-segmentation-12) () = default<br> |
|   | [**Segmentation**](#function-segmentation-22) (const [**BoundingBox**](structyolos_1_1BoundingBox.md) & box\_, float conf\_, int classId\_, const cv::Mat & mask\_) <br> |




























## Public Attributes Documentation




### variable box 

_Axis-aligned bounding box._ 
```C++
BoundingBox yolos::seg::Segmentation::box;
```




<hr>



### variable classId 

_Class ID._ 
```C++
int yolos::seg::Segmentation::classId;
```




<hr>



### variable conf 

_Confidence score._ 
```C++
float yolos::seg::Segmentation::conf;
```




<hr>



### variable mask 

_Binary mask (CV\_8UC1) in original image coordinates._ 
```C++
cv::Mat yolos::seg::Segmentation::mask;
```




<hr>
## Public Functions Documentation




### function Segmentation [1/2]

```C++
yolos::seg::Segmentation::Segmentation () = default
```




<hr>



### function Segmentation [2/2]

```C++
inline yolos::seg::Segmentation::Segmentation (
    const BoundingBox & box_,
    float conf_,
    int classId_,
    const cv::Mat & mask_
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/tasks/segmentation.hpp`

