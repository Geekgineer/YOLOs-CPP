

# Struct yolos::pose::PoseResult



[**ClassList**](annotated.md) **>** [**yolos**](namespaceyolos.md) **>** [**pose**](namespaceyolos_1_1pose.md) **>** [**PoseResult**](structyolos_1_1pose_1_1PoseResult.md)



_Pose estimation result containing bounding box, confidence, and keypoints._ 

* `#include <pose.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**BoundingBox**](structyolos_1_1BoundingBox.md) | [**box**](#variable-box)  <br>_Bounding box around the person._  |
|  int | [**classId**](#variable-classid)   = `{0}`<br>_Class ID (typically 0 for person)_  |
|  float | [**conf**](#variable-conf)   = `{0.0f}`<br>_Detection confidence._  |
|  std::vector&lt; [**KeyPoint**](structyolos_1_1KeyPoint.md) &gt; | [**keypoints**](#variable-keypoints)  <br>_Detected keypoints (17 for COCO format)_  |
















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**PoseResult**](#function-poseresult-12) () = default<br> |
|   | [**PoseResult**](#function-poseresult-22) (const [**BoundingBox**](structyolos_1_1BoundingBox.md) & box\_, float conf\_, int classId\_, const std::vector&lt; [**KeyPoint**](structyolos_1_1KeyPoint.md) &gt; & kpts) <br> |




























## Public Attributes Documentation




### variable box 

_Bounding box around the person._ 
```C++
BoundingBox yolos::pose::PoseResult::box;
```




<hr>



### variable classId 

_Class ID (typically 0 for person)_ 
```C++
int yolos::pose::PoseResult::classId;
```




<hr>



### variable conf 

_Detection confidence._ 
```C++
float yolos::pose::PoseResult::conf;
```




<hr>



### variable keypoints 

_Detected keypoints (17 for COCO format)_ 
```C++
std::vector<KeyPoint> yolos::pose::PoseResult::keypoints;
```




<hr>
## Public Functions Documentation




### function PoseResult [1/2]

```C++
yolos::pose::PoseResult::PoseResult () = default
```




<hr>



### function PoseResult [2/2]

```C++
inline yolos::pose::PoseResult::PoseResult (
    const BoundingBox & box_,
    float conf_,
    int classId_,
    const std::vector< KeyPoint > & kpts
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/tasks/pose.hpp`

