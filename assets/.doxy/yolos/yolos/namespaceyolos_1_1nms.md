

# Namespace yolos::nms



[**Namespace List**](namespaces.md) **>** [**yolos**](namespaceyolos.md) **>** [**nms**](namespaceyolos_1_1nms.md)










































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**NMSBoxes**](#function-nmsboxes) (const std::vector&lt; [**BoundingBox**](structyolos_1_1BoundingBox.md) &gt; & boxes, const std::vector&lt; float &gt; & scores, float scoreThreshold, float nmsThreshold, std::vector&lt; int &gt; & indices) <br>_Perform Non-Maximum Suppression on bounding boxes._  |
|  void | [**NMSBoxesBatched**](#function-nmsboxesbatched) (const std::vector&lt; [**BoundingBox**](structyolos_1_1BoundingBox.md) &gt; & boxes, const std::vector&lt; float &gt; & scores, const std::vector&lt; int &gt; & classIds, float scoreThreshold, float nmsThreshold, std::vector&lt; int &gt; & indices) <br>_Perform class-aware NMS by offsetting boxes by class ID._  |
|  void | [**NMSBoxesF**](#function-nmsboxesf) (const std::vector&lt; cv::Rect2f &gt; & boxes, const std::vector&lt; float &gt; & scores, float scoreThreshold, float nmsThreshold, std::vector&lt; int &gt; & indices) <br>_Perform NMS on float-precision bounding boxes (for letterbox space)_  |
|  void | [**NMSBoxesFBatched**](#function-nmsboxesfbatched) (const std::vector&lt; cv::Rect2f &gt; & boxes, const std::vector&lt; float &gt; & scores, const std::vector&lt; int &gt; & classIds, float scoreThreshold, float nmsThreshold, std::vector&lt; int &gt; & indices) <br>_Perform class-aware NMS on float-precision boxes._  |
|  std::vector&lt; int &gt; | [**NMSRotated**](#function-nmsrotated) (const std::vector&lt; [**OrientedBoundingBox**](structyolos_1_1OrientedBoundingBox.md) &gt; & boxes, const std::vector&lt; float &gt; & scores, float nmsThreshold=0.45f, int maxDet=300) <br>_Perform NMS on oriented bounding boxes using rotated IoU._  |
|  std::vector&lt; int &gt; | [**NMSRotatedBatched**](#function-nmsrotatedbatched) (const std::vector&lt; [**OrientedBoundingBox**](structyolos_1_1OrientedBoundingBox.md) &gt; & boxes, const std::vector&lt; float &gt; & scores, const std::vector&lt; int &gt; & classIds, float nmsThreshold=0.45f, int maxDet=300) <br>_Perform class-aware NMS on oriented bounding boxes._  |
|  float | [**computeRotatedIoU**](#function-computerotatediou) (const [**OrientedBoundingBox**](structyolos_1_1OrientedBoundingBox.md) & box1, const [**OrientedBoundingBox**](structyolos_1_1OrientedBoundingBox.md) & box2) <br>_Compute IoU between two oriented bounding boxes using OpenCV._  |




























## Public Functions Documentation




### function NMSBoxes 

_Perform Non-Maximum Suppression on bounding boxes._ 
```C++
inline void yolos::nms::NMSBoxes (
    const std::vector< BoundingBox > & boxes,
    const std::vector< float > & scores,
    float scoreThreshold,
    float nmsThreshold,
    std::vector< int > & indices
) 
```





**Parameters:**


* `boxes` Vector of bounding boxes 
* `scores` Vector of confidence scores 
* `scoreThreshold` Minimum score to consider 
* `nmsThreshold` IoU threshold for suppression 
* `indices` Output indices of boxes that survived NMS 




        

<hr>



### function NMSBoxesBatched 

_Perform class-aware NMS by offsetting boxes by class ID._ 
```C++
inline void yolos::nms::NMSBoxesBatched (
    const std::vector< BoundingBox > & boxes,
    const std::vector< float > & scores,
    const std::vector< int > & classIds,
    float scoreThreshold,
    float nmsThreshold,
    std::vector< int > & indices
) 
```





**Parameters:**


* `boxes` Vector of bounding boxes 
* `scores` Vector of confidence scores 
* `classIds` Vector of class IDs 
* `scoreThreshold` Minimum score to consider 
* `nmsThreshold` IoU threshold for suppression 
* `indices` Output indices of boxes that survived NMS 




        

<hr>



### function NMSBoxesF 

_Perform NMS on float-precision bounding boxes (for letterbox space)_ 
```C++
inline void yolos::nms::NMSBoxesF (
    const std::vector< cv::Rect2f > & boxes,
    const std::vector< float > & scores,
    float scoreThreshold,
    float nmsThreshold,
    std::vector< int > & indices
) 
```





**Parameters:**


* `boxes` Vector of cv::Rect2f boxes 
* `scores` Vector of confidence scores 
* `scoreThreshold` Minimum score to consider 
* `nmsThreshold` IoU threshold for suppression 
* `indices` Output indices of boxes that survived NMS 




        

<hr>



### function NMSBoxesFBatched 

_Perform class-aware NMS on float-precision boxes._ 
```C++
inline void yolos::nms::NMSBoxesFBatched (
    const std::vector< cv::Rect2f > & boxes,
    const std::vector< float > & scores,
    const std::vector< int > & classIds,
    float scoreThreshold,
    float nmsThreshold,
    std::vector< int > & indices
) 
```




<hr>



### function NMSRotated 

_Perform NMS on oriented bounding boxes using rotated IoU._ 
```C++
inline std::vector< int > yolos::nms::NMSRotated (
    const std::vector< OrientedBoundingBox > & boxes,
    const std::vector< float > & scores,
    float nmsThreshold=0.45f,
    int maxDet=300
) 
```





**Parameters:**


* `boxes` Vector of oriented bounding boxes 
* `scores` Vector of confidence scores 
* `nmsThreshold` IoU threshold for suppression 
* `maxDet` Maximum number of detections to keep 



**Returns:**

Indices of boxes that survived NMS 





        

<hr>



### function NMSRotatedBatched 

_Perform class-aware NMS on oriented bounding boxes._ 
```C++
inline std::vector< int > yolos::nms::NMSRotatedBatched (
    const std::vector< OrientedBoundingBox > & boxes,
    const std::vector< float > & scores,
    const std::vector< int > & classIds,
    float nmsThreshold=0.45f,
    int maxDet=300
) 
```





**Parameters:**


* `boxes` Vector of oriented bounding boxes 
* `scores` Vector of confidence scores 
* `classIds` Vector of class IDs 
* `nmsThreshold` IoU threshold for suppression 
* `maxDet` Maximum number of detections to keep 



**Returns:**

Indices of boxes that survived NMS 





        

<hr>



### function computeRotatedIoU 

_Compute IoU between two oriented bounding boxes using OpenCV._ 
```C++
inline float yolos::nms::computeRotatedIoU (
    const OrientedBoundingBox & box1,
    const OrientedBoundingBox & box2
) 
```





**Parameters:**


* `box1` First oriented bounding box 
* `box2` Second oriented bounding box 



**Returns:**

IoU value between 0 and 1 





        

<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/core/nms.hpp`

