

# Namespace yolos::drawing



[**Namespace List**](namespaces.md) **>** [**yolos**](namespaceyolos.md) **>** [**drawing**](namespaceyolos_1_1drawing.md)










































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**drawBoundingBox**](#function-drawboundingbox) (cv::Mat & image, const [**BoundingBox**](structyolos_1_1BoundingBox.md) & box, const std::string & label, const cv::Scalar & color, int thickness=2) <br>_Draw a single bounding box with label on an image._  |
|  void | [**drawBoundingBoxWithMask**](#function-drawboundingboxwithmask) (cv::Mat & image, const [**BoundingBox**](structyolos_1_1BoundingBox.md) & box, const std::string & label, const cv::Scalar & color, float maskAlpha=0.4f) <br>_Draw a bounding box with semi-transparent mask fill._  |
|  void | [**drawOrientedBoundingBox**](#function-draworientedboundingbox) (cv::Mat & image, const [**OrientedBoundingBox**](structyolos_1_1OrientedBoundingBox.md) & obb, const std::string & label, const cv::Scalar & color, int thickness=2) <br>_Draw an oriented bounding box on an image._  |
|  void | [**drawPoseSkeleton**](#function-drawposeskeleton) (cv::Mat & image, const std::vector&lt; [**KeyPoint**](structyolos_1_1KeyPoint.md) &gt; & keypoints, const std::vector&lt; std::pair&lt; int, int &gt;&gt; & skeleton, int kptRadius=4, float kptThreshold=0.5f, int lineThickness=2) <br>_Draw pose keypoints and skeleton on an image._  |
|  void | [**drawSegmentationMask**](#function-drawsegmentationmask) (cv::Mat & image, const cv::Mat & mask, const cv::Scalar & color, float alpha=0.5f) <br>_Draw a segmentation mask on an image._  |
|  std::vector&lt; cv::Scalar &gt; | [**generateColors**](#function-generatecolors) (const std::vector&lt; std::string &gt; & classNames, int seed=42) <br>_Generate consistent random colors for each class._  |
|  const std::vector&lt; cv::Scalar &gt; & | [**getPosePalette**](#function-getposepalette) () <br>_Get the Ultralytics pose palette colors._  |




























## Public Functions Documentation




### function drawBoundingBox 

_Draw a single bounding box with label on an image._ 
```C++
inline void yolos::drawing::drawBoundingBox (
    cv::Mat & image,
    const BoundingBox & box,
    const std::string & label,
    const cv::Scalar & color,
    int thickness=2
) 
```





**Parameters:**


* `image` Image to draw on 
* `box` Bounding box 
* `label` Text label 
* `color` Box color 
* `thickness` Line thickness 




        

<hr>



### function drawBoundingBoxWithMask 

_Draw a bounding box with semi-transparent mask fill._ 
```C++
inline void yolos::drawing::drawBoundingBoxWithMask (
    cv::Mat & image,
    const BoundingBox & box,
    const std::string & label,
    const cv::Scalar & color,
    float maskAlpha=0.4f
) 
```





**Parameters:**


* `image` Image to draw on 
* `box` Bounding box 
* `label` Text label 
* `color` Box color 
* `maskAlpha` Transparency of the mask fill (0-1) 




        

<hr>



### function drawOrientedBoundingBox 

_Draw an oriented bounding box on an image._ 
```C++
inline void yolos::drawing::drawOrientedBoundingBox (
    cv::Mat & image,
    const OrientedBoundingBox & obb,
    const std::string & label,
    const cv::Scalar & color,
    int thickness=2
) 
```





**Parameters:**


* `image` Image to draw on 
* `obb` Oriented bounding box 
* `label` Text label 
* `color` Box color 
* `thickness` Line thickness 




        

<hr>



### function drawPoseSkeleton 

_Draw pose keypoints and skeleton on an image._ 
```C++
inline void yolos::drawing::drawPoseSkeleton (
    cv::Mat & image,
    const std::vector< KeyPoint > & keypoints,
    const std::vector< std::pair< int, int >> & skeleton,
    int kptRadius=4,
    float kptThreshold=0.5f,
    int lineThickness=2
) 
```





**Parameters:**


* `image` Image to draw on 
* `keypoints` Vector of keypoints 
* `skeleton` Skeleton connections 
* `kptRadius` Keypoint circle radius 
* `kptThreshold` Minimum confidence to draw keypoint 
* `lineThickness` Skeleton line thickness 




        

<hr>



### function drawSegmentationMask 

_Draw a segmentation mask on an image._ 
```C++
inline void yolos::drawing::drawSegmentationMask (
    cv::Mat & image,
    const cv::Mat & mask,
    const cv::Scalar & color,
    float alpha=0.5f
) 
```





**Parameters:**


* `image` Image to draw on 
* `mask` Binary mask (CV\_8UC1) 
* `color` Mask color 
* `alpha` Mask transparency (0-1) 




        

<hr>



### function generateColors 

_Generate consistent random colors for each class._ 
```C++
inline std::vector< cv::Scalar > yolos::drawing::generateColors (
    const std::vector< std::string > & classNames,
    int seed=42
) 
```





**Parameters:**


* `classNames` Vector of class names 
* `seed` Random seed for reproducibility 



**Returns:**

Vector of BGR colors 





        

<hr>



### function getPosePalette 

_Get the Ultralytics pose palette colors._ 
```C++
inline const std::vector< cv::Scalar > & yolos::drawing::getPosePalette () 
```





**Returns:**

Vector of BGR colors for pose visualization 





        

<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/core/drawing.hpp`

