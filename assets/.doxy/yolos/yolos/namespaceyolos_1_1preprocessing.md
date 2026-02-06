

# Namespace yolos::preprocessing



[**Namespace List**](namespaces.md) **>** [**yolos**](namespaceyolos.md) **>** [**preprocessing**](namespaceyolos_1_1preprocessing.md)




















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**InferenceBuffer**](structyolos_1_1preprocessing_1_1InferenceBuffer.md) <br>_Pre-allocated inference buffer to avoid per-frame allocations._  |






















## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**descaleCoordsBatch**](#function-descalecoordsbatch) (float \* coords, size\_t count, float scale, float padX, float padY) <br>_Fast coordinate descaling (batch operation)_  |
|  void | [**getLetterboxParams**](#function-getletterboxparams) (const cv::Size & originalShape, const cv::Size & letterboxShape, float & scale, float & padX, float & padY) <br>_Get letterbox padding and scale parameters._  |
|  void | [**getScalePad**](#function-getscalepad) (const cv::Size & originalSize, const cv::Size & letterboxSize, float & scale, float & padX, float & padY) <br>_Get scale and padding info from letterbox operation._  |
|  void | [**letterBox**](#function-letterbox) (const cv::Mat & image, cv::Mat & outImage, const cv::Size & newShape, const cv::Scalar & color=cv::Scalar(114, 114, 114), bool autoSize=true, bool scaleFill=false, bool scaleUp=true, int stride=32) <br>_Resize an image with letterboxing to maintain aspect ratio._  |
|  void | [**letterBoxCentered**](#function-letterboxcentered) (const cv::Mat & image, cv::Mat & outImage, const cv::Size & newShape=cv::Size(640, 640), bool autoSize=false, bool scaleFill=false, bool scaleUp=true, bool center=true, int stride=32, const cv::Scalar & paddingValue=cv::Scalar(114, 114, 114), int interpolation=cv::INTER\_LINEAR) <br>_Alternative letterbox with center option (matches Ultralytics)_  |
|  void | [**letterBoxToBlob**](#function-letterboxtoblob) (const cv::Mat & image, std::vector&lt; float &gt; & blob, const cv::Size & targetSize, cv::Size & actualSize, float padColor=114.0f) <br>_Fast letterbox with direct blob output (avoids intermediate copies)_  |
|  void | [**letterBoxToBlob**](#function-letterboxtoblob) (const cv::Mat & image, [**InferenceBuffer**](structyolos_1_1preprocessing_1_1InferenceBuffer.md) & buffer, const cv::Size & targetSize, cv::Size & actualSize, bool dynamicShape=false) <br>_Fast letterbox with buffer reuse._  |
|  [**BoundingBox**](structyolos_1_1BoundingBox.md) | [**scaleCoords**](#function-scalecoords) (const cv::Size & letterboxShape, const [**BoundingBox**](structyolos_1_1BoundingBox.md) & coords, const cv::Size & originalShape, bool clip=true) <br>_Scale detection coordinates from letterbox space back to original image size._  |
|  [**KeyPoint**](structyolos_1_1KeyPoint.md) | [**scaleKeypoint**](#function-scalekeypoint) (const cv::Size & letterboxShape, const [**KeyPoint**](structyolos_1_1KeyPoint.md) & keypoint, const cv::Size & originalShape, bool clip=true) <br>_Scale keypoint coordinates from letterbox space back to original image size._  |




























## Public Functions Documentation




### function descaleCoordsBatch 

_Fast coordinate descaling (batch operation)_ 
```C++
inline void yolos::preprocessing::descaleCoordsBatch (
    float * coords,
    size_t count,
    float scale,
    float padX,
    float padY
) 
```





**Parameters:**


* `coords` Array of x,y coordinates to descale 
* `count` Number of coordinate pairs 
* `scale` Letterbox scale 
* `padX` X padding 
* `padY` Y padding 




        

<hr>



### function getLetterboxParams 

_Get letterbox padding and scale parameters._ 
```C++
inline void yolos::preprocessing::getLetterboxParams (
    const cv::Size & originalShape,
    const cv::Size & letterboxShape,
    float & scale,
    float & padX,
    float & padY
) 
```





**Parameters:**


* `originalShape` Original image size 
* `letterboxShape` Letterboxed image size 
* `scale` Scale factor applied 
* `padX` Horizontal padding 
* `padY` Vertical padding 




        

<hr>



### function getScalePad 

_Get scale and padding info from letterbox operation._ 
```C++
inline void yolos::preprocessing::getScalePad (
    const cv::Size & originalSize,
    const cv::Size & letterboxSize,
    float & scale,
    float & padX,
    float & padY
) 
```





**Parameters:**


* `originalSize` Original image size 
* `letterboxSize` Letterboxed image size 
* `scale` Scale factor 
* `padX` X padding 
* `padY` Y padding 




        

<hr>



### function letterBox 

_Resize an image with letterboxing to maintain aspect ratio._ 
```C++
inline void yolos::preprocessing::letterBox (
    const cv::Mat & image,
    cv::Mat & outImage,
    const cv::Size & newShape,
    const cv::Scalar & color=cv::Scalar(114, 114, 114),
    bool autoSize=true,
    bool scaleFill=false,
    bool scaleUp=true,
    int stride=32
) 
```





**Parameters:**


* `image` Input image 
* `outImage` Output resized and padded image 
* `newShape` Desired output size 
* `color` Padding color (default is gray 114,114,114) 
* `autoSize` If true, use minimum rectangle to resize 
* `scaleFill` Whether to scale to fill without keeping aspect ratio 
* `scaleUp` Whether to allow scaling up of the image 
* `stride` Stride size for padding alignment 




        

<hr>



### function letterBoxCentered 

_Alternative letterbox with center option (matches Ultralytics)_ 
```C++
inline void yolos::preprocessing::letterBoxCentered (
    const cv::Mat & image,
    cv::Mat & outImage,
    const cv::Size & newShape=cv::Size(640, 640),
    bool autoSize=false,
    bool scaleFill=false,
    bool scaleUp=true,
    bool center=true,
    int stride=32,
    const cv::Scalar & paddingValue=cv::Scalar(114, 114, 114),
    int interpolation=cv::INTER_LINEAR
) 
```





**Parameters:**


* `image` Input image 
* `outImage` Output resized and padded image 
* `newShape` Desired output size (default 640x640) 
* `autoSize` If true, use minimum rectangle to resize 
* `scaleFill` Whether to scale to fill without keeping aspect ratio 
* `scaleUp` Whether to allow scaling up of the image 
* `center` If true, center the placed image 
* `stride` Stride of the model 
* `paddingValue` Padding value (default is 114) 
* `interpolation` Interpolation method 




        

<hr>



### function letterBoxToBlob 

_Fast letterbox with direct blob output (avoids intermediate copies)_ 
```C++
inline void yolos::preprocessing::letterBoxToBlob (
    const cv::Mat & image,
    std::vector< float > & blob,
    const cv::Size & targetSize,
    cv::Size & actualSize,
    float padColor=114.0f
) 
```





**Parameters:**


* `image` Input BGR image 
* `blob` Output CHW float blob (pre-allocated) 
* `targetSize` Target size for inference 
* `actualSize` Actual output size after letterboxing 
* `padColor` Padding color value (0-255, default 114) 




        

<hr>



### function letterBoxToBlob 

_Fast letterbox with buffer reuse._ 
```C++
inline void yolos::preprocessing::letterBoxToBlob (
    const cv::Mat & image,
    InferenceBuffer & buffer,
    const cv::Size & targetSize,
    cv::Size & actualSize,
    bool dynamicShape=false
) 
```





**Parameters:**


* `image` Input BGR image 
* `buffer` Pre-allocated inference buffer 
* `targetSize` Target size for inference 
* `actualSize` Actual output size 
* `dynamicShape` Whether to use dynamic shape 




        

<hr>



### function scaleCoords 

_Scale detection coordinates from letterbox space back to original image size._ 
```C++
inline BoundingBox yolos::preprocessing::scaleCoords (
    const cv::Size & letterboxShape,
    const BoundingBox & coords,
    const cv::Size & originalShape,
    bool clip=true
) 
```





**Parameters:**


* `letterboxShape` Shape of the letterboxed image used for inference 
* `coords` Bounding box in letterbox coordinates 
* `originalShape` Original image size before letterboxing 
* `clip` Whether to clip coordinates to image boundaries 



**Returns:**

Scaled bounding box in original image coordinates 





        

<hr>



### function scaleKeypoint 

_Scale keypoint coordinates from letterbox space back to original image size._ 
```C++
inline KeyPoint yolos::preprocessing::scaleKeypoint (
    const cv::Size & letterboxShape,
    const KeyPoint & keypoint,
    const cv::Size & originalShape,
    bool clip=true
) 
```





**Parameters:**


* `letterboxShape` Shape of the letterboxed image 
* `keypoint` Keypoint in letterbox coordinates 
* `originalShape` Original image size 
* `clip` Whether to clip coordinates to image boundaries 



**Returns:**

Scaled keypoint in original image coordinates 





        

<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/core/preprocessing.hpp`

