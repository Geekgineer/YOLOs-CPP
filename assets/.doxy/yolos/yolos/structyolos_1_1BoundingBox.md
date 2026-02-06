

# Struct yolos::BoundingBox



[**ClassList**](annotated.md) **>** [**yolos**](namespaceyolos.md) **>** [**BoundingBox**](structyolos_1_1BoundingBox.md)





* `#include <types.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  int | [**height**](#variable-height)   = `{0}`<br>_Height of the bounding box._  |
|  int | [**width**](#variable-width)   = `{0}`<br>_Width of the bounding box._  |
|  int | [**x**](#variable-x)   = `{0}`<br>_X-coordinate of top-left corner._  |
|  int | [**y**](#variable-y)   = `{0}`<br>_Y-coordinate of top-left corner._  |
















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**BoundingBox**](#function-boundingbox-12) () = default<br> |
|   | [**BoundingBox**](#function-boundingbox-22) (int x\_, int y\_, int width\_, int height\_) <br> |
|  float | [**area**](#function-area) () noexcept const<br>_Compute area of the bounding box._  |
|  [**BoundingBox**](structyolos_1_1BoundingBox.md) | [**intersect**](#function-intersect) (const [**BoundingBox**](structyolos_1_1BoundingBox.md) & other) noexcept const<br>_Compute intersection with another bounding box._  |
|  float | [**iou**](#function-iou) (const [**BoundingBox**](structyolos_1_1BoundingBox.md) & other) noexcept const<br>_Compute IoU (Intersection over Union) with another bounding box._  |




























## Public Attributes Documentation




### variable height 

_Height of the bounding box._ 
```C++
int yolos::BoundingBox::height;
```




<hr>



### variable width 

_Width of the bounding box._ 
```C++
int yolos::BoundingBox::width;
```




<hr>



### variable x 

_X-coordinate of top-left corner._ 
```C++
int yolos::BoundingBox::x;
```




<hr>



### variable y 

_Y-coordinate of top-left corner._ 
```C++
int yolos::BoundingBox::y;
```




<hr>
## Public Functions Documentation




### function BoundingBox [1/2]

```C++
yolos::BoundingBox::BoundingBox () = default
```




<hr>



### function BoundingBox [2/2]

```C++
inline yolos::BoundingBox::BoundingBox (
    int x_,
    int y_,
    int width_,
    int height_
) 
```




<hr>



### function area 

_Compute area of the bounding box._ 
```C++
inline float yolos::BoundingBox::area () noexcept const
```




<hr>



### function intersect 

_Compute intersection with another bounding box._ 
```C++
inline BoundingBox yolos::BoundingBox::intersect (
    const BoundingBox & other
) noexcept const
```




<hr>



### function iou 

_Compute IoU (Intersection over Union) with another bounding box._ 
```C++
inline float yolos::BoundingBox::iou (
    const BoundingBox & other
) noexcept const
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/core/types.hpp`

