

# Struct yolos::OrientedBoundingBox



[**ClassList**](annotated.md) **>** [**yolos**](namespaceyolos.md) **>** [**OrientedBoundingBox**](structyolos_1_1OrientedBoundingBox.md)





* `#include <types.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  float | [**angle**](#variable-angle)   = `{0.0f}`<br>_Rotation angle in radians._  |
|  float | [**height**](#variable-height)   = `{0.0f}`<br>_Height of the box._  |
|  float | [**width**](#variable-width)   = `{0.0f}`<br>_Width of the box._  |
|  float | [**x**](#variable-x)   = `{0.0f}`<br>_X-coordinate of center._  |
|  float | [**y**](#variable-y)   = `{0.0f}`<br>_Y-coordinate of center._  |
















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**OrientedBoundingBox**](#function-orientedboundingbox-12) () = default<br> |
|   | [**OrientedBoundingBox**](#function-orientedboundingbox-22) (float x\_, float y\_, float width\_, float height\_, float angle\_) <br> |
|  float | [**area**](#function-area) () noexcept const<br>_Compute area of the oriented bounding box._  |




























## Public Attributes Documentation




### variable angle 

_Rotation angle in radians._ 
```C++
float yolos::OrientedBoundingBox::angle;
```




<hr>



### variable height 

_Height of the box._ 
```C++
float yolos::OrientedBoundingBox::height;
```




<hr>



### variable width 

_Width of the box._ 
```C++
float yolos::OrientedBoundingBox::width;
```




<hr>



### variable x 

_X-coordinate of center._ 
```C++
float yolos::OrientedBoundingBox::x;
```




<hr>



### variable y 

_Y-coordinate of center._ 
```C++
float yolos::OrientedBoundingBox::y;
```




<hr>
## Public Functions Documentation




### function OrientedBoundingBox [1/2]

```C++
yolos::OrientedBoundingBox::OrientedBoundingBox () = default
```




<hr>



### function OrientedBoundingBox [2/2]

```C++
inline yolos::OrientedBoundingBox::OrientedBoundingBox (
    float x_,
    float y_,
    float width_,
    float height_,
    float angle_
) 
```




<hr>



### function area 

_Compute area of the oriented bounding box._ 
```C++
inline float yolos::OrientedBoundingBox::area () noexcept const
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/core/types.hpp`

