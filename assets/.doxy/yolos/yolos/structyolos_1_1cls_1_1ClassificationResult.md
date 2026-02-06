

# Struct yolos::cls::ClassificationResult



[**ClassList**](annotated.md) **>** [**yolos**](namespaceyolos.md) **>** [**cls**](namespaceyolos_1_1cls.md) **>** [**ClassificationResult**](structyolos_1_1cls_1_1ClassificationResult.md)



_Classification result containing class ID, confidence, and class name._ 

* `#include <classification.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  int | [**classId**](#variable-classid)   = `{-1}`<br>_Predicted class ID._  |
|  std::string | [**className**](#variable-classname)   = `{}`<br>_Human-readable class name._  |
|  float | [**confidence**](#variable-confidence)   = `{0.0f}`<br>_Confidence score._  |
















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**ClassificationResult**](#function-classificationresult-12) () = default<br> |
|   | [**ClassificationResult**](#function-classificationresult-22) (int id, float conf, std::string name) <br> |




























## Public Attributes Documentation




### variable classId 

_Predicted class ID._ 
```C++
int yolos::cls::ClassificationResult::classId;
```




<hr>



### variable className 

_Human-readable class name._ 
```C++
std::string yolos::cls::ClassificationResult::className;
```




<hr>



### variable confidence 

_Confidence score._ 
```C++
float yolos::cls::ClassificationResult::confidence;
```




<hr>
## Public Functions Documentation




### function ClassificationResult [1/2]

```C++
yolos::cls::ClassificationResult::ClassificationResult () = default
```




<hr>



### function ClassificationResult [2/2]

```C++
inline yolos::cls::ClassificationResult::ClassificationResult (
    int id,
    float conf,
    std::string name
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/tasks/classification.hpp`

