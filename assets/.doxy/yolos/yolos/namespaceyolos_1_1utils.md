

# Namespace yolos::utils



[**Namespace List**](namespaces.md) **>** [**yolos**](namespaceyolos.md) **>** [**utils**](namespaceyolos_1_1utils.md)










































## Public Functions

| Type | Name |
| ---: | :--- |
|  std::enable\_if&lt; std::is\_arithmetic&lt; T &gt;::value, T &gt;::type | [**clamp**](#function-clamp) (const T & value, const T & low, const T & high) <br>_Clamp a value to a specified range [low, high]._  |
|  std::vector&lt; std::string &gt; | [**getClassNames**](#function-getclassnames) (const std::string & path) <br>_Load class names from a file (one class name per line)_  |
|  float | [**sigmoid**](#function-sigmoid) (float x) <br>_Apply sigmoid activation: 1 / (1 + exp(-x))_  |
|  void | [**sigmoidInplace**](#function-sigmoidinplace) (std::vector&lt; float &gt; & values) <br>_Apply sigmoid activation to a vector in-place._  |
|  size\_t | [**vectorProduct**](#function-vectorproduct) (const std::vector&lt; int64\_t &gt; & shape) <br>_Compute the product of elements in a vector._  |




























## Public Functions Documentation




### function clamp 

_Clamp a value to a specified range [low, high]._ 
```C++
template<typename T>
inline std::enable_if< std::is_arithmetic< T >::value, T >::type yolos::utils::clamp (
    const T & value,
    const T & low,
    const T & high
) 
```





**Template parameters:**


* `T` Arithmetic type (int, float, etc.) 



**Parameters:**


* `value` The value to clamp 
* `low` Lower bound 
* `high` Upper bound 



**Returns:**

Clamped value 





        

<hr>



### function getClassNames 

_Load class names from a file (one class name per line)_ 
```C++
inline std::vector< std::string > yolos::utils::getClassNames (
    const std::string & path
) 
```





**Parameters:**


* `path` Path to the class names file 



**Returns:**

Vector of class names 





        

<hr>



### function sigmoid 

_Apply sigmoid activation: 1 / (1 + exp(-x))_ 
```C++
inline float yolos::utils::sigmoid (
    float x
) 
```





**Parameters:**


* `x` Input value 



**Returns:**

Sigmoid of x 





        

<hr>



### function sigmoidInplace 

_Apply sigmoid activation to a vector in-place._ 
```C++
inline void yolos::utils::sigmoidInplace (
    std::vector< float > & values
) 
```





**Parameters:**


* `values` Vector of values to transform 




        

<hr>



### function vectorProduct 

_Compute the product of elements in a vector._ 
```C++
inline size_t yolos::utils::vectorProduct (
    const std::vector< int64_t > & shape
) 
```





**Parameters:**


* `shape` Vector of dimensions 



**Returns:**

Product of all elements 





        

<hr>

------------------------------
The documentation for this class was generated from the following file `include/yolos/core/utils.hpp`

