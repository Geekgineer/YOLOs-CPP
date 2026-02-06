

# File detection.hpp



[**FileList**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**yolos**](dir_0663062e3f7bb1f439b575391d32cc63.md) **>** [**tasks**](dir_2bef6d8bc01c1b0d186aba6b16825c1e.md) **>** [**detection.hpp**](detection_8hpp.md)

[Go to the source code of this file](detection_8hpp_source.md)



* `#include <opencv2/opencv.hpp>`
* `#include <vector>`
* `#include <string>`
* `#include <memory>`
* `#include <cfloat>`
* `#include "yolos/core/types.hpp"`
* `#include "yolos/core/version.hpp"`
* `#include "yolos/core/utils.hpp"`
* `#include "yolos/core/preprocessing.hpp"`
* `#include "yolos/core/nms.hpp"`
* `#include "yolos/core/drawing.hpp"`
* `#include "yolos/core/session_base.hpp"`













## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**yolos**](namespaceyolos.md) <br> |
| namespace | [**det**](namespaceyolos_1_1det.md) <br> |


## Classes

| Type | Name |
| ---: | :--- |
| struct | [**Detection**](structyolos_1_1det_1_1Detection.md) <br>[_**Detection**_](structyolos_1_1det_1_1Detection.md) _result containing bounding box, confidence, and class ID._ |
| class | [**YOLO26Detector**](classyolos_1_1det_1_1YOLO26Detector.md) <br>_YOLOv26 detector (forces V26 end-to-end postprocessing)_  |
| class | [**YOLODetector**](classyolos_1_1det_1_1YOLODetector.md) <br>_Base YOLO detector with runtime version auto-detection._  |
| class | [**YOLONASDetector**](classyolos_1_1det_1_1YOLONASDetector.md) <br>_YOLO-NAS detector (forces NAS postprocessing)_  |
| class | [**YOLOv10Detector**](classyolos_1_1det_1_1YOLOv10Detector.md) <br>_YOLOv10 detector (forces V10 end-to-end postprocessing)_  |
| class | [**YOLOv11Detector**](classyolos_1_1det_1_1YOLOv11Detector.md) <br>_YOLOv11 detector (forces standard postprocessing)_  |
| class | [**YOLOv7Detector**](classyolos_1_1det_1_1YOLOv7Detector.md) <br>_YOLOv7 detector (forces V7 postprocessing)_  |
| class | [**YOLOv8Detector**](classyolos_1_1det_1_1YOLOv8Detector.md) <br>_YOLOv8 detector (forces standard postprocessing)_  |



















































------------------------------
The documentation for this class was generated from the following file `include/yolos/tasks/detection.hpp`

