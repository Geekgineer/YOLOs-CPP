#include "evaluation.hpp"
#include <iostream>

int main(int argc, char** argv) {
    // Default paths
    bool useGPU = true;
    std::string modelPath = "../../models/yolo11n.onnx";
    std::string labels = "../../models/coco.names";
    std::string images = "../val2017";
    std::string gts = "../labels_val2017";  // YOLO txt format

    // Parse command-line arguments
    if (argc > 1) useGPU = std::string(argv[1]) == "1" || std::string(argv[1]) == "true";
    if (argc > 2) modelPath = argv[2];      // second arg: model path
    if (argc > 3) labels = argv[3];         // third arg: labels file
    if (argc > 4) images = argv[4];         // fourth arg: image folder
    if (argc > 5) gts = argv[5];            // fifth arg: labels folder

    std::cout << "Use GPU: " << (useGPU ? "Yes" : "No") << "\n";
    std::cout << "Using model: " << modelPath << "\n";
    std::cout << "Using labels: " << labels << "\n";
    std::cout << "Image folder: " << images << "\n";
    std::cout << "GT folder: " << gts << "\n";

    YOLODetector detector(modelPath, labels, useGPU);

    EvalConfig cfg;
    cfg.imgSize = 640;          // must match your model
    cfg.confThreshold = 0.001;  // recommended
    cfg.nmsThreshold = 0.7;

    Evaluator eval(detector, cfg);
    eval.evaluate(images, gts);

    return 0;
}
