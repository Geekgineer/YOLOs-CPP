/**
 * @file image_yoloe_seg.cpp
 * @brief YOLOE instance segmentation on a single image (image in → image out).
 *
 * For video files use `video_yoloe_seg` (MP4 → MP4).
 *
 * Usage:
 *   image_yoloe_seg [input.jpg] [output.jpg] [model.onnx] [labels_or_classes] [use_gpu]
 */

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "yolos/tasks/yoloe.hpp"

using namespace yolos::yoloe;

static std::vector<std::string> splitClasses(const std::string& s) {
    std::vector<std::string> result;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) result.push_back(token);
    }
    return result;
}

static std::string toLowerAscii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static bool isRasterImagePath(const std::string& path) {
    const std::string lower = toLowerAscii(path);
    for (const char* ext : {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}) {
        if (lower.size() >= strlen(ext) &&
            lower.compare(lower.size() - strlen(ext), strlen(ext), ext) == 0) {
            return true;
        }
    }
    return false;
}

static bool looksLikeFile(const std::string& s) {
    for (const char* ext : {".txt", ".names", ".csv"}) {
        if (s.size() > strlen(ext) &&
            s.compare(s.size() - strlen(ext), strlen(ext), ext) == 0) {
            return true;
        }
    }
    return s.find('/') != std::string::npos || s.find('\\') != std::string::npos;
}

static void drawLabel(cv::Mat& image, int bx, int by, const std::string& label, const cv::Scalar& color) {
    if (label.empty()) return;
    const int fontFace  = cv::FONT_HERSHEY_SIMPLEX;
    const double scale  = std::max(0.45, std::min(image.rows, image.cols) * 0.0009);
    const int thickness = std::max(1, static_cast<int>(scale * 2));
    int baseline = 0;
    cv::Size ts = cv::getTextSize(label, fontFace, scale, thickness, &baseline);
    const int pad = 4;
    int lx = std::max(0, std::min(bx, image.cols - ts.width - pad * 2));
    int ly = std::max(ts.height + pad * 2, by);
    cv::rectangle(image, cv::Point(lx, ly - ts.height - pad * 2), cv::Point(lx + ts.width + pad * 2, ly),
                  color, cv::FILLED);
    cv::putText(image, label, cv::Point(lx + pad, ly - pad), fontFace, scale, cv::Scalar(255, 255, 255),
                thickness, cv::LINE_AA);
}

int main(int argc, char* argv[]) {
    std::string inputPath  = "data/dog.jpg";
    std::string outputPath = "data/output_yoloe_seg.jpg";
    std::string modelPath  = "models/yoloe-26s-seg-text.onnx";
    std::string classArg   = "person,car,bus,bicycle,motorcycle,truck";
    bool        useGPU     = true;

    if (argc > 1) inputPath  = argv[1];
    if (argc > 2) outputPath = argv[2];
    if (argc > 3) modelPath  = argv[3];
    if (argc > 4) classArg   = argv[4];
    if (argc > 5) useGPU     = std::string(argv[5]) != "0";

    if (!isRasterImagePath(inputPath)) {
        std::cerr << "Error: input must be a raster image (.jpg, .png, ...). "
                     "For video → video use: video_yoloe_seg\n";
        return 1;
    }
    if (!isRasterImagePath(outputPath)) {
        std::cerr << "Error: output must be a raster image path (.jpg, .png, ...).\n";
        return 1;
    }

    std::unique_ptr<YOLOESegDetector> detector;
    try {
        const bool fromFile = looksLikeFile(classArg);
        if (fromFile) {
            std::cout << "[YOLOE] Prompt-free / file mode: " << classArg << std::endl;
            detector = createYOLOESegDetector(modelPath, classArg, useGPU, true);
        } else {
            std::vector<std::string> classNames = splitClasses(classArg);
            std::cout << "[YOLOE] Text-prompt mode (" << classNames.size() << " classes): ";
            for (size_t i = 0; i < classNames.size(); ++i) {
                std::cout << classNames[i];
                if (i + 1 < classNames.size()) std::cout << ", ";
            }
            std::cout << std::endl;
            detector = createYOLOESegDetector(modelPath, classNames, useGPU, true);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "[YOLOE] Model: " << modelPath << std::endl;
    std::cout << "[YOLOE] Device: " << (useGPU ? "GPU" : "CPU") << std::endl;

    cv::Mat frame = cv::imread(inputPath);
    if (frame.empty()) {
        std::cerr << "Error: Cannot read image: " << inputPath << std::endl;
        return 1;
    }

    const auto& classNames  = detector->getClassNames();
    const auto& classColors = detector->getClassColors();

    std::cout << "[Image] " << inputPath << "  " << frame.cols << "x" << frame.rows << std::endl;
    std::cout << "[Image] Output: " << outputPath << std::endl;

    auto t0 = std::chrono::steady_clock::now();
    std::vector<yolos::seg::Segmentation> segs = detector->segment(frame, 0.35f, 0.45f);
    auto t1 = std::chrono::steady_clock::now();
    const long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    detector->drawSegmentations(frame, segs, 0.45f);

    for (const auto& s : segs) {
        if (s.classId < 0 || static_cast<size_t>(s.classId) >= classNames.size()) continue;
        const std::string label =
            classNames[s.classId] + " " + std::to_string(static_cast<int>(s.conf * 100)) + "%";
        const cv::Scalar& color = classColors[s.classId % classColors.size()];
        drawLabel(frame, s.box.x, s.box.y, label, color);
    }

    char info[96];
    std::snprintf(info, sizeof(info), "YOLOE  %zu obj  %ld ms", segs.size(), ms);
    cv::putText(frame, info, cv::Point(12, 32), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0), 3,
                cv::LINE_AA);
    cv::putText(frame, info, cv::Point(12, 32), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 1,
                cv::LINE_AA);

    if (!cv::imwrite(outputPath, frame)) {
        std::cerr << "Error: Cannot write image: " << outputPath << std::endl;
        return 1;
    }

    std::cout << "Saved: " << outputPath << std::endl;
    return 0;
}
