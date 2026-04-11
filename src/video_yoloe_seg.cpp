/**
 * @file video_yoloe_seg.cpp
 * @brief Open-vocabulary **video** segmentation using YOLOE (video in → MP4 out).
 *
 * For a single image use `image_yoloe_seg` (image → image).
 *
 * Export the model with:
 *   python scripts/export_yoloe_onnx.py
 *
 * Usage:
 *   video_yoloe_seg [video.mp4] [output.mp4] [model.onnx] [labels_or_classes] [use_gpu]
 *
 * Arguments:
 *   video           Input video path (default: data/Transmission.mp4)
 *   output          Output MP4 path (default: data/output_yoloe_seg.mp4)
 *   model.onnx      ONNX model path  (default: models/yoloe-26s-seg-text.onnx)
 *   labels_or_cls   Labels file (.txt/.names) OR comma-separated class list
 *   use_gpu         1 = GPU, 0 = CPU (default: 1)
 *
 * Examples:
 *   ./video_yoloe_seg data/street.mp4 out.mp4 models/yoloe-26s-seg-text.onnx "person,car,bus" 1
 *   ./video_yoloe_seg data/street.mp4 out.mp4 models/yoloe-26s-seg-pf.onnx models/yoloe_pf.names 1
 */

#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <algorithm>
#include <cctype>

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

/// Reject still-image inputs — use image_yoloe_seg instead.
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
    std::string videoPath   = "data/Transmission.mp4";
    std::string outputPath  = "data/output_yoloe_seg.mp4";
    std::string modelPath   = "models/yoloe-26s-seg-text.onnx";
    std::string classArg    = "person,car,bus,bicycle,motorcycle,truck";
    bool        useGPU      = true;

    if (argc > 1) videoPath  = argv[1];
    if (argc > 2) outputPath = argv[2];
    if (argc > 3) modelPath  = argv[3];
    if (argc > 4) classArg   = argv[4];
    if (argc > 5) useGPU     = std::string(argv[5]) != "0";

    if (isRasterImagePath(videoPath)) {
        std::cerr << "Error: input looks like a still image. Use image_yoloe_seg for image → image.\n";
        return 1;
    }

    std::unique_ptr<YOLOESegDetector> detector;

    try {
        const bool fromFile = looksLikeFile(classArg);
        if (fromFile) {
            std::cout << "[YOLOE] Prompt-free / file mode: " << classArg << std::endl;
            detector = createYOLOESegDetector(modelPath, classArg, useGPU, /*agnosticNms=*/true);
        } else {
            std::vector<std::string> classNames = splitClasses(classArg);
            std::cout << "[YOLOE] Text-prompt mode (" << classNames.size() << " classes): ";
            for (size_t i = 0; i < classNames.size(); ++i) {
                std::cout << classNames[i];
                if (i + 1 < classNames.size()) std::cout << ", ";
            }
            std::cout << std::endl;
            detector = createYOLOESegDetector(modelPath, classNames, useGPU, /*agnosticNms=*/true);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "[YOLOE] Model: " << modelPath << std::endl;
    std::cout << "[YOLOE] Device: " << (useGPU ? "GPU" : "CPU") << std::endl;

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video: " << videoPath << std::endl;
        return -1;
    }

    const int    frameW      = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int    frameH      = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double       fpsIn       = cap.get(cv::CAP_PROP_FPS);
    if (fpsIn <= 0) fpsIn    = 25.0;
    const int    totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "[Video] " << videoPath << "  " << frameW << "x" << frameH << " @ " << fpsIn << " FPS  ("
              << totalFrames << " frames)" << std::endl;
    std::cout << "[Video] Output: " << outputPath << std::endl;

    cv::VideoWriter writer;
    const int outFourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    writer.open(outputPath, outFourcc, fpsIn, cv::Size(frameW, frameH), true);
    if (!writer.isOpened()) {
        std::cerr << "Error: Cannot open output video: " << outputPath << std::endl;
        return -1;
    }

    const auto& classNames  = detector->getClassNames();
    const auto& classColors = detector->getClassColors();

    cv::Mat frame;
    int     frameCount = 0;
    long    totalMs    = 0;

    std::cout << "\nProcessing..." << std::endl;

    while (cap.read(frame)) {
        if (frame.empty()) continue;

        auto t0 = std::chrono::steady_clock::now();
        std::vector<yolos::seg::Segmentation> segs = detector->segment(frame, 0.35f, 0.45f);
        auto t1 = std::chrono::steady_clock::now();
        totalMs += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        detector->drawSegmentations(frame, segs, 0.45f);

        for (const auto& s : segs) {
            if (s.classId < 0 || static_cast<size_t>(s.classId) >= classNames.size()) continue;
            const std::string label =
                classNames[s.classId] + " " + std::to_string(static_cast<int>(s.conf * 100)) + "%";
            const cv::Scalar& color = classColors[s.classId % classColors.size()];
            drawLabel(frame, s.box.x, s.box.y, label, color);
        }

        {
            const double avgFps = (totalMs > 0) ? (frameCount + 1) * 1000.0 / totalMs : 0.0;
            char fpsText[64];
            std::snprintf(fpsText, sizeof(fpsText), "YOLOE  %.1f FPS  %zu obj", avgFps, segs.size());
            cv::putText(frame, fpsText, cv::Point(12, 32), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0),
                        3, cv::LINE_AA);
            cv::putText(frame, fpsText, cv::Point(12, 32), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                        cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }

        writer.write(frame);
        ++frameCount;

        if (frameCount % 100 == 0) {
            const double avgFps = (totalMs > 0) ? frameCount * 1000.0 / totalMs : 0.0;
            std::cout << "  Frame " << frameCount << "/" << totalFrames << "  avg " << std::fixed
                      << std::setprecision(1) << avgFps << " FPS" << std::endl;
        }
    }

    cap.release();
    writer.release();

    const double avgFps = (totalMs > 0) ? frameCount * 1000.0 / totalMs : 0.0;
    std::cout << "\nDone. " << frameCount << " frames  " << std::fixed << std::setprecision(1) << avgFps
              << " FPS avg"
              << "\nSaved: " << outputPath << std::endl;
    return 0;
}
