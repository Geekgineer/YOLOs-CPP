/**
 * @file video_yolo26_seg_pose.cpp
 * @brief One video: segmentation + pose overlaid on the same frame with class labels.
 * Uses YOLO26m-seg and YOLO26m-pose. Output matches original resolution and quality.
 *
 * Usage:
 *   video_yolo26_seg_pose [video_path] [output_path] [models_dir]
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include "yolos/tasks/segmentation.hpp"
#include "yolos/tasks/pose.hpp"

using namespace yolos::seg;
using namespace yolos::pose;

static void drawLabel(cv::Mat& image, int boxX, int boxY, const std::string& label, const cv::Scalar& color) {
    if (label.empty()) return;
    const int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    const double fontScale = std::max(0.5, std::min(image.rows, image.cols) * 0.0009);
    const int thickness = std::max(1, static_cast<int>(fontScale * 2));
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);
    const int pad = 4;
    int lx = std::max(0, std::min(boxX, image.cols - textSize.width - pad * 2));
    int ly = std::max(textSize.height + pad * 2, boxY);
    cv::Point tl(lx, ly - textSize.height - pad * 2);
    cv::Point br(lx + textSize.width + pad * 2, ly);
    cv::rectangle(image, tl, br, color, cv::FILLED);
    cv::putText(image, label, cv::Point(lx + pad, ly - pad),
                fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
}

int main(int argc, char* argv[]) {
    std::string videoPath = "data/Transmission.mp4";
    std::string outputPath = "data/Transmission_yolo26_seg_pose.mp4";
    std::string modelsDir = "models";

    if (argc > 1) videoPath = argv[1];
    if (argc > 2) outputPath = argv[2];
    if (argc > 3) modelsDir = argv[3];

    std::string cocoPath = modelsDir + "/coco.names";
    std::string segPath = modelsDir + "/yolo26m-seg.onnx";
    std::string posePath = modelsDir + "/yolo26m-pose.onnx";

    std::cout << "Input:  " << videoPath << std::endl;
    std::cout << "Output: " << outputPath << std::endl;

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video: " << videoPath << std::endl;
        return -1;
    }

    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fpsIn = cap.get(cv::CAP_PROP_FPS);
    if (fpsIn <= 0) fpsIn = 25.0;
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "Video: " << frameWidth << "x" << frameHeight << " @ " << fpsIn << " FPS, " << totalFrames << " frames" << std::endl;

    auto exists = [](const std::string& path) { return std::ifstream(path).good(); };
    if (!exists(segPath) || !exists(posePath)) {
        std::cerr << "Missing models in " << modelsDir << "/: yolo26m-seg.onnx, yolo26m-pose.onnx" << std::endl;
        return -1;
    }

    bool useGPU = true;
    std::cout << "Loading YOLO26m-seg and YOLO26m-pose..." << std::endl;
    YOLOSegDetector segDet(segPath, cocoPath, useGPU);
    YOLOPoseDetector poseDet(posePath, cocoPath, useGPU);
    std::cout << "Models loaded." << std::endl;

    cv::VideoWriter writer;
    if (fourcc != -1) {
        writer.open(outputPath, fourcc, fpsIn, cv::Size(frameWidth, frameHeight), true);
    }
    if (!writer.isOpened()) {
        fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(outputPath, fourcc, fpsIn, cv::Size(frameWidth, frameHeight), true);
    }
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output: " << outputPath << std::endl;
        return -1;
    }

    const auto& poseClassNames = poseDet.getClassNames();
    const cv::Scalar poseColor(0, 255, 0);

    cv::Mat frame;
    int frameCount = 0;

    while (cap.read(frame)) {
        if (frame.empty()) continue;

        std::vector<Segmentation> segs = segDet.segment(frame, 0.4f, 0.45f);
        segDet.drawSegmentations(frame, segs, 0.5f);

        std::vector<PoseResult> poses = poseDet.detect(frame, 0.4f, 0.5f);
        poseDet.drawPoses(frame, poses, 4, 0.5f, 2);
        for (const auto& p : poses) {
            std::string label;
            if (p.classId >= 0 && static_cast<size_t>(p.classId) < poseClassNames.size()) {
                label = poseClassNames[p.classId] + " " + std::to_string(static_cast<int>(p.conf * 100)) + "%";
            } else {
                label = "person " + std::to_string(static_cast<int>(p.conf * 100)) + "%";
            }
            drawLabel(frame, p.box.x, p.box.y, label, poseColor);
        }

        writer.write(frame);
        frameCount++;
        if (frameCount % 100 == 0) {
            std::cout << "Processed " << frameCount << "/" << totalFrames << " frames" << std::endl;
        }
    }

    cap.release();
    writer.release();
    std::cout << "Done. Frames: " << frameCount << " -> " << outputPath << std::endl;
    return 0;
}
