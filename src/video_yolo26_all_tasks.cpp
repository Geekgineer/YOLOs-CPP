/**
 * @file video_yolo26_all_tasks.cpp
 * @brief Process a video with all YOLO26 nano tasks and write one composite output.
 *
 * Runs detection, segmentation, pose, OBB, and classification (YOLO26n models)
 * on each frame and composites results into a single 2x3 grid video.
 *
 * Usage:
 *   video_yolo26_all_tasks [video_path] [output_path] [models_dir]
 * Defaults: data/Transmission.mp4, data/Transmission_yolo26_all.mp4, models/
 *
 * Required ONNX models in models_dir (or use test models):
 *   yolo26n.onnx, yolo26n-seg.onnx, yolo26n-pose.onnx, yolo26n-obb.onnx, yolo26n-cls.onnx
 * Labels: coco.names (det/seg/pose), Dota.names (obb), ImageNet.names (cls)
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <string>
#include <fstream>
#include "yolos/tasks/detection.hpp"
#include "yolos/tasks/segmentation.hpp"
#include "yolos/tasks/pose.hpp"
#include "yolos/tasks/obb.hpp"
#include "yolos/tasks/classification.hpp"

using namespace yolos::det;
using namespace yolos::seg;
using namespace yolos::pose;
using namespace yolos::obb;
using namespace yolos::cls;

static const int GRID_COLS = 3;
static const int GRID_ROWS = 2;
static const char* TASK_LABELS[] = { "YOLO26n Detection", "YOLO26n Segmentation", "YOLO26n Pose",
                                     "YOLO26n OBB", "YOLO26n Classification", "FPS" };

int main(int argc, char* argv[]) {
    std::string videoPath = "data/Transmission.mp4";
    std::string outputPath = "data/Transmission_yolo26_all.mp4";
    std::string modelsDir = "models";

    if (argc > 1) videoPath = argv[1];
    if (argc > 2) outputPath = argv[2];
    if (argc > 3) modelsDir = argv[3];

    std::string cocoPath = modelsDir + "/coco.names";
    std::string dotaPath = modelsDir + "/Dota.names";
    std::string imagenetPath = modelsDir + "/ImageNet.names";
    std::string altImagenet = modelsDir + "/imagenet_classes.txt";
    if (std::ifstream(imagenetPath).good()) { /* use ImageNet.names */ }
    else if (std::ifstream(altImagenet).good()) imagenetPath = altImagenet;

    std::string detPath    = modelsDir + "/yolo26n.onnx";
    std::string segPath    = modelsDir + "/yolo26n-seg.onnx";
    std::string posePath   = modelsDir + "/yolo26n-pose.onnx";
    std::string obbPath    = modelsDir + "/yolo26n-obb.onnx";
    std::string clsPath    = modelsDir + "/yolo26n-cls.onnx";

    std::cout << "Input video:  " << videoPath << std::endl;
    std::cout << "Output video: " << outputPath << std::endl;
    std::cout << "Models dir:   " << modelsDir << std::endl;

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video: " << videoPath << std::endl;
        return -1;
    }

    int frameWidth  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fpsIn   = cap.get(cv::CAP_PROP_FPS);
    if (fpsIn <= 0) fpsIn = 30.0;
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "Video: " << frameWidth << "x" << frameHeight << " @ " << fpsIn << " FPS, " << totalFrames << " frames" << std::endl;

    const int cellW = frameWidth  / GRID_COLS;
    const int cellH = frameHeight / GRID_ROWS;
    const cv::Size cellSize(cellW, cellH);

    bool useGPU = true;

    auto exists = [](const std::string& path) { return std::ifstream(path).good(); };
    if (!exists(detPath) || !exists(segPath) || !exists(posePath) || !exists(obbPath) || !exists(clsPath)) {
        std::cerr << "Missing one or more ONNX models in " << modelsDir << "/. Required:\n"
                  << "  yolo26n.onnx, yolo26n-seg.onnx, yolo26n-pose.onnx, yolo26n-obb.onnx, yolo26n-cls.onnx\n"
                  << "Run: python3 scripts/download_yolo26n_onnx.py " << modelsDir << std::endl;
        return -1;
    }
    std::cout << "Loading YOLO26n models..." << std::endl;
    auto det = createDetector(detPath, cocoPath, yolos::YOLOVersion::V26, useGPU);
    YOLOSegDetector segDet(segPath, cocoPath, useGPU);
    YOLOPoseDetector poseDet(posePath, cocoPath, useGPU);
    YOLOOBBDetector obbDet(obbPath, dotaPath, useGPU);
    YOLO26Classifier clsDet(clsPath, imagenetPath, useGPU);
    std::cout << "All models loaded." << std::endl;

    cv::VideoWriter writer;
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    if (!writer.open(outputPath, fourcc, fpsIn, cv::Size(frameWidth, frameHeight), true)) {
        std::cerr << "Error: Could not open output: " << outputPath << std::endl;
        return -1;
    }

    cv::Mat frame;
    int frameCount = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    while (cap.read(frame)) {
        if (frame.empty()) continue;

        cv::Mat grid = cv::Mat::zeros(frameHeight, frameWidth, frame.type());
        grid.setTo(cv::Scalar(40, 40, 40));

        auto placePanel = [&](const cv::Mat& panel, int col, int row, const std::string& title) {
            cv::Mat resized;
            cv::resize(panel, resized, cellSize);
            int x = col * cellW;
            int y = row * cellH;
            cv::Rect roi(x, y, cellW, cellH);
            resized.copyTo(grid(roi));
            cv::rectangle(grid, roi, cv::Scalar(200, 200, 200), 1);
            cv::putText(grid, title, cv::Point(x + 4, y + 22),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        };

        cv::Mat fDet = frame.clone();
        std::vector<Detection> dets = det->detect(fDet);
        det->drawDetectionsWithMask(fDet, dets);
        placePanel(fDet, 0, 0, TASK_LABELS[0]);

        cv::Mat fSeg = frame.clone();
        std::vector<Segmentation> segs = segDet.segment(fSeg);
        segDet.drawSegmentations(fSeg, segs);
        placePanel(fSeg, 1, 0, TASK_LABELS[1]);

        cv::Mat fPose = frame.clone();
        std::vector<PoseResult> poses = poseDet.detect(fPose, 0.4f, 0.5f);
        poseDet.drawPoses(fPose, poses);
        placePanel(fPose, 2, 0, TASK_LABELS[2]);

        cv::Mat fObb = frame.clone();
        std::vector<OBBResult> obbs = obbDet.detect(fObb);
        obbDet.drawDetections(fObb, obbs);
        placePanel(fObb, 0, 1, TASK_LABELS[3]);

        cv::Mat fCls = frame.clone();
        ClassificationResult clsRes = clsDet.classify(fCls);
        clsDet.drawResult(fCls, clsRes, cv::Point(10, 30));
        placePanel(fCls, 1, 1, TASK_LABELS[4]);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double fps = (frameCount > 0 && ms > 0) ? (frameCount * 1000.0 / ms) : 0;
        std::string fpsStr = "FPS: " + std::to_string(static_cast<int>(fps + 0.5));
        cv::Mat fpsPanel = frame.clone();
        cv::putText(fpsPanel, fpsStr, cv::Point(cellW/4, cellH/2 - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 255), 2);
        placePanel(fpsPanel, 2, 1, TASK_LABELS[5]);

        writer.write(grid);
        frameCount++;
        if (frameCount % 30 == 0)
            std::cout << "Processed " << frameCount << "/" << totalFrames << " frames" << std::endl;
    }

    cap.release();
    writer.release();

    auto t2 = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(t2 - t0).count();
    double avgFps = (frameCount > 0 && totalMs > 0) ? (frameCount * 1000.0 / totalMs) : 0;
    std::cout << "Done. Frames: " << frameCount << ", time: " << (totalMs/1000) << " s, avg FPS: " << avgFps << std::endl;
    std::cout << "Output: " << outputPath << std::endl;
    return 0;
}
