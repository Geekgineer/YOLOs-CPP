/**
 * @file video_inference.cpp
 * @brief Object detection in a video stream using YOLO models (v5, v7, v8, v9, v10, v11, v12).
 * 
 * This file implements an object detection application that utilizes YOLO 
 * (You Only Look Once) models, specifically versions 5, 7, 8, 9, 10, 11 and 12. 
 * The application processes a video stream to detect objects and saves 
 * the results to a new video file with bounding boxes around detected objects.
 *
 * The application supports the following functionality:
 * - Loading a video stream from disk or camera.
 * - Initializing the YOLO detector with the desired model and labels.
 * - Detecting objects within each frame of the video.
 * - Drawing bounding boxes around detected objects and saving the result.
 *
 * Configuration parameters can be adjusted to suit specific requirements:
 * - `isGPU`: Set to true to enable GPU processing for improved performance; 
 *   set to false for CPU processing.
 * - `labelsPath`: Path to the class labels file (e.g., COCO dataset).
 * - `videoPath`: Path to the input video file (e.g., input.mp4).
 * - `outputPath`: Path for saving the output video file (e.g., output.mp4).
 * - `modelPath`: Path to the desired YOLO model file (e.g., yolo.onnx format).
 *
 * The application can be extended to use different YOLO versions by modifying 
 * the model path and the corresponding detector class.
 *
 * Usage Instructions:
 * 1. Compile the application with the necessary OpenCV and YOLO dependencies.
 * 2. Ensure that the specified video and model files are present in the 
 *    provided paths.
 * 3. Run the executable to initiate the object detection process.
 *
 * @note The code includes commented-out sections to demonstrate how to switch 
 * between different YOLO models and video inputs.
 *
 * Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
 * Date: 29.09.2024
 */
// Include necessary headers
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "det/YOLO.hpp"
// Thread-safe queue implementation
template <typename T>
class SafeQueue {
public:
    SafeQueue() : q(), m(), c() {}

    // Add an element to the queue.
    void enqueue(T t) {
        std::lock_guard<std::mutex> lock(m);
        q.push(t);
        c.notify_one();
    }

    // Get the first element from the queue.
    bool dequeue(T& t) {
        std::unique_lock<std::mutex> lock(m);
        while (q.empty()) {
            if (finished) return false;
            c.wait(lock);
        }
        t = q.front();
        q.pop();
        return true;
    }

    void setFinished() {
        std::lock_guard<std::mutex> lock(m);
        finished = true;
        c.notify_all();
    }

private:
    std::queue<T> q;
    mutable std::mutex m;
    std::condition_variable c;
    bool finished = false;
};

int main(int argc, char* argv[])
{
    // Paths to the model, labels, input video, and output video
    std::string labelsPath = "../models/coco.names";
    std::string videoPath = "../data/SIG_experience_center.mp4"; // Input video path
    std::string outputPath = "../data/SIG_experience_center_processed.mp4"; // Output video path
    std::string modelPath = "../models/yolo11n.onnx";

    if (argc > 1){
        modelPath = argv[1];
    }
    if (argc > 2){
        videoPath = argv[2];
    }
    if (argc > 3){
        outputPath = argv[3];
    }
    if (argc > 4){
        labelsPath = argv[4];
    }

    // Initialize the YOLO detector
    bool isGPU = true; // Set to false for CPU processing
    YOLODetector detector(modelPath, labelsPath, isGPU);

    // Processing routine for a single video (reuses existing threaded pipeline)
    auto processSingleVideo = [&](const std::string& inputVideoPath, const std::string& outputVideoPath) -> bool {
        cv::VideoCapture cap(inputVideoPath);
        if (!cap.isOpened())
        {
            std::cerr << "Error: Could not open or find the video file: " << inputVideoPath << "\n";
            return false;
        }

        int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
        int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

        cv::VideoWriter out(outputVideoPath, fourcc, fps, cv::Size(frameWidth, frameHeight), true);
        if (!out.isOpened())
        {
            std::cerr << "Error: Could not open the output video file for writing: " << outputVideoPath << "\n";
            cap.release();
            return false;
        }

        SafeQueue<cv::Mat> frameQueue;
        SafeQueue<std::pair<int, cv::Mat>> processedQueue;

        std::thread captureThread([&]() {
            cv::Mat frame;
            while (cap.read(frame))
            {
                frameQueue.enqueue(frame.clone());
            }
            frameQueue.setFinished();
        });

        std::thread processingThread([&]() {
            cv::Mat frame;
            int frameIndex = 0;
            while (frameQueue.dequeue(frame))
            {
                std::vector<Detection> results = detector.detect(frame);
                detector.drawBoundingBoxMask(frame, results);
                processedQueue.enqueue(std::make_pair(frameIndex++, frame));
            }
            processedQueue.setFinished();
        });

        std::thread writingThread([&]() {
            std::pair<int, cv::Mat> processedFrame;
            while (processedQueue.dequeue(processedFrame))
            {
                out.write(processedFrame.second);
            }
        });

        captureThread.join();
        processingThread.join();
        writingThread.join();

        cap.release();
        out.release();
        return true;
    };

    // Iterate over collected videos and process them
    for (const auto& vp : videoFiles)
    {
        std::cout << "\nProcessing: " << vp << std::endl;
        fs::path vpp(vp);
        fs::path outPath = vpp.parent_path() / (vpp.stem().string() + "_processed" + vpp.extension().string());
        bool ok = processSingleVideo(vp, outPath.string());
        if (!ok)
        {
            std::cerr << "Failed processing: " << vp << "\n";
        }
    }

    return 0;
}
