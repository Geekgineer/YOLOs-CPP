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
#include <filesystem>
#include <algorithm>
#include <cctype>

// Uncomment the version
//#define YOLO5 // Uncomment for YOLOv5
//#define YOLO7 // Uncomment for YOLOv7
//#define YOLO8 // Uncomment for YOLOv8
//#define YOLO9 // Uncomment for YOLOv9
//#define YOLO10 // Uncomment for YOLOv10
#define YOLO11 // Uncomment for YOLOv11
//#define YOLO12 // Uncomment for YOLOv12

#ifdef YOLO5
    #include "det/YOLO5.hpp"
#endif
#ifdef YOLO7
    #include "det/YOLO7.hpp"
#endif
#ifdef YOLO8
    #include "det/YOLO8.hpp"
#endif
#ifdef YOLO9
    #include "det/YOLO9.hpp"
#endif
#ifdef YOLO10
    #include "det/YOLO10.hpp"
#endif
#ifdef YOLO11
    #include "det/YOLO11.hpp"
#endif
#ifdef YOLO12
    #include "det/YOLO12.hpp"
#endif

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

int main(int argc, char** argv)
{
    // Paths to the model and labels
    const std::string labelsPath = "models/coco.names";

    // Model paths for different YOLO versions
    #ifdef YOLO5
        std::string modelPath = "models/yolo5-n6.onnx";
    #endif
    #ifdef YOLO7
        const std::string modelPath = "models/yolo7-tiny.onnx";
    #endif
    #ifdef YOLO8
        std::string modelPath = "models/yolo8n.onnx";
    #endif
    #ifdef YOLO9
        const std::string modelPath = "models/yolov9s.onnx";
    #endif
    #ifdef YOLO10
        std::string modelPath = "models/yolo10n_uint8.onnx";
    #endif
    #ifdef YOLO11
        const std::string modelPath = "models/yolo11n.onnx";
    #endif
    #ifdef YOLO12
        const std::string modelPath = "models/yolo12n.onnx";
    #endif

    // Initialize the YOLO detector
    bool isGPU = true; // Set to false for CPU processing
    #ifdef YOLO5
        YOLO5Detector detector(modelPath, labelsPath, isGPU);
    #endif
    #ifdef YOLO7
        YOLO7Detector detector(modelPath, labelsPath, isGPU);
    #endif
    #ifdef YOLO8
        YOLO8Detector detector(modelPath, labelsPath, isGPU);
    #endif
    #ifdef YOLO9
        YOLO9Detector detector(modelPath, labelsPath, isGPU);
    #endif
    #ifdef YOLO11
        YOLO11Detector detector(modelPath, labelsPath, isGPU);
    #endif
    #ifdef YOLO12
        YOLO12Detector detector(modelPath, labelsPath, isGPU);
    #endif

    // Helpers
    auto toLower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        return s;
    };

    auto isVideoFile = [&](const std::filesystem::path& p) {
        static const std::vector<std::string> exts = {
            ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".mpg", ".mpeg"
        };
        std::string ext = toLower(p.extension().string());
        return std::find(exts.begin(), exts.end(), ext) != exts.end();
    };

    // Processing routine for a single video
    auto processSingleVideo = [&](const std::string& inputVideoPath, const std::string& outputVideoPath) -> bool {
        // Open the video file
        cv::VideoCapture cap(inputVideoPath);
        if (!cap.isOpened())
        {
            std::cerr << "Error: Could not open or find the video file: " << inputVideoPath << "\n";
            return false;
        }

        // Get video properties
        int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
        int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC)); // Get codec of input video

        // Create a VideoWriter object to save the output video with the same codec
        cv::VideoWriter out(outputVideoPath, fourcc, fps, cv::Size(frameWidth, frameHeight), true);
        if (!out.isOpened())
        {
            std::cerr << "Error: Could not open the output video file for writing: " << outputVideoPath << "\n";
            cap.release();
            return false;
        }

        // Thread-safe queues
        SafeQueue<cv::Mat> frameQueue;
        SafeQueue<std::pair<int, cv::Mat>> processedQueue;

        // Capture thread
        std::thread captureThread([&]() {
            cv::Mat frame;
            while (cap.read(frame))
            {
                frameQueue.enqueue(frame.clone()); // Clone to ensure thread safety
            }
            frameQueue.setFinished();
        });

        // Processing thread
        std::thread processingThread([&]() {
            cv::Mat frame;
            int frameIndex = 0;
            while (frameQueue.dequeue(frame))
            {
                // Detect objects in the frame
                std::vector<Detection> results = detector.detect(frame);

                // Draw bounding boxes on the frame
                detector.drawBoundingBoxMask(frame, results); // Uncomment for mask drawing

                // Enqueue the processed frame
                processedQueue.enqueue(std::make_pair(frameIndex++, frame));
            }
            processedQueue.setFinished();
        });

        // Writing thread
        std::thread writingThread([&]() {
            std::pair<int, cv::Mat> processedFrame;
            while (processedQueue.dequeue(processedFrame))
            {
                out.write(processedFrame.second);
            }
        });

        // Wait for all threads to finish
        captureThread.join();
        processingThread.join();
        writingThread.join();

        // Release resources
        cap.release();
        out.release();

        return true;
    };

    // CLI handling: expect one argument: path to a video file or a folder containing videos
    if (argc < 2)
    {
        std::cerr << "Usage: <executable> <path_to_video_or_directory>\n";
        return 1;
    }

    std::filesystem::path inputPath(argv[1]);
    if (!std::filesystem::exists(inputPath))
    {
        std::cerr << "Error: Path does not exist: " << inputPath << "\n";
        return 1;
    }

    if (std::filesystem::is_regular_file(inputPath))
    {
        if (!isVideoFile(inputPath))
        {
            std::cerr << "Error: Provided file is not a supported video: " << inputPath << "\n";
            return 1;
        }
        std::filesystem::path outputPath = inputPath.parent_path() / (inputPath.stem().string() + "_processed" + inputPath.extension().string());
        bool ok = processSingleVideo(inputPath.string(), outputPath.string());
        return ok ? 0 : 1;
    }
    else if (std::filesystem::is_directory(inputPath))
    {
        std::vector<std::filesystem::path> videoFiles;
        for (const auto& entry : std::filesystem::directory_iterator(inputPath))
        {
            if (entry.is_regular_file() && isVideoFile(entry.path()))
            {
                videoFiles.push_back(entry.path());
            }
        }

        if (videoFiles.empty())
        {
            std::cerr << "Error: No supported video files found in directory: " << inputPath << "\n";
            return 1;
        }

        int numFailed = 0;
        for (const auto& vp : videoFiles)
        {
            std::filesystem::path outputPath = vp.parent_path() / (vp.stem().string() + "_processed" + vp.extension().string());
            std::cout << "Processing: " << vp << " -> " << outputPath << std::endl;
            if (!processSingleVideo(vp.string(), outputPath.string()))
            {
                std::cerr << "Failed processing: " << vp << "\n";
                ++numFailed;
            }
        }

        if (numFailed > 0)
        {
            std::cerr << "Completed with " << numFailed << " failures." << std::endl;
            return 1;
        }

        std::cout << "All videos processed successfully." << std::endl;
        return 0;
    }
    else
    {
        std::cerr << "Error: Provided path is neither a regular file nor a directory: " << inputPath << "\n";
        return 1;
    }
}
