/**
 * @file camera_inference.cpp
 * @brief Real-time object detection using YOLO models (v5, v7, v8, v9, v10, v11, v12) with camera input.
 * 
 * This file serves as the main entry point for a real-time object detection 
 * application that utilizes YOLO (You Only Look Once) models, specifically 
 * versions 5, 7, 8, 9, 10, 11 and 12. The application captures video frames from a 
 * specified camera device, processes those frames to detect objects, and 
 * displays the results with bounding boxes around detected objects.
 *
 * The program operates in a multi-threaded environment, featuring the following 
 * threads:
 * 1. **Producer Thread**: Responsible for capturing frames from the video source 
 *    and enqueuing them into a thread-safe bounded queue for subsequent processing.
 * 2. **Consumer Thread**: Dequeues frames from the producer's queue, executes 
 *    object detection using the specified YOLO model, and enqueues the processed 
 *    frames along with detection results into another thread-safe bounded queue.
 * 3. **Display Thread**: Dequeues processed frames from the consumer's queue, 
 *    draws bounding boxes around detected objects, and displays the frames to the 
 *    user.
 *
 * Configuration parameters can be adjusted to suit specific requirements:
 * - `isGPU`: Set to true to enable GPU processing for improved performance; 
 *   set to false for CPU processing.
 * - `labelsPath`: Path to the class labels file (e.g., COCO dataset).
 * - `modelPath`: Path to the desired YOLO model file (e.g., ONNX format).
 * - `videoSource`: Path to the video capture device (e.g., camera).
 *
 * The application employs a double buffering technique by maintaining two bounded 
 * queues to efficiently manage the flow of frames between the producer and 
 * consumer threads. This setup helps prevent processing delays due to slow frame 
 * capture or detection times.
 *
 * Debugging messages can be enabled by defining the `DEBUG_MODE` macro, allowing 
 * developers to trace the execution flow and internal state of the application 
 * during runtime.
 *
 * Usage Instructions:
 * 1. Compile the application with the necessary OpenCV and YOLO dependencies.
 * 2. Run the executable to initiate the object detection process.
 * 3. Press 'q' to quit the application at any time.
 *
 * @note Ensure that the required model files and labels are present in the 
 * specified paths before running the application.
 *
 * Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
 * Date: 29.09.2024
 */


#include <iostream>
#include <vector>
#include <thread>
#include <atomic>

#include <opencv2/highgui/highgui.hpp>
// #ifndef DEBUG_MODE
// #define DEBUG_MODE
// #endif
// #ifndef TIMING_MODE
// #define TIMING_MODE
// #endif
#include "yolos/tasks/detection.hpp"

using namespace yolos::det;


// Include the bounded queue
#include "tools/BoundedThreadSafeQueue.hpp"

int main(int argc, char* argv[])
{
    // Configuration parameters
    const bool isGPU = true;
    std::string labelsPath = "../models/coco.names";
    std::string modelPath = "../models/yolo11n.onnx";

    std::string videoSource = "/dev/video0"; // your usb cam device
    if (argc > 1){
        modelPath = argv[1];
    }
    if (argc > 2){
        videoSource = argv[2];
    }
    if (argc > 3){
        labelsPath = argv[3];
    }
    YOLODetector detector(modelPath, labelsPath, isGPU);


    // Open video capture
    cv::VideoCapture cap;
    cap.open(videoSource, cv::CAP_V4L2); // Specify V4L2 backend for better performance
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open the camera!\n";
        return -1;
    }

    // Set camera properties
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);

    // Initialize queues with bounded capacity
    const size_t max_queue_size = 2; // Double buffering
    BoundedThreadSafeQueue<cv::Mat> frameQueue(max_queue_size);
    BoundedThreadSafeQueue<std::pair<cv::Mat, std::vector<Detection>>> processedQueue(max_queue_size);
    std::atomic<bool> stopFlag(false);

    // Producer thread: Capture frames
    std::thread producer([&]() {
        cv::Mat frame;
        while (!stopFlag.load() && cap.read(frame))
        {
            if (!frameQueue.enqueue(frame))
                break; // Queue is finished
        }
        frameQueue.set_finished();
    });

    // Consumer thread: Process frames
    std::thread consumer([&]() {
        cv::Mat frame;
        while (!stopFlag.load() && frameQueue.dequeue(frame))
        {
            // Perform detection
            std::vector<Detection> detections = detector.detect(frame);

            // Enqueue processed frame
            if (!processedQueue.enqueue(std::make_pair(frame, detections)))
                break;
        }
        processedQueue.set_finished();
    });

    std::pair<cv::Mat, std::vector<Detection>> item;

    #ifdef __APPLE__
    // For macOS, ensure UI runs on the main thread
    while (!stopFlag.load() && processedQueue.dequeue(item))
    {
        cv::Mat displayFrame = item.first;
        detector.drawDetectionsWithMask(displayFrame, item.second);

        cv::imshow("Detections", displayFrame);
        if (cv::waitKey(1) == 'q')
        {
            stopFlag.store(true);
            frameQueue.set_finished();
            processedQueue.set_finished();
            break;
        }
    }
    #else
    // Display thread: Show processed frames
    std::thread displayThread([&]() {
        while (!stopFlag.load() && processedQueue.dequeue(item))
        {
            cv::Mat displayFrame = item.first;
            // detector.drawDetections(displayFrame, item.second);
            detector.drawDetectionsWithMask(displayFrame, item.second);

            // Display the frame
            cv::imshow("Detections", displayFrame);
            // Use a small delay and check for 'q' key press to quit
            if (cv::waitKey(1) == 'q') {
                stopFlag.store(true);
                frameQueue.set_finished();
                processedQueue.set_finished();
                break;
            }
        }
    });
    displayThread.join();
    #endif

    // Join all threads
    producer.join();
    consumer.join();

    // Release resources
    cap.release();
    cv::destroyAllWindows();

    return 0;
}