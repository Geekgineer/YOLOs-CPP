#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <memory>

#include <opencv2/opencv.hpp>

// Include necessary headers from the YOLOs-CPP project
#include "YOLOv8.hpp"
#include "YOLOv8-SEG.hpp"
#include "YOLOv8-POSE.hpp"
#include "Config.hpp"
#include "ScopedTimer.hpp"

// --- Helper Function to Create Detector ---
std::unique_ptr<Detector> create_detector(const std::string& model_type, const std::string& model_path, bool is_gpu, int device_id) {
    Config config;
    config.model_path = model_path;
    config.isGPU = is_gpu;
    config.gpuID = device_id;

    if (model_type == "detection") {
        return std::make_unique<YOLOv8>(config);
    } else if (model_type == "segmentation") {
        return std::make_unique<YOLOv8_SEG>(config);
    } else if (model_type == "pose") {
        return std::make_unique<YOLOv8_POSE>(config);
    } else {
        throw std::runtime_error("Unknown model type: " + model_type);
    }
}

// --- Image Benchmark Function ---
void benchmark_image(const std::string& model_type, const std::string& model_path, const std::string& image_path, bool is_gpu, int iterations) {
    auto detector = create_detector(model_type, model_path, is_gpu, 0);
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Could not read image at " << image_path << std::endl;
        return;
    }

    std::vector<double> pre_times, infer_times, post_times, total_times;

    // Warmup
    for (int i = 0; i < 10; ++i) {
        detector->detect(image);
    }

    for (int i = 0; i < iterations; ++i) {
        ScopedTimer pre_timer("preprocess");
        cv::Mat processed_image = detector->preprocess(image);
        pre_times.push_back(pre_timer.elapsed_ms());

        ScopedTimer infer_timer("inference");
        std::vector<std::vector<cv::Mat>> outputs = detector->inference(processed_image);
        infer_times.push_back(infer_timer.elapsed_ms());

        ScopedTimer post_timer("postprocess");
        if (model_type == "detection") {
             dynamic_cast<YOLOv8*>(detector.get())->postprocess(outputs, image.size());
        } else if (model_type == "segmentation") {
             dynamic_cast<YOLOv8_SEG*>(detector.get())->postprocess(outputs, image.size());
        } else if (model_type == "pose") {
             dynamic_cast<YOLOv8_POSE*>(detector.get())->postprocess(outputs, image.size());
        }
        post_times.push_back(post_timer.elapsed_ms());

        total_times.push_back(pre_times.back() + infer_times.back() + post_times.back());
    }

    auto get_stats = [](std::vector<double>& v) {
        std::sort(v.begin(), v.end());
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        double avg = sum / v.size();
        double med = v[v.size() / 2];
        return std::make_tuple(avg, med, v.front(), v.back());
    };

    auto [pre_avg, pre_med, pre_min, pre_max] = get_stats(pre_times);
    auto [infer_avg, infer_med, infer_min, infer_max] = get_stats(infer_times);
    auto [post_avg, post_med, post_min, post_max] = get_stats(post_times);
    auto [total_avg, total_med, total_min, total_max] = get_stats(total_times);

    printf("%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
           pre_avg, infer_avg, post_avg, total_avg,
           pre_med, infer_med, post_med, total_med,
           pre_min, infer_min, post_min, total_min,
           1000.0 / total_avg);
}

// --- Video Benchmark Function ---
void benchmark_video(const std::string& model_type, const std::string& model_path, const std::string& video_path, bool is_gpu) {
    auto detector = create_detector(model_type, model_path, is_gpu, 0);
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video at " << video_path << std::endl;
        return;
    }

    long frame_count = 0;
    double total_time_ms = 0;

    cv::Mat frame;
    ScopedTimer video_timer("video_total");
    while (cap.read(frame)) {
        detector->detect(frame);
        frame_count++;
    }
    total_time_ms = video_timer.elapsed_ms();
    
    double avg_fps = (frame_count / total_time_ms) * 1000.0;
    double avg_ms_per_frame = total_time_ms / frame_count;

    printf("%ld,%.3f,%.3f\n", frame_count, avg_ms_per_frame, avg_fps);
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage for Image: " << argv[0] << " image <model_type> <model_path> <image_path> <iterations> [gpu]" << std::endl;
        std::cerr << "Usage for Video: " << argv[0] << " video <model_type> <model_path> <video_path> [gpu]" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    std::string model_type = argv[2];
    std::string model_path = argv[3];
    std::string media_path = argv[4];
    bool is_gpu = (argc > (mode == "image" ? 6 : 5) && std::string(argv[argc - 1]) == "gpu");

    try {
        if (mode == "image") {
            int iterations = std::stoi(argv[5]);
            benchmark_image(model_type, model_path, media_path, is_gpu, iterations);
        } else if (mode == "video") {
            benchmark_video(model_type, model_path, media_path, is_gpu);
        } else {
            std::cerr << "Error: Unknown mode '" << mode << "'. Use 'image' or 'video'." << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
