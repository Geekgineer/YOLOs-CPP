#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <memory>
#include <chrono>

#include <opencv2/opencv.hpp>
#include "det/YOLO11.hpp"
#include "tools/Config.hpp"
#include "tools/ScopedTimer.hpp"

std::unique_ptr<YOLO11Detector> create_detector(
    const std::string& model_path,
    const std::string& labels_path,
    bool is_gpu)
{
    return std::make_unique<YOLO11Detector>(model_path, labels_path, is_gpu);
}

void benchmark_image(const std::string& model_path,
                     const std::string& labels_path,
                     const std::string& image_path,
                     bool is_gpu,
                     int iterations)
{
    auto detector = create_detector(model_path, labels_path, is_gpu);
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Could not read image at " << image_path << "\n";
        return;
    }

    std::vector<double> infer_times_us;

    // Warm-up
    for (int i = 0; i < 10; ++i) detector->detect(image);

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        detector->detect(image);
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        infer_times_us.push_back(us);
    }

    std::sort(infer_times_us.begin(), infer_times_us.end());
    double sum = std::accumulate(infer_times_us.begin(), infer_times_us.end(), 0.0);
    double avg = sum / infer_times_us.size();
    double med = infer_times_us[infer_times_us.size() / 2];
    double mn = infer_times_us.front();
    double mx = infer_times_us.back();

    printf("avg=%.3f ms, med=%.3f ms, min=%.3f ms, max=%.3f ms\n",
           avg / 1000.0, med / 1000.0, mn / 1000.0, mx / 1000.0);
}

void benchmark_video(const std::string& model_path,
                     const std::string& labels_path,
                     const std::string& video_path,
                     bool is_gpu)
{
    auto detector = create_detector(model_path, labels_path, is_gpu);
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video at " << video_path << "\n";
        return;
    }

    auto start = std::chrono::steady_clock::now();
    long frame_count = 0;
    cv::Mat frame;
    while (cap.read(frame)) {
        detector->detect(frame);
        ++frame_count;
    }
    auto end = std::chrono::steady_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double total_ms = total_us / 1000.0;
    double avg_ms = total_ms / frame_count;
    double fps = frame_count * 1000.0 / total_ms;

    printf("frames=%ld, total_ms=%.3f, avg_ms=%.6f, fps=%.2f\n",
           frame_count, total_ms, avg_ms, fps);
}

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " image <model> <labels> <image> <iters> [gpu]\n"
                  << "  " << argv[0] << " video <model> <labels> <video> [gpu]\n";
        return 1;
    }

    std::string mode = argv[1];
    std::string model = argv[2];
    std::string labels = argv[3];
    std::string media = argv[4];
    bool gpu = (argc > (mode == "image" ? 6 : 5) && std::string(argv[argc-1]) == "gpu");

    try {
        if (mode == "image") {
            int iters = std::stoi(argv[5]);
            benchmark_image(model, labels, media, gpu, iters);
        } else if (mode == "video") {
            benchmark_video(model, labels, media, gpu);
        } else {
            std::cerr << "Error: mode must be 'image' or 'video'\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
