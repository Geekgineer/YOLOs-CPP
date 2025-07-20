#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <fstream>
#include <map>
#include <sys/resource.h>
#include <iomanip>

#include <opencv2/opencv.hpp>

// For now, include only YOLO11 detector for testing
#include "det/YOLO11.hpp"

#include "tools/ScopedTimer.hpp"

// Benchmark configuration structure
struct BenchmarkConfig {
    std::string model_type;      // "yolo5", "yolo8", "yolo11", etc.
    std::string task_type;       // "detection", "segmentation", "obb", "pose"
    std::string model_path;
    std::string labels_path;
    bool use_gpu = true;
    int thread_count = 1;
    bool quantized = false;
    std::string precision = "fp32";
};

// Performance metrics structure
struct PerformanceMetrics {
    double load_time_ms = 0.0;
    double preprocess_avg_ms = 0.0;
    double inference_avg_ms = 0.0;
    double postprocess_avg_ms = 0.0;
    double total_avg_ms = 0.0;
    double fps = 0.0;
    double memory_mb = 0.0;
    double map_score = 0.0;  // For accuracy measurement when ground truth available
    int frame_count = 0;
};

// Memory measurement utility
double getCurrentMemoryUsageMB() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0; // Convert KB to MB on Linux
}

// Simplified detector factory for YOLO11 only (for now)
std::unique_ptr<YOLO11Detector> createYOLO11Detector(const BenchmarkConfig& config) {
    return std::make_unique<YOLO11Detector>(config.model_path, config.labels_path, config.use_gpu);
}

// Comprehensive image benchmark with detailed timing
PerformanceMetrics benchmark_image_comprehensive(const BenchmarkConfig& config,
                                                const std::string& image_path,
                                                int iterations = 100) {
    PerformanceMetrics metrics;
    
    // Measure model loading time
    auto load_start = std::chrono::high_resolution_clock::now();
    auto detector = createYOLO11Detector(config);
    auto load_end = std::chrono::high_resolution_clock::now();
    metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    
    // Load test image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Could not read image: " + image_path);
    }
    
    std::vector<double> preprocess_times, inference_times, postprocess_times, total_times;
    
    // Warm-up runs
    for (int i = 0; i < 10; ++i) {
        detector->detect(image);
    }
    
    // Measure memory usage
    double initial_memory = getCurrentMemoryUsageMB();
    
    // Benchmark runs with detailed timing
    for (int i = 0; i < iterations; ++i) {
        auto total_start = std::chrono::high_resolution_clock::now();
        
        // Note: If your detectors expose separate preprocess/inference/postprocess methods,
        // you can measure them separately here. For now, measuring total detection time.
        auto infer_start = std::chrono::high_resolution_clock::now();
        auto results = detector->detect(image);
        auto infer_end = std::chrono::high_resolution_clock::now();
        
        auto total_end = std::chrono::high_resolution_clock::now();
        
        double infer_time = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
        double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
        
        // For this implementation, we'll treat inference time as total time
        // In a more detailed implementation, you'd separate preprocessing/postprocessing
        preprocess_times.push_back(0.0);  // Placeholder
        inference_times.push_back(infer_time);
        postprocess_times.push_back(0.0); // Placeholder
        total_times.push_back(total_time);
    }
    
    // Calculate final memory usage
    double final_memory = getCurrentMemoryUsageMB();
    metrics.memory_mb = final_memory - initial_memory;
    
    // Calculate statistics
    auto calc_avg = [](const std::vector<double>& times) {
        return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    };
    
    metrics.preprocess_avg_ms = calc_avg(preprocess_times);
    metrics.inference_avg_ms = calc_avg(inference_times);
    metrics.postprocess_avg_ms = calc_avg(postprocess_times);
    metrics.total_avg_ms = calc_avg(total_times);
    metrics.fps = 1000.0 / metrics.total_avg_ms;
    
    return metrics;
}

// Video benchmark with throughput measurement
PerformanceMetrics benchmark_video_comprehensive(const BenchmarkConfig& config,
                                               const std::string& video_path) {
    PerformanceMetrics metrics;
    
    // Measure model loading time
    auto load_start = std::chrono::high_resolution_clock::now();
    auto detector = createYOLO11Detector(config);
    auto load_end = std::chrono::high_resolution_clock::now();
    metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    
    // Open video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Could not open video: " + video_path);
    }
    
    double initial_memory = getCurrentMemoryUsageMB();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    cv::Mat frame;
    
    std::vector<double> frame_times;
    
    while (cap.read(frame)) {
        auto frame_start = std::chrono::high_resolution_clock::now();
        auto results = detector->detect(frame);
        auto frame_end = std::chrono::high_resolution_clock::now();
        
        double frame_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        frame_times.push_back(frame_time);
        
        frame_count++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    double final_memory = getCurrentMemoryUsageMB();
    metrics.memory_mb = final_memory - initial_memory;
    
    metrics.frame_count = frame_count;
    metrics.total_avg_ms = std::accumulate(frame_times.begin(), frame_times.end(), 0.0) / frame_times.size();
    metrics.fps = (frame_count * 1000.0) / total_time;
    
    return metrics;
}

// Camera benchmark for real-time performance
PerformanceMetrics benchmark_camera_comprehensive(const BenchmarkConfig& config,
                                                int camera_id = 0,
                                                int duration_seconds = 30) {
    PerformanceMetrics metrics;
    
    auto load_start = std::chrono::high_resolution_clock::now();
    auto detector = createYOLO11Detector(config);
    auto load_end = std::chrono::high_resolution_clock::now();
    metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        throw std::runtime_error("Could not open camera: " + std::to_string(camera_id));
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
    
    double initial_memory = getCurrentMemoryUsageMB();
    
    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::seconds(duration_seconds);
    
    int frame_count = 0;
    std::vector<double> frame_times;
    cv::Mat frame;
    
    while (std::chrono::steady_clock::now() < end_time && cap.read(frame)) {
        auto frame_start = std::chrono::high_resolution_clock::now();
        auto results = detector->detect(frame);
        auto frame_finish = std::chrono::high_resolution_clock::now();
        
        double frame_time = std::chrono::duration<double, std::milli>(frame_finish - frame_start).count();
        frame_times.push_back(frame_time);
        
        frame_count++;
    }
    
    auto actual_end = std::chrono::steady_clock::now();
    double actual_duration = std::chrono::duration<double>(actual_end - start_time).count();
    
    double final_memory = getCurrentMemoryUsageMB();
    metrics.memory_mb = final_memory - initial_memory;
    
    metrics.frame_count = frame_count;
    metrics.total_avg_ms = std::accumulate(frame_times.begin(), frame_times.end(), 0.0) / frame_times.size();
    metrics.fps = frame_count / actual_duration;
    
    return metrics;
}

// CSV output utilities
void printCSVHeader() {
    std::cout << "model_type,task_type,device,threads,precision,load_ms,preprocess_ms,inference_ms,postprocess_ms,total_ms,fps,memory_mb,map_score,frame_count\n";
}

void printCSVRow(const BenchmarkConfig& config, const PerformanceMetrics& metrics) {
    std::cout << config.model_type << ","
            << config.task_type << ","
            << (config.use_gpu ? "gpu" : "cpu") << ","
            << config.thread_count << ","
            << config.precision << ","
              << std::fixed << std::setprecision(3)
              << metrics.load_time_ms << ","
              << metrics.preprocess_avg_ms << ","
              << metrics.inference_avg_ms << ","
              << metrics.postprocess_avg_ms << ","
              << metrics.total_avg_ms << ","
              <<"fps=="<< metrics.fps << ","
              << metrics.memory_mb << ","
              << metrics.map_score << ","
              << metrics.frame_count << std::endl;
}

// Configuration parsing
BenchmarkConfig parseConfig(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <mode> <model_type> <task_type> <model_path> <labels_path> <input_path> [options]\n"
                << "Modes: image, video, camera\n"
                << "Model types: yolo5, yolo7, yolo8, yolo9, yolo10, yolo11, yolo12\n"
                << "Task types: detection, segmentation, obb, pose\n"
                << "Options: --gpu, --cpu, --threads=N, --quantized, --iterations=N\n";
        throw std::runtime_error("Invalid arguments");
    }
    
    BenchmarkConfig config;
    config.model_type = argv[2];
    config.task_type = argv[3];
    config.model_path = argv[4];
    config.labels_path = argv[5];
    
    // Parse optional arguments
    for (int i = 7; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--gpu" || arg == "gpu") {
            config.use_gpu = true;
        } else if (arg == "--cpu" || arg == "cpu") {
            config.use_gpu = false;
        } else if (arg.substr(0, 10) == "--threads=") {
            config.thread_count = std::stoi(arg.substr(10));
        } else if (arg == "--quantized") {
            config.quantized = true;
            config.precision = "int8";
        }
    }
    
    return config;
}



int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <mode> <model_type> <task_type> <model_path> <labels_path> <input_path> [options]\n"
                      << "Modes: image, video, camera, comprehensive\n"
                      << "Model types: yolo5, yolo7, yolo8, yolo9, yolo10, yolo11, yolo12\n"
                      << "Task types: detection, segmentation, obb, pose\n"
                      << "Options: --gpu, --cpu, --threads=N, --quantized, --iterations=N, --duration=N\n"
                      << "\nExamples:\n"
                      << "  " << argv[0] << " image yolo11 detection model.onnx labels.txt test.jpg --gpu\n"
                      << "  " << argv[0] << " video yolo8 segmentation model.onnx labels.txt test.mp4 --cpu --threads=4\n"
                      << "  " << argv[0] << " camera yolo11 pose model.onnx labels.txt 0 --gpu --duration=30\n"
                      << "  " << argv[0] << " comprehensive  # Run all supported combinations\n";
            return 1;
        }

        std::string mode = argv[1];
        
        if (mode == "comprehensive") {
            // Run comprehensive benchmark across all supported configurations
            std::cout << "Running comprehensive YOLOs-CPP benchmark...\n";
            printCSVHeader();
            
            // Define test configurations
            std::vector<std::tuple<std::string, std::string, std::string>> test_configs = {
                {"yolo5", "detection", "../models/yolo5n.onnx"},
                {"yolo7", "detection", "../models/yolo7-tiny.onnx"},
                {"yolo8", "detection", "../models/yolo8n.onnx"},
                {"yolo9", "detection", "../models/yolo9s.onnx"},
                {"yolo10", "detection", "../models/yolo10n.onnx"},
                {"yolo11", "detection", "../models/yolo11n.onnx"},
                {"yolo12", "detection", "../models/yolo12n.onnx"},
                {"yolo8", "segmentation", "../models/yolo8n-seg.onnx"},
                {"yolo9", "segmentation", "../models/yolo9s-seg.onnx"},
                {"yolo11", "segmentation", "../models/yolo11n-seg.onnx"},
                {"yolo8", "obb", "../models/yolo8n-obb.onnx"},
                {"yolo11", "obb", "../models/yolo11n-obb.onnx"},
                {"yolo8", "pose", "../models/yolo8n-pose.onnx"},
                {"yolo11", "pose", "../models/yolo11n-pose.onnx"}
            };
            
            std::vector<bool> gpu_configs = {false, true};
            std::vector<int> thread_configs = {1, 4, 8};
            
            for (const auto& [model_type, task_type, model_path] : test_configs) {
                // Check if model file exists
                if (!std::ifstream(model_path).good()) {
                    std::cerr << "Skipping " << model_type << "/" << task_type << " - model not found: " << model_path << "\n";
                    continue;
                }
                
                for (bool use_gpu : gpu_configs) {
                    for (int threads : thread_configs) {
                        if (use_gpu && threads > 1) continue; // GPU doesn't use thread config
                        
                        BenchmarkConfig config;
                        config.model_type = model_type;
                        config.task_type = task_type;
                        config.model_path = model_path;
                        config.labels_path = "../models/coco.names";
                        config.use_gpu = use_gpu;
                        config.thread_count = threads;
                        
                        try {
                            // Run image benchmark
                            auto metrics = benchmark_image_comprehensive(config, "../data/dog.jpg", 50);
                            printCSVRow(config, metrics);
                        } catch (const std::exception& e) {
                            std::cerr << "Error benchmarking " << model_type << "/" << task_type 
                                      << " on " << (use_gpu ? "GPU" : "CPU") << ": " << e.what() << "\n";
                        }
                    }
                }
            }
            
            return 0;
        }
        
        // Parse configuration for single benchmark
        BenchmarkConfig config = parseConfig(argc, argv);
        std::string input_path = argv[6];
        
        int iterations = 100;
        int duration = 30;
        
        // Parse additional options
        for (int i = 7; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.substr(0, 13) == "--iterations=") {
                iterations = std::stoi(arg.substr(13));
            } else if (arg.substr(0, 11) == "--duration=") {
                duration = std::stoi(arg.substr(11));
            }
        }
        
        printCSVHeader();
        
        PerformanceMetrics metrics;
        
        if (mode == "image") {
            metrics = benchmark_image_comprehensive(config, input_path, iterations);
        } else if (mode == "video") {
            metrics = benchmark_video_comprehensive(config, input_path);
        } else if (mode == "camera") {
            int camera_id = std::stoi(input_path);
            metrics = benchmark_camera_comprehensive(config, camera_id, duration);
        } else {
            std::cerr << "Error: Invalid mode '" << mode << "'. Use 'image', 'video', 'camera', or 'comprehensive'.\n";
            return 1;
        }
        
        printCSVRow(config, metrics);
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
