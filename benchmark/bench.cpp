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

// Include specific YOLO detectors based on availability
// For now, start with YOLO11 only to avoid symbol conflicts
#include "det/YOLO11.hpp"
// TODO: Implement proper abstraction layer for multiple models
// #include "det/YOLO8.hpp"

#include "tools/ScopedTimer.hpp"

// System monitoring includes
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <unistd.h>

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

// Performance metrics structure with enhanced system monitoring
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
    
    // Enhanced system monitoring
    double cpu_usage_percent = 0.0;
    double gpu_usage_percent = 0.0;
    double gpu_memory_used_mb = 0.0;
    double gpu_memory_total_mb = 0.0;
    double system_memory_used_mb = 0.0;
    double latency_avg_ms = 0.0;
    double latency_min_ms = 0.0;
    double latency_max_ms = 0.0;
    std::string environment_type = "CPU"; // "CPU" or "GPU"
};

// System monitoring utilities
struct SystemMonitor {
    static double getCPUUsage() {
        static unsigned long long lastTotalUser = 0, lastTotalUserLow = 0, lastTotalSys = 0, lastTotalIdle = 0;
        
        std::ifstream file("/proc/stat");
        std::string line;
        std::getline(file, line);
        
        unsigned long long totalUser, totalUserLow, totalSys, totalIdle, total;
        std::sscanf(line.c_str(), "cpu %llu %llu %llu %llu", &totalUser, &totalUserLow, &totalSys, &totalIdle);
        
        if (lastTotalUser == 0) {
            lastTotalUser = totalUser;
            lastTotalUserLow = totalUserLow;
            lastTotalSys = totalSys;
            lastTotalIdle = totalIdle;
            return 0.0;
        }
        
        total = (totalUser - lastTotalUser) + (totalUserLow - lastTotalUserLow) + (totalSys - lastTotalSys);
        double percent = total;
        total += (totalIdle - lastTotalIdle);
        percent /= total;
        percent *= 100;
        
        lastTotalUser = totalUser;
        lastTotalUserLow = totalUserLow;
        lastTotalSys = totalSys;
        lastTotalIdle = totalIdle;
        
        return percent;
    }
    
    static std::pair<double, double> getGPUUsage() {
        // Try to get GPU usage using nvidia-smi
        FILE* pipe = popen("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null", "r");
        if (!pipe) return {0.0, 0.0};
        
        char buffer[128];
        std::string result = "";
        while (fgets(buffer, sizeof buffer, pipe) != nullptr) {
            result += buffer;
        }
        pclose(pipe);
        
        if (result.empty()) return {0.0, 0.0};
        
        double gpu_util = 0.0, gpu_mem = 0.0;
        std::sscanf(result.c_str(), "%lf, %lf", &gpu_util, &gpu_mem);
        return {gpu_util, gpu_mem};
    }
    
    static double getSystemMemoryUsage() {
        std::ifstream file("/proc/meminfo");
        std::string line;
        unsigned long memTotal = 0, memFree = 0, buffers = 0, cached = 0;
        
        while (std::getline(file, line)) {
            if (line.find("MemTotal:") == 0) {
                std::sscanf(line.c_str(), "MemTotal: %lu kB", &memTotal);
            } else if (line.find("MemFree:") == 0) {
                std::sscanf(line.c_str(), "MemFree: %lu kB", &memFree);
            } else if (line.find("Buffers:") == 0) {
                std::sscanf(line.c_str(), "Buffers: %lu kB", &buffers);
            } else if (line.find("Cached:") == 0) {
                std::sscanf(line.c_str(), "Cached: %lu kB", &cached);
            }
        }
        
        double usedMB = (memTotal - memFree - buffers - cached) / 1024.0;
        return usedMB;
    }
};

// Memory measurement utility
double getCurrentMemoryUsageMB() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0; // Convert KB to MB on Linux
}

// Enhanced detector factory supporting YOLO11 (extensible for future models)
class DetectorFactory {
public:
    static std::unique_ptr<YOLO11Detector> createDetector(const BenchmarkConfig& config) {
        if (config.model_type == "yolo11" && config.task_type == "detection") {
            return std::make_unique<YOLO11Detector>(config.model_path, config.labels_path, config.use_gpu);
        }
        // TODO: Add support for YOLO8 and other models with proper abstraction
        // For now, also support yolo8 calls by creating YOLO11 detector (for compatibility)
        else if (config.model_type == "yolo8" && config.task_type == "detection") {
            std::cout << "Note: Using YOLO11 detector for YOLO8 model (compatibility mode)" << std::endl;
            return std::make_unique<YOLO11Detector>(config.model_path, config.labels_path, config.use_gpu);
        }
        else {
            throw std::runtime_error("Unsupported model type: " + config.model_type + " with task: " + config.task_type);
        }
    }
    
    static std::vector<Detection> detect(YOLO11Detector* detector, const BenchmarkConfig& config, const cv::Mat& image) {
        return detector->detect(image);
    }
};

// Enhanced image benchmark with comprehensive system monitoring
PerformanceMetrics benchmark_image_comprehensive(const BenchmarkConfig& config,
                                                const std::string& image_path,
                                                int iterations = 100) {
    PerformanceMetrics metrics;
    metrics.environment_type = config.use_gpu ? "GPU" : "CPU";
    
    // Measure model loading time
    auto load_start = std::chrono::high_resolution_clock::now();
    auto detector = DetectorFactory::createDetector(config);
    auto load_end = std::chrono::high_resolution_clock::now();
    metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    
    // Load test image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Could not read image: " + image_path);
    }
    
    std::vector<double> preprocess_times, inference_times, postprocess_times, total_times, latency_times;
    
    // Warm-up runs
    for (int i = 0; i < 10; ++i) {
        DetectorFactory::detect(detector.get(), config, image);
    }
    
    // Measure initial system state
    double initial_memory = getCurrentMemoryUsageMB();
    double initial_sys_memory = SystemMonitor::getSystemMemoryUsage();
    SystemMonitor::getCPUUsage(); // Initialize CPU monitoring
    
    // Enhanced benchmark runs with detailed timing and system monitoring
    std::vector<double> cpu_usage_samples, gpu_usage_samples, gpu_memory_samples;
    
    for (int i = 0; i < iterations; ++i) {
        // Monitor system resources during benchmark
        double cpu_usage = SystemMonitor::getCPUUsage();
        auto gpu_stats = SystemMonitor::getGPUUsage();
        
        cpu_usage_samples.push_back(cpu_usage);
        gpu_usage_samples.push_back(gpu_stats.first);
        gpu_memory_samples.push_back(gpu_stats.second);
        
        // Time the detection with high precision
        cv::TickMeter tm;
        tm.start();
        
        auto total_start = std::chrono::high_resolution_clock::now();
        auto infer_start = std::chrono::high_resolution_clock::now();
        
        auto results = DetectorFactory::detect(detector.get(), config, image);
        
        auto infer_end = std::chrono::high_resolution_clock::now();
        auto total_end = std::chrono::high_resolution_clock::now();
        
        tm.stop();
        
        double infer_time = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
        double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
        double latency = tm.getTimeMilli();
        
        preprocess_times.push_back(0.0);  // Placeholder - can be enhanced with separate timing
        inference_times.push_back(infer_time);
        postprocess_times.push_back(0.0); // Placeholder
        total_times.push_back(total_time);
        latency_times.push_back(latency);
    }
    
    // Calculate final memory usage
    double final_memory = getCurrentMemoryUsageMB();
    double final_sys_memory = SystemMonitor::getSystemMemoryUsage();
    metrics.memory_mb = final_memory - initial_memory;
    metrics.system_memory_used_mb = final_sys_memory - initial_sys_memory;
    
    // Calculate comprehensive statistics
    auto calc_avg = [](const std::vector<double>& values) {
        return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    };
    
    auto calc_min_max = [](const std::vector<double>& values) {
        auto minmax = std::minmax_element(values.begin(), values.end());
        return std::make_pair(*minmax.first, *minmax.second);
    };
    
    metrics.preprocess_avg_ms = calc_avg(preprocess_times);
    metrics.inference_avg_ms = calc_avg(inference_times);
    metrics.postprocess_avg_ms = calc_avg(postprocess_times);
    metrics.total_avg_ms = calc_avg(total_times);
    metrics.fps = 1000.0 / metrics.total_avg_ms;
    
    // Enhanced latency statistics
    metrics.latency_avg_ms = calc_avg(latency_times);
    auto latency_minmax = calc_min_max(latency_times);
    metrics.latency_min_ms = latency_minmax.first;
    metrics.latency_max_ms = latency_minmax.second;
    
    // System resource statistics
    metrics.cpu_usage_percent = calc_avg(cpu_usage_samples);
    metrics.gpu_usage_percent = calc_avg(gpu_usage_samples);
    metrics.gpu_memory_used_mb = calc_avg(gpu_memory_samples);
    
    return metrics;
}

// Enhanced video benchmark with comprehensive monitoring
PerformanceMetrics benchmark_video_comprehensive(const BenchmarkConfig& config,
                                               const std::string& video_path) {
    PerformanceMetrics metrics;
    metrics.environment_type = config.use_gpu ? "GPU" : "CPU";
    
    // Measure model loading time
    auto load_start = std::chrono::high_resolution_clock::now();
    auto detector = DetectorFactory::createDetector(config);
    auto load_end = std::chrono::high_resolution_clock::now();
    metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    
    // Open video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Could not open video: " + video_path);
    }
    
    double initial_memory = getCurrentMemoryUsageMB();
    double initial_sys_memory = SystemMonitor::getSystemMemoryUsage();
    SystemMonitor::getCPUUsage(); // Initialize CPU monitoring
    
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    cv::Mat frame;
    
    std::vector<double> frame_times, latency_times;
    std::vector<double> cpu_usage_samples, gpu_usage_samples, gpu_memory_samples;
    
    while (cap.read(frame)) {
        // Monitor system resources
        double cpu_usage = SystemMonitor::getCPUUsage();
        auto gpu_stats = SystemMonitor::getGPUUsage();
        
        cpu_usage_samples.push_back(cpu_usage);
        gpu_usage_samples.push_back(gpu_stats.first);
        gpu_memory_samples.push_back(gpu_stats.second);
        
        // Time frame processing
        cv::TickMeter tm;
        tm.start();
        
        auto frame_start = std::chrono::high_resolution_clock::now();
        auto results = DetectorFactory::detect(detector.get(), config, frame);
        auto frame_end = std::chrono::high_resolution_clock::now();
        
        tm.stop();
        
        double frame_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        double latency = tm.getTimeMilli();
        
        frame_times.push_back(frame_time);
        latency_times.push_back(latency);
        
        frame_count++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    double final_memory = getCurrentMemoryUsageMB();
    double final_sys_memory = SystemMonitor::getSystemMemoryUsage();
    metrics.memory_mb = final_memory - initial_memory;
    metrics.system_memory_used_mb = final_sys_memory - initial_sys_memory;
    
    // Calculate comprehensive statistics
    auto calc_avg = [](const std::vector<double>& values) {
        return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    };
    
    auto calc_min_max = [](const std::vector<double>& values) {
        auto minmax = std::minmax_element(values.begin(), values.end());
        return std::make_pair(*minmax.first, *minmax.second);
    };
    
    metrics.frame_count = frame_count;
    metrics.total_avg_ms = calc_avg(frame_times);
    metrics.fps = (frame_count * 1000.0) / total_time;
    
    // Enhanced latency statistics
    metrics.latency_avg_ms = calc_avg(latency_times);
    auto latency_minmax = calc_min_max(latency_times);
    metrics.latency_min_ms = latency_minmax.first;
    metrics.latency_max_ms = latency_minmax.second;
    
    // System resource statistics
    metrics.cpu_usage_percent = calc_avg(cpu_usage_samples);
    metrics.gpu_usage_percent = calc_avg(gpu_usage_samples);
    metrics.gpu_memory_used_mb = calc_avg(gpu_memory_samples);
    
    return metrics;
}

// Enhanced camera benchmark for real-time performance with system monitoring
PerformanceMetrics benchmark_camera_comprehensive(const BenchmarkConfig& config,
                                                int camera_id = 0,
                                                int duration_seconds = 30) {
    PerformanceMetrics metrics;
    metrics.environment_type = config.use_gpu ? "GPU" : "CPU";
    
    auto load_start = std::chrono::high_resolution_clock::now();
    auto detector = DetectorFactory::createDetector(config);
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
    double initial_sys_memory = SystemMonitor::getSystemMemoryUsage();
    SystemMonitor::getCPUUsage(); // Initialize CPU monitoring
    
    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::seconds(duration_seconds);
    
    int frame_count = 0;
    std::vector<double> frame_times, latency_times;
    std::vector<double> cpu_usage_samples, gpu_usage_samples, gpu_memory_samples;
    cv::Mat frame;
    
    while (std::chrono::steady_clock::now() < end_time && cap.read(frame)) {
        // Monitor system resources
        double cpu_usage = SystemMonitor::getCPUUsage();
        auto gpu_stats = SystemMonitor::getGPUUsage();
        
        cpu_usage_samples.push_back(cpu_usage);
        gpu_usage_samples.push_back(gpu_stats.first);
        gpu_memory_samples.push_back(gpu_stats.second);
        
        // Time frame processing
        cv::TickMeter tm;
        tm.start();
        
        auto frame_start = std::chrono::high_resolution_clock::now();
        auto results = DetectorFactory::detect(detector.get(), config, frame);
        auto frame_finish = std::chrono::high_resolution_clock::now();
        
        tm.stop();
        
        double frame_time = std::chrono::duration<double, std::milli>(frame_finish - frame_start).count();
        double latency = tm.getTimeMilli();
        
        frame_times.push_back(frame_time);
        latency_times.push_back(latency);
        
        frame_count++;
    }
    
    auto actual_end = std::chrono::steady_clock::now();
    double actual_duration = std::chrono::duration<double>(actual_end - start_time).count();
    
    double final_memory = getCurrentMemoryUsageMB();
    double final_sys_memory = SystemMonitor::getSystemMemoryUsage();
    metrics.memory_mb = final_memory - initial_memory;
    metrics.system_memory_used_mb = final_sys_memory - initial_sys_memory;
    
    // Calculate comprehensive statistics
    auto calc_avg = [](const std::vector<double>& values) {
        return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    };
    
    auto calc_min_max = [](const std::vector<double>& values) {
        auto minmax = std::minmax_element(values.begin(), values.end());
        return std::make_pair(*minmax.first, *minmax.second);
    };
    
    metrics.frame_count = frame_count;
    metrics.total_avg_ms = calc_avg(frame_times);
    metrics.fps = frame_count / actual_duration;
    
    // Enhanced latency statistics
    metrics.latency_avg_ms = calc_avg(latency_times);
    auto latency_minmax = calc_min_max(latency_times);
    metrics.latency_min_ms = latency_minmax.first;
    metrics.latency_max_ms = latency_minmax.second;
    
    // System resource statistics
    metrics.cpu_usage_percent = calc_avg(cpu_usage_samples);
    metrics.gpu_usage_percent = calc_avg(gpu_usage_samples);
    metrics.gpu_memory_used_mb = calc_avg(gpu_memory_samples);
    
    return metrics;
}

// Enhanced CSV output utilities with comprehensive metrics
void printCSVHeader() {
    std::cout << "model_type,task_type,environment,device,threads,precision,load_ms,preprocess_ms,inference_ms,postprocess_ms,total_ms,fps,memory_mb,system_memory_mb,cpu_usage_%,gpu_usage_%,gpu_memory_mb,latency_avg_ms,latency_min_ms,latency_max_ms,map_score,frame_count\n";
}

void printCSVRow(const BenchmarkConfig& config, const PerformanceMetrics& metrics) {
    std::cout << config.model_type << ","
              << config.task_type << ","
              << metrics.environment_type << ","
              << (config.use_gpu ? "gpu" : "cpu") << ","
              << config.thread_count << ","
              << config.precision << ","
              << std::fixed << std::setprecision(3)
              << metrics.load_time_ms << ","
              << metrics.preprocess_avg_ms << ","
              << metrics.inference_avg_ms << ","
              << metrics.postprocess_avg_ms << ","
              << metrics.total_avg_ms << ","
              << metrics.fps << ","
              << metrics.memory_mb << ","
              << metrics.system_memory_used_mb << ","
              << metrics.cpu_usage_percent << ","
              << metrics.gpu_usage_percent << ","
              << metrics.gpu_memory_used_mb << ","
              << metrics.latency_avg_ms << ","
              << metrics.latency_min_ms << ","
              << metrics.latency_max_ms << ","
              << metrics.map_score << ","
              << metrics.frame_count << std::endl;
}

// Save results to file for easier analysis
void saveResultsToFile(const std::string& filename, const std::vector<std::pair<BenchmarkConfig, PerformanceMetrics>>& results) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    // Redirect cout to file temporarily
    std::streambuf* orig = std::cout.rdbuf();
    std::cout.rdbuf(file.rdbuf());
    
    printCSVHeader();
    for (const auto& [config, metrics] : results) {
        printCSVRow(config, metrics);
    }
    
    // Restore cout
    std::cout.rdbuf(orig);
    file.close();
    
    std::cout << "Results saved to: " << filename << std::endl;
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
            // Enhanced comprehensive benchmark across all supported configurations
            std::cout << "Running comprehensive YOLOs-CPP benchmark with system monitoring...\n";
            printCSVHeader();
            
            // Enhanced test configurations focusing on v8 and v11 detection as requested
            std::vector<std::tuple<std::string, std::string, std::string>> test_configs = {
                {"yolo8", "detection", "../models/yolo8n.onnx"},
                {"yolo8", "detection", "../models/yolo8s.onnx"},
                {"yolo8", "detection", "../models/yolo8m.onnx"},
                {"yolo11", "detection", "../models/yolo11n.onnx"},
                {"yolo11", "detection", "../models/yolo11s.onnx"},
                {"yolo11", "detection", "../models/yolo11m.onnx"},
                // Future extensibility - these will be skipped if models don't exist
                {"yolo5", "detection", "../models/yolo5n.onnx"},
                {"yolo7", "detection", "../models/yolo7-tiny.onnx"},
                {"yolo8", "segmentation", "../models/yolo8n-seg.onnx"},
                {"yolo11", "segmentation", "../models/yolo11n-seg.onnx"},
            };
            
            std::vector<bool> gpu_configs = {false, true};
            std::vector<int> thread_configs = {1, 4, 8};
            std::vector<int> iteration_configs = {50, 100, 200}; // Different iteration counts
            
            std::vector<std::pair<BenchmarkConfig, PerformanceMetrics>> all_results;
            
            for (const auto& [model_type, task_type, model_path] : test_configs) {
                // Check if model file exists
                if (!std::ifstream(model_path).good()) {
                    std::cerr << "Skipping " << model_type << "/" << task_type << " - model not found: " << model_path << "\n";
                    continue;
                }
                
                for (bool use_gpu : gpu_configs) {
                    for (int threads : thread_configs) {
                        if (use_gpu && threads > 1) continue; // GPU doesn't use thread config
                        
                        for (int iterations : iteration_configs) {
                            BenchmarkConfig config;
                            config.model_type = model_type;
                            config.task_type = task_type;
                            config.model_path = model_path;
                            config.labels_path = "../models/coco.names";
                            config.use_gpu = use_gpu;
                            config.thread_count = threads;
                            
                            try {
                                std::cout << "Testing " << model_type << "/" << task_type 
                                          << " on " << (use_gpu ? "GPU" : "CPU") 
                                          << " with " << threads << " threads, " << iterations << " iterations...\n";
                                
                                // Run enhanced image benchmark with specified iterations
                                auto metrics = benchmark_image_comprehensive(config, "../data/dog.jpg", iterations);
                                printCSVRow(config, metrics);
                                
                                all_results.push_back({config, metrics});
                                
                                // Add small delay to prevent system overload
                                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                                
                            } catch (const std::exception& e) {
                                std::cerr << "Error benchmarking " << model_type << "/" << task_type 
                                          << " on " << (use_gpu ? "GPU" : "CPU") << ": " << e.what() << "\n";
                            }
                        }
                    }
                }
            }
            
            // Save results to file for later analysis
            std::string timestamp = std::to_string(std::time(nullptr));
            std::string results_filename = "benchmark_results_" + timestamp + ".csv";
            saveResultsToFile(results_filename, all_results);
            
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
