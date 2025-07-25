// #include <iostream>
// #include <vector>
// #include <string>
// #include <numeric>
// #include <algorithm>
// #include <memory>
// #include <chrono>
// #include <thread>
// #include <fstream>
// #include <map>
// #include <sys/resource.h>
// #include <iomanip>

// #include <opencv2/opencv.hpp>

// // Include specific YOLO detectors based on availability
// // For now, start with YOLO11 only to avoid symbol conflicts
// #include "det/YOLO11.hpp"
// // TODO: Implement proper abstraction layer for multiple models
// // #include "det/YOLO8.hpp"

// #include "tools/ScopedTimer.hpp"

// // System monitoring includes
// #include <sys/stat.h>
// #include <fstream>
// #include <sstream>
// #include <unistd.h>

// // Benchmark configuration structure
// struct BenchmarkConfig {
//     std::string model_type;      // "yolo5", "yolo8", "yolo11", etc.
//     std::string task_type;       // "detection", "segmentation", "obb", "pose"
//     std::string model_path;
//     std::string labels_path;
//     bool use_gpu = true;
//     int thread_count = 1;
//     bool quantized = false;
//     std::string precision = "fp32";
// };

// // Performance metrics structure with enhanced system monitoring
// struct PerformanceMetrics {
//     double load_time_ms = 0.0;
//     double preprocess_avg_ms = 0.0;
//     double inference_avg_ms = 0.0;
//     double postprocess_avg_ms = 0.0;
//     double total_avg_ms = 0.0;
//     double fps = 0.0;
//     double memory_mb = 0.0;
//     double map_score = 0.0;  // For accuracy measurement when ground truth available
//     int frame_count = 0;
    
//     // Enhanced system monitoring
//     double cpu_usage_percent = 0.0;
//     double gpu_usage_percent = 0.0;
//     double gpu_memory_used_mb = 0.0;
//     double gpu_memory_total_mb = 0.0;
//     double system_memory_used_mb = 0.0;
//     double latency_avg_ms = 0.0;
//     double latency_min_ms = 0.0;
//     double latency_max_ms = 0.0;
//     std::string environment_type = "CPU"; // "CPU" or "GPU"
// };

// // System monitoring utilities
// struct SystemMonitor {
//     static double getCPUUsage() {
//         static unsigned long long lastTotalUser = 0, lastTotalUserLow = 0, lastTotalSys = 0, lastTotalIdle = 0;
        
//         std::ifstream file("/proc/stat");
//         std::string line;
//         std::getline(file, line);
        
//         unsigned long long totalUser, totalUserLow, totalSys, totalIdle, total;
//         std::sscanf(line.c_str(), "cpu %llu %llu %llu %llu", &totalUser, &totalUserLow, &totalSys, &totalIdle);
        
//         if (lastTotalUser == 0) {
//             lastTotalUser = totalUser;
//             lastTotalUserLow = totalUserLow;
//             lastTotalSys = totalSys;
//             lastTotalIdle = totalIdle;
//             return 0.0;
//         }
        
//         total = (totalUser - lastTotalUser) + (totalUserLow - lastTotalUserLow) + (totalSys - lastTotalSys);
//         double percent = total;
//         total += (totalIdle - lastTotalIdle);
//         percent /= total;
//         percent *= 100;
        
//         lastTotalUser = totalUser;
//         lastTotalUserLow = totalUserLow;
//         lastTotalSys = totalSys;
//         lastTotalIdle = totalIdle;
        
//         return percent;
//     }
    
//     static std::pair<double, double> getGPUUsage() {
//         // Try to get GPU usage using nvidia-smi
//         FILE* pipe = popen("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null", "r");
//         if (!pipe) return {0.0, 0.0};
        
//         char buffer[128];
//         std::string result = "";
//         while (fgets(buffer, sizeof buffer, pipe) != nullptr) {
//             result += buffer;
//         }
//         pclose(pipe);
        
//         if (result.empty()) return {0.0, 0.0};
        
//         double gpu_util = 0.0, gpu_mem = 0.0;
//         std::sscanf(result.c_str(), "%lf, %lf", &gpu_util, &gpu_mem);
//         return {gpu_util, gpu_mem};
//     }
    
//     static double getSystemMemoryUsage() {
//         std::ifstream file("/proc/meminfo");
//         std::string line;
//         unsigned long memTotal = 0, memFree = 0, buffers = 0, cached = 0;
        
//         while (std::getline(file, line)) {
//             if (line.find("MemTotal:") == 0) {
//                 std::sscanf(line.c_str(), "MemTotal: %lu kB", &memTotal);
//             } else if (line.find("MemFree:") == 0) {
//                 std::sscanf(line.c_str(), "MemFree: %lu kB", &memFree);
//             } else if (line.find("Buffers:") == 0) {
//                 std::sscanf(line.c_str(), "Buffers: %lu kB", &buffers);
//             } else if (line.find("Cached:") == 0) {
//                 std::sscanf(line.c_str(), "Cached: %lu kB", &cached);
//             }
//         }
        
//         double usedMB = (memTotal - memFree - buffers - cached) / 1024.0;
//         return usedMB;
//     }
// };

// // Memory measurement utility
// double getCurrentMemoryUsageMB() {
//     struct rusage usage;
//     getrusage(RUSAGE_SELF, &usage);
//     return usage.ru_maxrss / 1024.0; // Convert KB to MB on Linux
// }

// // Enhanced detector factory supporting YOLO11 (extensible for future models)
// class DetectorFactory {
// public:
//     static std::unique_ptr<YOLO11Detector> createDetector(const BenchmarkConfig& config) {
//         if (config.model_type == "yolo11" && config.task_type == "detection") {
//             return std::make_unique<YOLO11Detector>(config.model_path, config.labels_path, config.use_gpu);
//         }
//         // TODO: Add support for YOLO8 and other models with proper abstraction
//         // For now, also support yolo8 calls by creating YOLO11 detector (for compatibility)
//         else if (config.model_type == "yolo8" && config.task_type == "detection") {
//             std::cout << "Note: Using YOLO11 detector for YOLO8 model (compatibility mode)" << std::endl;
//             return std::make_unique<YOLO11Detector>(config.model_path, config.labels_path, config.use_gpu);
//         }
//         else {
//             throw std::runtime_error("Unsupported model type: " + config.model_type + " with task: " + config.task_type);
//         }
//     }
    
//     static std::vector<Detection> detect(YOLO11Detector* detector, const BenchmarkConfig& config, const cv::Mat& image) {
//         return detector->detect(image);
//     }
// };

// // Enhanced image benchmark with comprehensive system monitoring
// PerformanceMetrics benchmark_image_comprehensive(const BenchmarkConfig& config,
//                                                 const std::string& image_path,
//                                                 int iterations = 100) {
//     PerformanceMetrics metrics;
//     metrics.environment_type = config.use_gpu ? "GPU" : "CPU";
    
//     // Measure model loading time
//     auto load_start = std::chrono::high_resolution_clock::now();
//     auto detector = DetectorFactory::createDetector(config);
//     auto load_end = std::chrono::high_resolution_clock::now();
//     metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    
//     // Load test image
//     cv::Mat image = cv::imread(image_path);
//     if (image.empty()) {
//         throw std::runtime_error("Could not read image: " + image_path);
//     }
    
//     std::vector<double> preprocess_times, inference_times, postprocess_times, total_times, latency_times;
    
//     // Warm-up runs
//     for (int i = 0; i < 10; ++i) {
//         DetectorFactory::detect(detector.get(), config, image);
//     }
    
//     // Measure initial system state
//     double initial_memory = getCurrentMemoryUsageMB();
//     double initial_sys_memory = SystemMonitor::getSystemMemoryUsage();
//     SystemMonitor::getCPUUsage(); // Initialize CPU monitoring
    
//     // Enhanced benchmark runs with detailed timing and system monitoring
//     std::vector<double> cpu_usage_samples, gpu_usage_samples, gpu_memory_samples;
    
//     for (int i = 0; i < iterations; ++i) {
//         // Monitor system resources during benchmark
//         double cpu_usage = SystemMonitor::getCPUUsage();
//         auto gpu_stats = SystemMonitor::getGPUUsage();
        
//         cpu_usage_samples.push_back(cpu_usage);
//         gpu_usage_samples.push_back(gpu_stats.first);
//         gpu_memory_samples.push_back(gpu_stats.second);
        
//         // Time the detection with high precision
//         cv::TickMeter tm;
//         tm.start();
        
//         auto total_start = std::chrono::high_resolution_clock::now();
//         auto infer_start = std::chrono::high_resolution_clock::now();
        
//         auto results = DetectorFactory::detect(detector.get(), config, image);
        
//         auto infer_end = std::chrono::high_resolution_clock::now();
//         auto total_end = std::chrono::high_resolution_clock::now();
        
//         tm.stop();
        
//         double infer_time = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
//         double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
//         double latency = tm.getTimeMilli();
        
//         preprocess_times.push_back(0.0);  // Placeholder - can be enhanced with separate timing
//         inference_times.push_back(infer_time);
//         postprocess_times.push_back(0.0); // Placeholder
//         total_times.push_back(total_time);
//         latency_times.push_back(latency);
//     }
    
//     // Calculate final memory usage
//     double final_memory = getCurrentMemoryUsageMB();
//     double final_sys_memory = SystemMonitor::getSystemMemoryUsage();
//     metrics.memory_mb = final_memory - initial_memory;
//     metrics.system_memory_used_mb = final_sys_memory - initial_sys_memory;
    
//     // Calculate comprehensive statistics
//     auto calc_avg = [](const std::vector<double>& values) {
//         return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
//     };
    
//     auto calc_min_max = [](const std::vector<double>& values) {
//         auto minmax = std::minmax_element(values.begin(), values.end());
//         return std::make_pair(*minmax.first, *minmax.second);
//     };
    
//     metrics.preprocess_avg_ms = calc_avg(preprocess_times);
//     metrics.inference_avg_ms = calc_avg(inference_times);
//     metrics.postprocess_avg_ms = calc_avg(postprocess_times);
//     metrics.total_avg_ms = calc_avg(total_times);
//     metrics.fps = 1000.0 / metrics.total_avg_ms;
    
//     // Enhanced latency statistics
//     metrics.latency_avg_ms = calc_avg(latency_times);
//     auto latency_minmax = calc_min_max(latency_times);
//     metrics.latency_min_ms = latency_minmax.first;
//     metrics.latency_max_ms = latency_minmax.second;
    
//     // System resource statistics
//     metrics.cpu_usage_percent = calc_avg(cpu_usage_samples);
//     metrics.gpu_usage_percent = calc_avg(gpu_usage_samples);
//     metrics.gpu_memory_used_mb = calc_avg(gpu_memory_samples);
    
//     return metrics;
// }

// // Enhanced video benchmark with comprehensive monitoring
// PerformanceMetrics benchmark_video_comprehensive(const BenchmarkConfig& config,
//                                                const std::string& video_path) {
//     PerformanceMetrics metrics;
//     metrics.environment_type = config.use_gpu ? "GPU" : "CPU";
    
//     // Measure model loading time
//     auto load_start = std::chrono::high_resolution_clock::now();
//     auto detector = DetectorFactory::createDetector(config);
//     auto load_end = std::chrono::high_resolution_clock::now();
//     metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    
//     // Open video
//     cv::VideoCapture cap(video_path);
//     if (!cap.isOpened()) {
//         throw std::runtime_error("Could not open video: " + video_path);
//     }
    
//     double initial_memory = getCurrentMemoryUsageMB();
//     double initial_sys_memory = SystemMonitor::getSystemMemoryUsage();
//     SystemMonitor::getCPUUsage(); // Initialize CPU monitoring
    
//     auto start_time = std::chrono::high_resolution_clock::now();
//     int frame_count = 0;
//     cv::Mat frame;
    
//     std::vector<double> frame_times, latency_times;
//     std::vector<double> cpu_usage_samples, gpu_usage_samples, gpu_memory_samples;
    
//     while (cap.read(frame)) {
//         // Monitor system resources
//         double cpu_usage = SystemMonitor::getCPUUsage();
//         auto gpu_stats = SystemMonitor::getGPUUsage();
        
//         cpu_usage_samples.push_back(cpu_usage);
//         gpu_usage_samples.push_back(gpu_stats.first);
//         gpu_memory_samples.push_back(gpu_stats.second);
        
//         // Time frame processing
//         cv::TickMeter tm;
//         tm.start();
        
//         auto frame_start = std::chrono::high_resolution_clock::now();
//         auto results = DetectorFactory::detect(detector.get(), config, frame);
//         auto frame_end = std::chrono::high_resolution_clock::now();
        
//         tm.stop();
        
//         double frame_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
//         double latency = tm.getTimeMilli();
        
//         frame_times.push_back(frame_time);
//         latency_times.push_back(latency);
        
//         frame_count++;
//     }
    
//     auto end_time = std::chrono::high_resolution_clock::now();
//     double total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
//     double final_memory = getCurrentMemoryUsageMB();
//     double final_sys_memory = SystemMonitor::getSystemMemoryUsage();
//     metrics.memory_mb = final_memory - initial_memory;
//     metrics.system_memory_used_mb = final_sys_memory - initial_sys_memory;
    
//     // Calculate comprehensive statistics
//     auto calc_avg = [](const std::vector<double>& values) {
//         return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
//     };
    
//     auto calc_min_max = [](const std::vector<double>& values) {
//         auto minmax = std::minmax_element(values.begin(), values.end());
//         return std::make_pair(*minmax.first, *minmax.second);
//     };
    
//     metrics.frame_count = frame_count;
//     metrics.total_avg_ms = calc_avg(frame_times);
//     metrics.fps = (frame_count * 1000.0) / total_time;
    
//     // Enhanced latency statistics
//     metrics.latency_avg_ms = calc_avg(latency_times);
//     auto latency_minmax = calc_min_max(latency_times);
//     metrics.latency_min_ms = latency_minmax.first;
//     metrics.latency_max_ms = latency_minmax.second;
    
//     // System resource statistics
//     metrics.cpu_usage_percent = calc_avg(cpu_usage_samples);
//     metrics.gpu_usage_percent = calc_avg(gpu_usage_samples);
//     metrics.gpu_memory_used_mb = calc_avg(gpu_memory_samples);
    
//     return metrics;
// }

// // Enhanced camera benchmark for real-time performance with system monitoring
// PerformanceMetrics benchmark_camera_comprehensive(const BenchmarkConfig& config,
//                                                 int camera_id = 0,
//                                                 int duration_seconds = 30) {
//     PerformanceMetrics metrics;
//     metrics.environment_type = config.use_gpu ? "GPU" : "CPU";
    
//     auto load_start = std::chrono::high_resolution_clock::now();
//     auto detector = DetectorFactory::createDetector(config);
//     auto load_end = std::chrono::high_resolution_clock::now();
//     metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    
//     cv::VideoCapture cap(camera_id);
//     if (!cap.isOpened()) {
//         throw std::runtime_error("Could not open camera: " + std::to_string(camera_id));
//     }
    
//     cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
//     cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
//     cap.set(cv::CAP_PROP_FPS, 30);
    
//     double initial_memory = getCurrentMemoryUsageMB();
//     double initial_sys_memory = SystemMonitor::getSystemMemoryUsage();
//     SystemMonitor::getCPUUsage(); // Initialize CPU monitoring
    
//     auto start_time = std::chrono::steady_clock::now();
//     auto end_time = start_time + std::chrono::seconds(duration_seconds);
    
//     int frame_count = 0;
//     std::vector<double> frame_times, latency_times;
//     std::vector<double> cpu_usage_samples, gpu_usage_samples, gpu_memory_samples;
//     cv::Mat frame;
    
//     while (std::chrono::steady_clock::now() < end_time && cap.read(frame)) {
//         // Monitor system resources
//         double cpu_usage = SystemMonitor::getCPUUsage();
//         auto gpu_stats = SystemMonitor::getGPUUsage();
        
//         cpu_usage_samples.push_back(cpu_usage);
//         gpu_usage_samples.push_back(gpu_stats.first);
//         gpu_memory_samples.push_back(gpu_stats.second);
        
//         // Time frame processing
//         cv::TickMeter tm;
//         tm.start();
        
//         auto frame_start = std::chrono::high_resolution_clock::now();
//         auto results = DetectorFactory::detect(detector.get(), config, frame);
//         auto frame_finish = std::chrono::high_resolution_clock::now();
        
//         tm.stop();
        
//         double frame_time = std::chrono::duration<double, std::milli>(frame_finish - frame_start).count();
//         double latency = tm.getTimeMilli();
        
//         frame_times.push_back(frame_time);
//         latency_times.push_back(latency);
        
//         frame_count++;
//     }
    
//     auto actual_end = std::chrono::steady_clock::now();
//     double actual_duration = std::chrono::duration<double>(actual_end - start_time).count();
    
//     double final_memory = getCurrentMemoryUsageMB();
//     double final_sys_memory = SystemMonitor::getSystemMemoryUsage();
//     metrics.memory_mb = final_memory - initial_memory;
//     metrics.system_memory_used_mb = final_sys_memory - initial_sys_memory;
    
//     // Calculate comprehensive statistics
//     auto calc_avg = [](const std::vector<double>& values) {
//         return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
//     };
    
//     auto calc_min_max = [](const std::vector<double>& values) {
//         auto minmax = std::minmax_element(values.begin(), values.end());
//         return std::make_pair(*minmax.first, *minmax.second);
//     };
    
//     metrics.frame_count = frame_count;
//     metrics.total_avg_ms = calc_avg(frame_times);
//     metrics.fps = frame_count / actual_duration;
    
//     // Enhanced latency statistics
//     metrics.latency_avg_ms = calc_avg(latency_times);
//     auto latency_minmax = calc_min_max(latency_times);
//     metrics.latency_min_ms = latency_minmax.first;
//     metrics.latency_max_ms = latency_minmax.second;
    
//     // System resource statistics
//     metrics.cpu_usage_percent = calc_avg(cpu_usage_samples);
//     metrics.gpu_usage_percent = calc_avg(gpu_usage_samples);
//     metrics.gpu_memory_used_mb = calc_avg(gpu_memory_samples);
    
//     return metrics;
// }

// // Enhanced CSV output utilities with comprehensive metrics
// void printCSVHeader() {
//     std::cout << "model_type,task_type,environment,device,threads,precision,load_ms,preprocess_ms,inference_ms,postprocess_ms,total_ms,fps,memory_mb,system_memory_mb,cpu_usage_%,gpu_usage_%,gpu_memory_mb,latency_avg_ms,latency_min_ms,latency_max_ms,map_score,frame_count\n";
// }

// void printCSVRow(const BenchmarkConfig& config, const PerformanceMetrics& metrics) {
//     std::cout << config.model_type << ","
//               << config.task_type << ","
//               << metrics.environment_type << ","
//               << (config.use_gpu ? "gpu" : "cpu") << ","
//               << config.thread_count << ","
//               << config.precision << ","
//               << std::fixed << std::setprecision(3)
//               << metrics.load_time_ms << ","
//               << metrics.preprocess_avg_ms << ","
//               << metrics.inference_avg_ms << ","
//               << metrics.postprocess_avg_ms << ","
//               << metrics.total_avg_ms << ","
//               << metrics.fps << ","
//               << metrics.memory_mb << ","
//               << metrics.system_memory_used_mb << ","
//               << metrics.cpu_usage_percent << ","
//               << metrics.gpu_usage_percent << ","
//               << metrics.gpu_memory_used_mb << ","
//               << metrics.latency_avg_ms << ","
//               << metrics.latency_min_ms << ","
//               << metrics.latency_max_ms << ","
//               << metrics.map_score << ","
//               << metrics.frame_count << std::endl;
// }

// // Save results to file for easier analysis
// void saveResultsToFile(const std::string& filename, const std::vector<std::pair<BenchmarkConfig, PerformanceMetrics>>& results) {
//     std::ofstream file(filename);
//     if (!file.is_open()) {
//         std::cerr << "Failed to open file: " << filename << std::endl;
//         return;
//     }
    
//     // Redirect cout to file temporarily
//     std::streambuf* orig = std::cout.rdbuf();
//     std::cout.rdbuf(file.rdbuf());
    
//     printCSVHeader();
//     for (const auto& [config, metrics] : results) {
//         printCSVRow(config, metrics);
//     }
    
//     // Restore cout
//     std::cout.rdbuf(orig);
//     file.close();
    
//     std::cout << "Results saved to: " << filename << std::endl;
// }

// // Configuration parsing
// BenchmarkConfig parseConfig(int argc, char** argv) {
//     if (argc < 6) {
//         std::cerr << "Usage: " << argv[0] << " <mode> <model_type> <task_type> <model_path> <labels_path> <input_path> [options]\n"
//                 << "Modes: image, video, camera\n"
//                 << "Model types: yolo5, yolo7, yolo8, yolo9, yolo10, yolo11, yolo12\n"
//                 << "Task types: detection, segmentation, obb, pose\n"
//                 << "Options: --gpu, --cpu, --threads=N, --quantized, --iterations=N\n";
//         throw std::runtime_error("Invalid arguments");
//     }
    
//     BenchmarkConfig config;
//     config.model_type = argv[2];
//     config.task_type = argv[3];
//     config.model_path = argv[4];
//     config.labels_path = argv[5];
    
//     // Parse optional arguments
//     for (int i = 7; i < argc; ++i) {
//         std::string arg = argv[i];
//         if (arg == "--gpu" || arg == "gpu") {
//             config.use_gpu = true;
//         } else if (arg == "--cpu" || arg == "cpu") {
//             config.use_gpu = false;
//         } else if (arg.substr(0, 10) == "--threads=") {
//             config.thread_count = std::stoi(arg.substr(10));
//         } else if (arg == "--quantized") {
//             config.quantized = true;
//             config.precision = "int8";
//         }
//     }
    
//     return config;
// }



// int main(int argc, char** argv) {
//     try {
//         if (argc < 2) {
//             std::cerr << "Usage: " << argv[0] << " <mode> <model_type> <task_type> <model_path> <labels_path> <input_path> [options]\n"
//                       << "Modes: image, video, camera, comprehensive\n"
//                       << "Model types: yolo5, yolo7, yolo8, yolo9, yolo10, yolo11, yolo12\n"
//                       << "Task types: detection, segmentation, obb, pose\n"
//                       << "Options: --gpu, --cpu, --threads=N, --quantized, --iterations=N, --duration=N\n"
//                       << "\nExamples:\n"
//                       << "  " << argv[0] << " image yolo11 detection model.onnx labels.txt test.jpg --gpu\n"
//                       << "  " << argv[0] << " video yolo8 segmentation model.onnx labels.txt test.mp4 --cpu --threads=4\n"
//                       << "  " << argv[0] << " camera yolo11 pose model.onnx labels.txt 0 --gpu --duration=30\n"
//                       << "  " << argv[0] << " comprehensive  # Run all supported combinations\n";
//             return 1;
//         }

//         std::string mode = argv[1];
        
//         if (mode == "comprehensive") {
//             // Enhanced comprehensive benchmark across all supported configurations
//             std::cout << "Running comprehensive YOLOs-CPP benchmark with system monitoring...\n";
//             printCSVHeader();
            
//             // Enhanced test configurations focusing on v8 and v11 detection as requested
//             std::vector<std::tuple<std::string, std::string, std::string>> test_configs = {
//                 {"yolo8", "detection", "../models/yolo8n.onnx"},
//                 {"yolo8", "detection", "../models/yolo8s.onnx"},
//                 {"yolo8", "detection", "../models/yolo8m.onnx"},
//                 {"yolo11", "detection", "../models/yolo11n.onnx"},
//                 {"yolo11", "detection", "../models/yolo11s.onnx"},
//                 {"yolo11", "detection", "../models/yolo11m.onnx"},
//                 // Future extensibility - these will be skipped if models don't exist
//                 {"yolo5", "detection", "../models/yolo5n.onnx"},
//                 {"yolo7", "detection", "../models/yolo7-tiny.onnx"},
//                 {"yolo8", "segmentation", "../models/yolo8n-seg.onnx"},
//                 {"yolo11", "segmentation", "../models/yolo11n-seg.onnx"},
//             };
            
//             std::vector<bool> gpu_configs = {false, true};
//             std::vector<int> thread_configs = {1, 4, 8};
//             std::vector<int> iteration_configs = {50, 100, 200}; // Different iteration counts
            
//             std::vector<std::pair<BenchmarkConfig, PerformanceMetrics>> all_results;
            
//             for (const auto& [model_type, task_type, model_path] : test_configs) {
//                 // Check if model file exists
//                 if (!std::ifstream(model_path).good()) {
//                     std::cerr << "Skipping " << model_type << "/" << task_type << " - model not found: " << model_path << "\n";
//                     continue;
//                 }
                
//                 for (bool use_gpu : gpu_configs) {
//                     for (int threads : thread_configs) {
//                         if (use_gpu && threads > 1) continue; // GPU doesn't use thread config
                        
//                         for (int iterations : iteration_configs) {
//                             BenchmarkConfig config;
//                             config.model_type = model_type;
//                             config.task_type = task_type;
//                             config.model_path = model_path;
//                             config.labels_path = "../models/coco.names";
//                             config.use_gpu = use_gpu;
//                             config.thread_count = threads;
                            
//                             try {
//                                 std::cout << "Testing " << model_type << "/" << task_type 
//                                           << " on " << (use_gpu ? "GPU" : "CPU") 
//                                           << " with " << threads << " threads, " << iterations << " iterations...\n";
                                
//                                 // Run enhanced image benchmark with specified iterations
//                                 auto metrics = benchmark_image_comprehensive(config, "../data/dog.jpg", iterations);
//                                 printCSVRow(config, metrics);
                                
//                                 all_results.push_back({config, metrics});
                                
//                                 // Add small delay to prevent system overload
//                                 std::this_thread::sleep_for(std::chrono::milliseconds(500));
                                
//                             } catch (const std::exception& e) {
//                                 std::cerr << "Error benchmarking " << model_type << "/" << task_type 
//                                           << " on " << (use_gpu ? "GPU" : "CPU") << ": " << e.what() << "\n";
//                             }
//                         }
//                     }
//                 }
//             }
            
//             // Save results to file for later analysis
//             std::string timestamp = std::to_string(std::time(nullptr));
//             std::string results_filename = "benchmark_results_" + timestamp + ".csv";
//             saveResultsToFile(results_filename, all_results);
            
//             return 0;
//         }
        
//         // Parse configuration for single benchmark
//         BenchmarkConfig config = parseConfig(argc, argv);
//         std::string input_path = argv[6];
        
//         int iterations = 100;
//         int duration = 30;
        
//         // Parse additional options
//         for (int i = 7; i < argc; ++i) {
//             std::string arg = argv[i];
//             if (arg.substr(0, 13) == "--iterations=") {
//                 iterations = std::stoi(arg.substr(13));
//             } else if (arg.substr(0, 11) == "--duration=") {
//                 duration = std::stoi(arg.substr(11));
//             }
//         }
        
//         printCSVHeader();
        
//         PerformanceMetrics metrics;
        
//         if (mode == "image") {
//             metrics = benchmark_image_comprehensive(config, input_path, iterations);
//         } else if (mode == "video") {
//             metrics = benchmark_video_comprehensive(config, input_path);
//         } else if (mode == "camera") {
//             int camera_id = std::stoi(input_path);
//             metrics = benchmark_camera_comprehensive(config, camera_id, duration);
//         } else {
//             std::cerr << "Error: Invalid mode '" << mode << "'. Use 'image', 'video', 'camera', or 'comprehensive'.\n";
//             return 1;
//         }
        
//         printCSVRow(config, metrics);
        
//     } catch (const std::exception& e) {
//         std::cerr << "Exception: " << e.what() << "\n";
//         return 1;
//     }
    
//     return 0;
// }



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
#include <filesystem>
#include <stdexcept>
#include <sstream>

#include <opencv2/opencv.hpp>

// Include specific YOLO detectors based on availability
#include "det/YOLO11.hpp"
// TODO: Implement proper abstraction layer for multiple models
// #include "det/YOLO8.hpp"

#include "tools/ScopedTimer.hpp"

// System monitoring includes
#include <sys/stat.h>
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
    
    // New fields for enhanced configuration
    int batch_size = 1;
    std::string input_resolution = "640x640";
    float confidence_threshold = 0.25f;
    float nms_threshold = 0.45f;
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
    double latency_std_ms = 0.0;  // Added standard deviation
    double latency_p95_ms = 0.0;  // Added 95th percentile
    double latency_p99_ms = 0.0;  // Added 99th percentile
    std::string environment_type = "CPU"; // "CPU" or "GPU"
    
    // Error handling metrics
    int failed_detections = 0;
    int total_detections = 0;
};

// System monitoring utilities with improved error handling
class SystemMonitor {
private:
    static bool gpu_available;
    static bool gpu_checked;
    
public:
    static double getCPUUsage() {
        static unsigned long long lastTotalUser = 0, lastTotalUserLow = 0, lastTotalSys = 0, lastTotalIdle = 0;
        
        std::ifstream file("/proc/stat");
        if (!file.is_open()) {
            return 0.0;  // Fallback for non-Linux systems
        }
        
        std::string line;
        if (!std::getline(file, line)) {
            return 0.0;
        }
        
        unsigned long long totalUser, totalUserLow, totalSys, totalIdle, total;
        if (std::sscanf(line.c_str(), "cpu %llu %llu %llu %llu", &totalUser, &totalUserLow, &totalSys, &totalIdle) != 4) {
            return 0.0;
        }
        
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
        if (total > 0) {
            percent = (percent / total) * 100.0;
        } else {
            percent = 0.0;
        }
        
        lastTotalUser = totalUser;
        lastTotalUserLow = totalUserLow;
        lastTotalSys = totalSys;
        lastTotalIdle = totalIdle;
        
        return percent;
    }
    
    static std::tuple<double, double, double> getGPUUsage() {
        if (!gpu_checked) {
            gpu_available = checkGPUAvailable();
            gpu_checked = true;
        }
        
        if (!gpu_available) {
            return {0.0, 0.0, 0.0};
        }
        
        // Try to get GPU usage using nvidia-smi
        FILE* pipe = popen("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null", "r");
        if (!pipe) return {0.0, 0.0, 0.0};
        
        char buffer[256];
        std::string result = "";
        while (fgets(buffer, sizeof buffer, pipe) != nullptr) {
            result += buffer;
        }
        pclose(pipe);
        
        if (result.empty()) return {0.0, 0.0, 0.0};
        
        double gpu_util = 0.0, gpu_mem_used = 0.0, gpu_mem_total = 0.0;
        if (std::sscanf(result.c_str(), "%lf, %lf, %lf", &gpu_util, &gpu_mem_used, &gpu_mem_total) == 3) {
            return {gpu_util, gpu_mem_used, gpu_mem_total};
        }
        
        return {0.0, 0.0, 0.0};
    }
    
    static double getSystemMemoryUsage() {
        std::ifstream file("/proc/meminfo");
        if (!file.is_open()) {
            return 0.0;  // Fallback for non-Linux systems
        }
        
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
        
        if (memTotal > 0) {
            double usedMB = (memTotal - memFree - buffers - cached) / 1024.0;
            return usedMB;
        }
        return 0.0;
    }
    
private:
    static bool checkGPUAvailable() {
        FILE* pipe = popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null", "r");
        if (!pipe) return false;
        
        char buffer[128];
        bool has_output = fgets(buffer, sizeof buffer, pipe) != nullptr;
        pclose(pipe);
        
        return has_output;
    }
};

bool SystemMonitor::gpu_available = false;
bool SystemMonitor::gpu_checked = false;

// Memory measurement utility with error handling
double getCurrentMemoryUsageMB() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        #ifdef __APPLE__
            return usage.ru_maxrss / (1024.0 * 1024.0); // macOS uses bytes
        #else
            return usage.ru_maxrss / 1024.0; // Linux uses KB
        #endif
    }
    return 0.0;
}

// Enhanced statistics calculation utilities
class StatisticsCalculator {
public:
    static double calculateMean(const std::vector<double>& values) {
        if (values.empty()) return 0.0;
        return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    }
    
    static double calculateStdDev(const std::vector<double>& values, double mean) {
        if (values.size() < 2) return 0.0;
        
        double variance = 0.0;
        for (double value : values) {
            variance += (value - mean) * (value - mean);
        }
        variance /= (values.size() - 1);
        return std::sqrt(variance);
    }
    
    static double calculatePercentile(std::vector<double> values, double percentile) {
        if (values.empty()) return 0.0;
        
        std::sort(values.begin(), values.end());
        double index = (percentile / 100.0) * (values.size() - 1);
        
        if (index == std::floor(index)) {
            return values[static_cast<size_t>(index)];
        } else {
            size_t lower = static_cast<size_t>(std::floor(index));
            size_t upper = static_cast<size_t>(std::ceil(index));
            double weight = index - std::floor(index);
            return values[lower] * (1 - weight) + values[upper] * weight;
        }
    }
    
    static std::pair<double, double> calculateMinMax(const std::vector<double>& values) {
        if (values.empty()) return {0.0, 0.0};
        auto minmax = std::minmax_element(values.begin(), values.end());
        return {*minmax.first, *minmax.second};
    }
};

// Enhanced detector factory with better error handling
class DetectorFactory {
public:
    static std::unique_ptr<YOLO11Detector> createDetector(const BenchmarkConfig& config) {
        // Validate model file exists
        if (!std::filesystem::exists(config.model_path)) {
            throw std::runtime_error("Model file not found: " + config.model_path);
        }
        
        // Validate labels file exists
        if (!std::filesystem::exists(config.labels_path)) {
            throw std::runtime_error("Labels file not found: " + config.labels_path);
        }
        
        if (config.model_type == "yolo11" && config.task_type == "detection") {
            return std::make_unique<YOLO11Detector>(config.model_path, config.labels_path, config.use_gpu);
        }
        // Support for YOLO8 using YOLO11 detector (compatibility mode)
        else if (config.model_type == "yolo8" && config.task_type == "detection") {
            std::cout << "Note: Using YOLO11 detector for YOLO8 model (compatibility mode)" << std::endl;
            return std::make_unique<YOLO11Detector>(config.model_path, config.labels_path, config.use_gpu);
        }
        else {
            throw std::runtime_error("Unsupported model type: " + config.model_type + " with task: " + config.task_type);
        }
    }
    
    static std::vector<Detection> detect(YOLO11Detector* detector, const BenchmarkConfig& config, const cv::Mat& image) {
        if (image.empty()) {
            throw std::runtime_error("Empty image provided for detection");
        }
        
        try {
            return detector->detect(image);
        } catch (const std::exception& e) {
            throw std::runtime_error("Detection failed: " + std::string(e.what()));
        }
    }
};

// Enhanced image benchmark with comprehensive system monitoring and better error handling
PerformanceMetrics benchmark_image_comprehensive(const BenchmarkConfig& config,
                                                const std::string& image_path,
                                                int iterations = 100) {
    PerformanceMetrics metrics;
    metrics.environment_type = config.use_gpu ? "GPU" : "CPU";
    
    // Validate input image
    if (!std::filesystem::exists(image_path)) {
        throw std::runtime_error("Image file not found: " + image_path);
    }
    
    // Measure model loading time
    auto load_start = std::chrono::high_resolution_clock::now();
    std::unique_ptr<YOLO11Detector> detector;
    try {
        detector = DetectorFactory::createDetector(config);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create detector: " + std::string(e.what()));
    }
    auto load_end = std::chrono::high_resolution_clock::now();
    metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    
    // Load test image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Could not read image: " + image_path);
    }
    
    std::vector<double> preprocess_times, inference_times, postprocess_times, total_times, latency_times;
    
    // Warm-up runs with error handling
    std::cout << "Performing warm-up runs..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        try {
            DetectorFactory::detect(detector.get(), config, image);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Warm-up run " << i << " failed: " << e.what() << std::endl;
        }
    }
    
    // Measure initial system state
    double initial_memory = getCurrentMemoryUsageMB();
    double initial_sys_memory = SystemMonitor::getSystemMemoryUsage();
    SystemMonitor::getCPUUsage(); // Initialize CPU monitoring
    
    // Enhanced benchmark runs with detailed timing and system monitoring
    std::vector<double> cpu_usage_samples, gpu_usage_samples, gpu_memory_samples;
    int successful_iterations = 0;
    int failed_iterations = 0;
    
    std::cout << "Running " << iterations << " benchmark iterations..." << std::endl;
    
    for (int i = 0; i < iterations; ++i) {
        try {
            // Monitor system resources during benchmark
            double cpu_usage = SystemMonitor::getCPUUsage();
            auto [gpu_util, gpu_mem_used, gpu_mem_total] = SystemMonitor::getGPUUsage();
            
            cpu_usage_samples.push_back(cpu_usage);
            gpu_usage_samples.push_back(gpu_util);
            gpu_memory_samples.push_back(gpu_mem_used);
            
            if (i == 0) {
                metrics.gpu_memory_total_mb = gpu_mem_total;
            }
            
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
            
            successful_iterations++;
            metrics.total_detections += results.size();
            
        } catch (const std::exception& e) {
            std::cerr << "Warning: Iteration " << i << " failed: " << e.what() << std::endl;
            failed_iterations++;
            metrics.failed_detections++;
        }
        
        // Progress indicator
        if ((i + 1) % (iterations / 10) == 0 || i == iterations - 1) {
            std::cout << "Completed " << (i + 1) << "/" << iterations << " iterations" << std::endl;
        }
    }
    
    if (successful_iterations == 0) {
        throw std::runtime_error("All benchmark iterations failed");
    }
    
    // Calculate final memory usage
    double final_memory = getCurrentMemoryUsageMB();
    double final_sys_memory = SystemMonitor::getSystemMemoryUsage();
    metrics.memory_mb = final_memory - initial_memory;
    metrics.system_memory_used_mb = final_sys_memory - initial_sys_memory;
    
    // Calculate comprehensive statistics
    metrics.preprocess_avg_ms = StatisticsCalculator::calculateMean(preprocess_times);
    metrics.inference_avg_ms = StatisticsCalculator::calculateMean(inference_times);
    metrics.postprocess_avg_ms = StatisticsCalculator::calculateMean(postprocess_times);
    metrics.total_avg_ms = StatisticsCalculator::calculateMean(total_times);
    metrics.fps = 1000.0 / metrics.total_avg_ms;
    
    // Enhanced latency statistics
    metrics.latency_avg_ms = StatisticsCalculator::calculateMean(latency_times);
    metrics.latency_std_ms = StatisticsCalculator::calculateStdDev(latency_times, metrics.latency_avg_ms);
    auto [latency_min, latency_max] = StatisticsCalculator::calculateMinMax(latency_times);
    metrics.latency_min_ms = latency_min;
    metrics.latency_max_ms = latency_max;
    metrics.latency_p95_ms = StatisticsCalculator::calculatePercentile(latency_times, 95.0);
    metrics.latency_p99_ms = StatisticsCalculator::calculatePercentile(latency_times, 99.0);
    
    // System resource statistics
    metrics.cpu_usage_percent = StatisticsCalculator::calculateMean(cpu_usage_samples);
    metrics.gpu_usage_percent = StatisticsCalculator::calculateMean(gpu_usage_samples);
    metrics.gpu_memory_used_mb = StatisticsCalculator::calculateMean(gpu_memory_samples);
    
    metrics.frame_count = successful_iterations;
    
    std::cout << "Benchmark completed: " << successful_iterations << " successful, " 
              << failed_iterations << " failed iterations" << std::endl;
    
    return metrics;
}

// Enhanced video benchmark with comprehensive monitoring and error handling
PerformanceMetrics benchmark_video_comprehensive(const BenchmarkConfig& config,
                                               const std::string& video_path) {
    PerformanceMetrics metrics;
    metrics.environment_type = config.use_gpu ? "GPU" : "CPU";
    
    // Validate input video
    if (!std::filesystem::exists(video_path)) {
        throw std::runtime_error("Video file not found: " + video_path);
    }
    
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
    
    // Get video properties
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Processing video: " << total_frames << " frames at " << video_fps << " FPS" << std::endl;
    
    double initial_memory = getCurrentMemoryUsageMB();
    double initial_sys_memory = SystemMonitor::getSystemMemoryUsage();
    SystemMonitor::getCPUUsage(); // Initialize CPU monitoring
    
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    int failed_frames = 0;
    cv::Mat frame;
    
    std::vector<double> frame_times, latency_times;
    std::vector<double> cpu_usage_samples, gpu_usage_samples, gpu_memory_samples;
    
    while (cap.read(frame)) {
        try {
            // Monitor system resources
            double cpu_usage = SystemMonitor::getCPUUsage();
            auto [gpu_util, gpu_mem_used, gpu_mem_total] = SystemMonitor::getGPUUsage();
            
            cpu_usage_samples.push_back(cpu_usage);
            gpu_usage_samples.push_back(gpu_util);
            gpu_memory_samples.push_back(gpu_mem_used);
            
            if (frame_count == 0) {
                metrics.gpu_memory_total_mb = gpu_mem_total;
            }
            
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
            
            metrics.total_detections += results.size();
            frame_count++;
            
        } catch (const std::exception& e) {
            std::cerr << "Warning: Frame " << frame_count << " processing failed: " << e.what() << std::endl;
            failed_frames++;
            metrics.failed_detections++;
        }
        
        // Progress indicator
        if (total_frames > 0 && frame_count % std::max(1, total_frames / 20) == 0) {
            std::cout << "Processed " << frame_count << "/" << total_frames << " frames" << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    if (frame_count == 0) {
        throw std::runtime_error("No frames were successfully processed");
    }
    
    double final_memory = getCurrentMemoryUsageMB();
    double final_sys_memory = SystemMonitor::getSystemMemoryUsage();
    metrics.memory_mb = final_memory - initial_memory;
    metrics.system_memory_used_mb = final_sys_memory - initial_sys_memory;
    
    // Calculate comprehensive statistics
    metrics.frame_count = frame_count;
    metrics.total_avg_ms = StatisticsCalculator::calculateMean(frame_times);
    metrics.fps = (frame_count * 1000.0) / total_time;
    
    // Enhanced latency statistics
    metrics.latency_avg_ms = StatisticsCalculator::calculateMean(latency_times);
    metrics.latency_std_ms = StatisticsCalculator::calculateStdDev(latency_times, metrics.latency_avg_ms);
    auto [latency_min, latency_max] = StatisticsCalculator::calculateMinMax(latency_times);
    metrics.latency_min_ms = latency_min;
    metrics.latency_max_ms = latency_max;
    metrics.latency_p95_ms = StatisticsCalculator::calculatePercentile(latency_times, 95.0);
    metrics.latency_p99_ms = StatisticsCalculator::calculatePercentile(latency_times, 99.0);
    
    // System resource statistics
    metrics.cpu_usage_percent = StatisticsCalculator::calculateMean(cpu_usage_samples);
    metrics.gpu_usage_percent = StatisticsCalculator::calculateMean(gpu_usage_samples);
    metrics.gpu_memory_used_mb = StatisticsCalculator::calculateMean(gpu_memory_samples);
    
    std::cout << "Video processing completed: " << frame_count << " successful, " 
              << failed_frames << " failed frames" << std::endl;
    
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
    
    // Set camera properties with error checking
    if (!cap.set(cv::CAP_PROP_FRAME_WIDTH, 640) ||
        !cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480) ||
        !cap.set(cv::CAP_PROP_FPS, 30)) {
        std::cerr << "Warning: Could not set all camera properties" << std::endl;
    }
    
    // Verify actual camera settings
    double actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double actual_fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Camera settings: " << actual_width << "x" << actual_height 
              << " at " << actual_fps << " FPS" << std::endl;
    
    double initial_memory = getCurrentMemoryUsageMB();
    double initial_sys_memory = SystemMonitor::getSystemMemoryUsage();
    SystemMonitor::getCPUUsage(); // Initialize CPU monitoring
    
    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::seconds(duration_seconds);
    
    int frame_count = 0;
    int failed_frames = 0;
    std::vector<double> frame_times, latency_times;
    std::vector<double> cpu_usage_samples, gpu_usage_samples, gpu_memory_samples;
    cv::Mat frame;
    
    std::cout << "Starting " << duration_seconds << " second camera benchmark..." << std::endl;
    
    while (std::chrono::steady_clock::now() < end_time) {
        if (!cap.read(frame)) {
            std::cerr << "Warning: Failed to read frame from camera" << std::endl;
            failed_frames++;
            continue;
        }
        
        try {
            // Monitor system resources
            double cpu_usage = SystemMonitor::getCPUUsage();
            auto [gpu_util, gpu_mem_used, gpu_mem_total] = SystemMonitor::getGPUUsage();
            
            cpu_usage_samples.push_back(cpu_usage);
            gpu_usage_samples.push_back(gpu_util);
            gpu_memory_samples.push_back(gpu_mem_used);
            
            if (frame_count == 0) {
                metrics.gpu_memory_total_mb = gpu_mem_total;
            }
            
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
            
            metrics.total_detections += results.size();
            frame_count++;
            
        } catch (const std::exception& e) {
            std::cerr << "Warning: Frame " << frame_count << " processing failed: " << e.what() << std::endl;
            failed_frames++;
            metrics.failed_detections++;
        }
        
        // Progress indicator every 5 seconds
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        if (elapsed > 0 && elapsed % 5 == 0 && frame_count > 0) {
            static int last_reported = -1;
            if (elapsed != last_reported) {
                std::cout << "Elapsed: " << elapsed << "s, Frames: " << frame_count 
                          << ", FPS: " << frame_count / elapsed << std::endl;
                last_reported = elapsed;
            }
        }
    }
    
    auto actual_end = std::chrono::steady_clock::now();
    double actual_duration = std::chrono::duration<double>(actual_end - start_time).count();
    
    if (frame_count == 0) {
        throw std::runtime_error("No frames were successfully processed");
    }
    
    double final_memory = getCurrentMemoryUsageMB();
    double final_sys_memory = SystemMonitor::getSystemMemoryUsage();
    metrics.memory_mb = final_memory - initial_memory;
    metrics.system_memory_used_mb = final_sys_memory - initial_sys_memory;
    
    // Calculate comprehensive statistics
    metrics.frame_count = frame_count;
    metrics.total_avg_ms = StatisticsCalculator::calculateMean(frame_times);
    metrics.fps = frame_count / actual_duration;
    
    // Enhanced latency statistics
    metrics.latency_avg_ms = StatisticsCalculator::calculateMean(latency_times);
    metrics.latency_std_ms = StatisticsCalculator::calculateStdDev(latency_times, metrics.latency_avg_ms);
    auto [latency_min, latency_max] = StatisticsCalculator::calculateMinMax(latency_times);
    metrics.latency_min_ms = latency_min;
    metrics.latency_max_ms = latency_max;
    metrics.latency_p95_ms = StatisticsCalculator::calculatePercentile(latency_times, 95.0);
    metrics.latency_p99_ms = StatisticsCalculator::calculatePercentile(latency_times, 99.0);
    
    // System resource statistics
    metrics.cpu_usage_percent = StatisticsCalculator::calculateMean(cpu_usage_samples);
    metrics.gpu_usage_percent = StatisticsCalculator::calculateMean(gpu_usage_samples);
    metrics.gpu_memory_used_mb = StatisticsCalculator::calculateMean(gpu_memory_samples);
    
    std::cout << "Camera processing completed: " << frame_count << " successful, " 
              << failed_frames << " failed frames" << std::endl;
    
    return metrics;
}

// Enhanced CSV output utilities with comprehensive metrics
void printCSVHeader() {
    std::cout << "model_type,task_type,environment,device,threads,precision,load_ms,preprocess_ms,inference_ms,postprocess_ms,total_ms,fps,memory_mb,system_memory_mb,cpu_usage_%,gpu_usage_%,gpu_memory_mb,latency_avg_ms,latency_min_ms,latency_max_ms,latency_std_ms,latency_p95_ms,latency_p99_ms,map_score,frame_count,failed_detections,total_detections\n";
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
              << metrics.latency_std_ms << ","
              << metrics.latency_p95_ms << ","
              << metrics.latency_p99_ms << ","
              << metrics.map_score << ","
              << metrics.frame_count << ","
              << metrics.failed_detections << ","
              << metrics.total_detections << std::endl;
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
            
            // Create results directory
            std::filesystem::create_directories("results");
            
            // Test configurations focusing on available models
            std::vector<std::tuple<std::string, std::string, std::string>> test_configs = {
                {"yolo11", "detection", "models/yolo11n.onnx"},
                {"yolo8", "detection", "models/yolov8n.onnx"},
            };
            
            std::vector<bool> gpu_configs = {false, true};
            std::vector<int> thread_configs = {1, 4};
            std::vector<int> iteration_configs = {50, 100};
            
            std::vector<std::pair<BenchmarkConfig, PerformanceMetrics>> all_results;
            
            // Image benchmark results
            std::string image_results_file = "results/image_benchmark_" + std::to_string(std::time(nullptr)) + ".csv";
            std::ofstream image_file(image_results_file);
            if (image_file.is_open()) {
                std::streambuf* orig = std::cout.rdbuf();
                std::cout.rdbuf(image_file.rdbuf());
                printCSVHeader();
                std::cout.rdbuf(orig);
                image_file.close();
            }
            
            // Video benchmark results
            std::string video_results_file = "results/video_benchmark_" + std::to_string(std::time(nullptr)) + ".csv";
            std::ofstream video_file(video_results_file);
            if (video_file.is_open()) {
                std::streambuf* orig = std::cout.rdbuf();
                std::cout.rdbuf(video_file.rdbuf());
                printCSVHeader();
                std::cout.rdbuf(orig);
                video_file.close();
            }
            
            std::cout << "Starting comprehensive benchmark...\n";
            
            for (const auto& [model_type, task_type, model_path] : test_configs) {
                // Check if model file exists
                if (!std::filesystem::exists(model_path)) {
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
                            config.labels_path = "models/coco.names";
                            config.use_gpu = use_gpu;
                            config.thread_count = threads;
                            
                            try {
                                std::cout << "Testing " << model_type << "/" << task_type 
                                          << " on " << (use_gpu ? "GPU" : "CPU") 
                                          << " with " << threads << " threads, " << iterations << " iterations...\n";
                                
                                // Run image benchmark
                                auto image_metrics = benchmark_image_comprehensive(config, "data/dog.jpg", iterations);
                                
                                // Append to image results file
                                std::ofstream img_file(image_results_file, std::ios::app);
                                if (img_file.is_open()) {
                                    std::streambuf* orig = std::cout.rdbuf();
                                    std::cout.rdbuf(img_file.rdbuf());
                                    printCSVRow(config, image_metrics);
                                    std::cout.rdbuf(orig);
                                    img_file.close();
                                }
                                
                                // Run video benchmark
                                if (std::filesystem::exists("data/dogs.mp4")) {
                                    auto video_metrics = benchmark_video_comprehensive(config, "data/dogs.mp4");
                                    
                                    // Append to video results file
                                    std::ofstream vid_file(video_results_file, std::ios::app);
                                    if (vid_file.is_open()) {
                                        std::streambuf* orig = std::cout.rdbuf();
                                        std::cout.rdbuf(vid_file.rdbuf());
                                        printCSVRow(config, video_metrics);
                                        std::cout.rdbuf(orig);
                                        vid_file.close();
                                    }
                                }
                                
                                all_results.push_back({config, image_metrics});
                                
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
            
            std::cout << "Comprehensive benchmark completed!\n";
            std::cout << "Results saved to:\n";
            std::cout << "- " << image_results_file << "\n";
            std::cout << "- " << video_results_file << "\n";
            
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