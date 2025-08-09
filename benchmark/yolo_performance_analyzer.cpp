#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <cstdio>
#include <ctime>

#include <opencv2/opencv.hpp>

// Project headers
#include "det/YOLO11.hpp"
#include "tools/ScopedTimer.hpp"

// ---------- Logging control ----------
#ifdef DEBUG
  #define DBG_STDOUT 1
#else
  #define DBG_STDOUT 0
#endif

#if DBG_STDOUT
  #define DEBUG_LOG(x) do { std::cout << x; } while(0)
#else
  #define DEBUG_LOG(x) do {} while(0)
#endif

#ifdef _WIN32
  #define NOMINMAX
  #include <windows.h>
  #include <psapi.h>
  #pragma comment(lib, "psapi.lib")
#else
  #include <sys/resource.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

struct BenchmarkConfig {
  std::string model_type;
  std::string task_type;
  std::string model_path;
  std::string labels_path;
  bool use_gpu = false;
  int thread_count = 1;
  bool quantized = false;
  std::string precision = "fp32";
};

struct PerformanceMetrics {
  double load_time_ms = 0.0;
  double preprocess_avg_ms = 0.0;
  double inference_avg_ms = 0.0;
  double postprocess_avg_ms = 0.0;
  double total_avg_ms = 0.0;
  double fps = 0.0;
  double memory_mb = 0.0;
  double map_score = 0.0;
  int frame_count = 0;

  // Extra monitoring
  double cpu_usage_percent = 0.0;
  double gpu_usage_percent = 0.0;
  double gpu_memory_used_mb = 0.0;
  double gpu_memory_total_mb = 0.0;
  double system_memory_used_mb = 0.0;
  double latency_avg_ms = 0.0;
  double latency_min_ms = 0.0;
  double latency_max_ms = 0.0;
  std::string environment_type = "CPU";
};

// ----------------- Monitoring -----------------
struct SystemMonitor {
  static double getCPUUsage() {
#ifdef _WIN32
    return 0.0;
#else
    static unsigned long long lastUser = 0, lastUserLow = 0, lastSys = 0, lastIdle = 0;
    std::ifstream file("/proc/stat");
    std::string line; if (!std::getline(file, line)) return 0.0;
    unsigned long long user, userLow, sys, idle;
    if (std::sscanf(line.c_str(), "cpu %llu %llu %llu %llu", &user, &userLow, &sys, &idle) != 4) return 0.0;
    if (lastUser == 0) { lastUser = user; lastUserLow = userLow; lastSys = sys; lastIdle = idle; return 0.0; }
    auto total = (user - lastUser) + (userLow - lastUserLow) + (sys - lastSys);
    double percent = static_cast<double>(total);
    total += (idle - lastIdle);
    if (total == 0) return 0.0;
    percent = (percent / total) * 100.0;
    lastUser = user; lastUserLow = userLow; lastSys = sys; lastIdle = idle;
    return percent;
#endif
  }

  static std::pair<double, double> getGPUUsage() {
#ifdef _WIN32
    return {0.0, 0.0};
#else
    FILE* pipe = popen("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null", "r");
    if (!pipe) return {0.0, 0.0};
    char buffer[256]; std::string result;
    while (fgets(buffer, sizeof buffer, pipe) != nullptr) result += buffer;
    pclose(pipe);
    if (result.empty()) return {0.0, 0.0};
    double gpu_util = 0.0, gpu_mem = 0.0;
    if (std::sscanf(result.c_str(), "%lf, %lf", &gpu_util, &gpu_mem) != 2) return {0.0, 0.0};
    return {gpu_util, gpu_mem};
#endif
  }

  static double getSystemMemoryUsage() {
#ifdef _WIN32
    MEMORYSTATUSEX statex{}; statex.dwLength = sizeof(statex);
    if (!GlobalMemoryStatusEx(&statex)) return 0.0;
    DWORDLONG total = statex.ullTotalPhys, avail = statex.ullAvailPhys;
    if (total < avail) return 0.0;
    return static_cast<double>(total - avail) / (1024.0 * 1024.0);
#else
    std::ifstream file("/proc/meminfo"); std::string line;
    unsigned long memTotal = 0, memFree = 0, buffers = 0, cached = 0;
    while (std::getline(file, line)) {
      if (line.rfind("MemTotal:", 0) == 0) std::sscanf(line.c_str(), "MemTotal: %lu kB", &memTotal);
      else if (line.rfind("MemFree:", 0) == 0) std::sscanf(line.c_str(), "MemFree: %lu kB", &memFree);
      else if (line.rfind("Buffers:", 0) == 0) std::sscanf(line.c_str(), "Buffers: %lu kB", &buffers);
      else if (line.rfind("Cached:", 0) == 0) std::sscanf(line.c_str(), "Cached: %lu kB", &cached);
    }
    return static_cast<double>(memTotal - memFree - buffers - cached) / 1024.0;
#endif
  }
};

double getCurrentMemoryUsageMB() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS_EX pmc{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
    return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
  }
  return 0.0;
#else
  struct rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
  }
  return 0.0;
#endif
}

// ----------------- Detector wrapper -----------------
class DetectorFactory {
public:
  static std::unique_ptr<YOLO11Detector> createDetector(const BenchmarkConfig& config) {
    bool is_quantized = config.model_path.find("quantized") != std::string::npos;

    if (config.model_type == "yolo11" && config.task_type == "detection") {
      if (is_quantized) DEBUG_LOG("Note: Testing YOLO11 quantized model (smaller size)\n");
      return std::make_unique<YOLO11Detector>(config.model_path, config.labels_path, config.use_gpu);
    } else if (config.model_type == "yolo8" && config.task_type == "detection") {
      if (is_quantized) DEBUG_LOG("Note: Testing YOLO8 quantized model (smaller size)\n");
      else DEBUG_LOG("Note: Using YOLO11 detector for YOLO8 model (compat mode)\n");
      return std::make_unique<YOLO11Detector>(config.model_path, config.labels_path, config.use_gpu);
    } else if (config.model_type == "yolo11_quantized" && config.task_type == "detection") {
      DEBUG_LOG("Note: Testing YOLO11 quantized model (smaller size)\n");
      return std::make_unique<YOLO11Detector>(config.model_path, config.labels_path, config.use_gpu);
    } else if (config.model_type == "yolo8_quantized" && config.task_type == "detection") {
      DEBUG_LOG("Note: Testing YOLO8 quantized model (smaller size)\n");
      return std::make_unique<YOLO11Detector>(config.model_path, config.labels_path, config.use_gpu);
    }

    throw std::runtime_error("Unsupported model type: " + config.model_type + " with task: " + config.task_type);
  }

  static std::vector<Detection> detect(YOLO11Detector* detector, const BenchmarkConfig&, const cv::Mat& image) {
    static int call_count = 0; call_count++;
    if (call_count <= 3) {
      DEBUG_LOG("  Processing frame " << call_count << " | Input: " << image.cols << "x" << image.rows
                 << " -> Model Input: 640x640\n");
    }
    return detector->detect(image);
  }
};

// ----------------- Bench: Image -----------------
PerformanceMetrics benchmark_image_comprehensive(const BenchmarkConfig& config,
                                                 const std::string& image_path,
                                                 int iterations = 100) {
  PerformanceMetrics metrics;
  metrics.environment_type = config.use_gpu ? "GPU" : "CPU";

  auto load_start = std::chrono::high_resolution_clock::now();
  auto detector   = DetectorFactory::createDetector(config);
  auto load_end   = std::chrono::high_resolution_clock::now();
  metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

  cv::Mat image = cv::imread(image_path);
  if (image.empty()) throw std::runtime_error("Could not read image: " + image_path);

  std::vector<double> preprocess_times, inference_times, postprocess_times, total_times, latency_times;

  for (int i = 0; i < 10; ++i) DetectorFactory::detect(detector.get(), config, image); // warmup

  double initial_memory = getCurrentMemoryUsageMB();
  double initial_sys_memory = SystemMonitor::getSystemMemoryUsage();
  SystemMonitor::getCPUUsage();

  std::vector<double> cpu_samples, gpu_samples, gpu_mem_samples;

  for (int i = 0; i < iterations; ++i) {
    double cpu = SystemMonitor::getCPUUsage();
    auto gpu   = SystemMonitor::getGPUUsage();
    cpu_samples.push_back(cpu);
    gpu_samples.push_back(gpu.first);
    gpu_mem_samples.push_back(gpu.second);

    cv::TickMeter tm; tm.start();
    auto total_start = std::chrono::high_resolution_clock::now();
    auto infer_start = std::chrono::high_resolution_clock::now();
    auto results     = DetectorFactory::detect(detector.get(), config, image); (void)results;
    auto infer_end   = std::chrono::high_resolution_clock::now();
    auto total_end   = std::chrono::high_resolution_clock::now();
    tm.stop();

    double infer_ms  = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    double total_ms  = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    double latency   = tm.getTimeMilli();

    preprocess_times.push_back(0.0);
    inference_times.push_back(infer_ms);
    postprocess_times.push_back(0.0);
    total_times.push_back(total_ms);
    latency_times.push_back(latency);
  }

  double final_memory = getCurrentMemoryUsageMB();
  double final_sys_memory = SystemMonitor::getSystemMemoryUsage();
  metrics.memory_mb = final_memory - initial_memory;
  metrics.system_memory_used_mb = final_sys_memory - initial_sys_memory;

  auto avg    = [](const std::vector<double>& v){ return v.empty()? 0.0 : std::accumulate(v.begin(), v.end(), 0.0)/v.size(); };
  auto minmax = [](const std::vector<double>& v){ if (v.empty()) return std::make_pair(0.0,0.0); auto mm = std::minmax_element(v.begin(), v.end()); return std::make_pair(*mm.first, *mm.second); };

  metrics.preprocess_avg_ms = avg(preprocess_times);
  metrics.inference_avg_ms  = avg(inference_times);
  metrics.postprocess_avg_ms= avg(postprocess_times);
  metrics.total_avg_ms      = avg(total_times);
  metrics.fps               = (metrics.total_avg_ms > 0.0) ? (1000.0 / metrics.total_avg_ms) : 0.0;

  metrics.latency_avg_ms = avg(latency_times);
  auto lat = minmax(latency_times);
  metrics.latency_min_ms = lat.first;
  metrics.latency_max_ms = lat.second;

  metrics.cpu_usage_percent = avg(cpu_samples);
  metrics.gpu_usage_percent = avg(gpu_samples);
  metrics.gpu_memory_used_mb= avg(gpu_mem_samples);

  metrics.frame_count = iterations;
  return metrics;
}

// ----------------- Bench: Video file -----------------
PerformanceMetrics benchmark_video_comprehensive(const BenchmarkConfig& config,
                                                 const std::string& video_path) {
  PerformanceMetrics metrics;
  metrics.environment_type = config.use_gpu ? "GPU" : "CPU";

  auto load_start = std::chrono::high_resolution_clock::now();
  auto detector   = DetectorFactory::createDetector(config);
  auto load_end   = std::chrono::high_resolution_clock::now();
  metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) throw std::runtime_error("Could not open video: " + video_path);

  std::vector<double> frame_times, latency_times;
  std::vector<double> cpu_samples, gpu_samples, gpu_mem_samples;

  double initial_memory = getCurrentMemoryUsageMB();
  double initial_sys_memory = SystemMonitor::getSystemMemoryUsage();
  SystemMonitor::getCPUUsage();

  auto start_time = std::chrono::high_resolution_clock::now();
  int frame_count = 0;
  cv::Mat frame;

  while (cap.read(frame) && !frame.empty()) {
    double cpu = SystemMonitor::getCPUUsage();
    auto gpu   = SystemMonitor::getGPUUsage();
    cpu_samples.push_back(cpu);
    gpu_samples.push_back(gpu.first);
    gpu_mem_samples.push_back(gpu.second);

    cv::TickMeter tm; tm.start();
    auto frame_start = std::chrono::high_resolution_clock::now();
    auto results     = DetectorFactory::detect(detector.get(), config, frame); (void)results;
    auto frame_end   = std::chrono::high_resolution_clock::now();
    tm.stop();

    double frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
    double latency  = tm.getTimeMilli();

    frame_times.push_back(frame_ms);
    latency_times.push_back(latency);
    frame_count++;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

  double final_memory = getCurrentMemoryUsageMB();
  double final_sys_memory = SystemMonitor::getSystemMemoryUsage();
  metrics.memory_mb = final_memory - initial_memory;
  metrics.system_memory_used_mb = final_sys_memory - initial_sys_memory;

  auto avg    = [](const std::vector<double>& v){ return v.empty()? 0.0 : std::accumulate(v.begin(), v.end(), 0.0)/v.size(); };
  auto minmax = [](const std::vector<double>& v){ if (v.empty()) return std::make_pair(0.0,0.0); auto mm = std::minmax_element(v.begin(), v.end()); return std::make_pair(*mm.first, *mm.second); };

  metrics.frame_count     = frame_count;
  metrics.total_avg_ms    = avg(frame_times);
  metrics.fps             = (total_time_ms > 0.0) ? ((frame_count * 1000.0) / total_time_ms) : 0.0;
  metrics.latency_avg_ms  = avg(latency_times);
  auto lat                = minmax(latency_times);
  metrics.latency_min_ms  = lat.first;
  metrics.latency_max_ms  = lat.second;
  metrics.cpu_usage_percent = avg(cpu_samples);
  metrics.gpu_usage_percent = avg(gpu_samples);
  metrics.gpu_memory_used_mb= avg(gpu_mem_samples);

  return metrics;
}

// ----------------- Bench: Camera -----------------
PerformanceMetrics benchmark_camera_comprehensive(const BenchmarkConfig& config,
                                                  int camera_id = 0,
                                                  int duration_seconds = 30) {
  PerformanceMetrics metrics;
  metrics.environment_type = config.use_gpu ? "GPU" : "CPU";

  auto load_start = std::chrono::high_resolution_clock::now();
  auto detector   = DetectorFactory::createDetector(config);
  auto load_end   = std::chrono::high_resolution_clock::now();
  metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

  cv::VideoCapture cap(camera_id);
  if (!cap.isOpened()) throw std::runtime_error("Could not open camera with ID: " + std::to_string(camera_id));

  std::vector<double> frame_times, latency_times;
  std::vector<double> cpu_samples, gpu_samples, gpu_mem_samples;

  double initial_memory = getCurrentMemoryUsageMB();
  double initial_sys_memory = SystemMonitor::getSystemMemoryUsage();
  SystemMonitor::getCPUUsage();

  auto start_time = std::chrono::high_resolution_clock::now();
  auto end_target = start_time + std::chrono::seconds(duration_seconds);
  int frame_count = 0;
  cv::Mat frame;

  while (std::chrono::high_resolution_clock::now() < end_target) {
    if (!cap.read(frame) || frame.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }

    double cpu = SystemMonitor::getCPUUsage();
    auto gpu   = SystemMonitor::getGPUUsage();
    cpu_samples.push_back(cpu);
    gpu_samples.push_back(gpu.first);
    gpu_mem_samples.push_back(gpu.second);

    cv::TickMeter tm; tm.start();
    auto frame_start = std::chrono::high_resolution_clock::now();
    auto results     = DetectorFactory::detect(detector.get(), config, frame); (void)results;
    auto frame_end   = std::chrono::high_resolution_clock::now();
    tm.stop();

    double frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
    double latency  = tm.getTimeMilli();

    frame_times.push_back(frame_ms);
    latency_times.push_back(latency);
    frame_count++;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

  double final_memory = getCurrentMemoryUsageMB();
  double final_sys_memory = SystemMonitor::getSystemMemoryUsage();
  metrics.memory_mb = final_memory - initial_memory;
  metrics.system_memory_used_mb = final_sys_memory - initial_sys_memory;

  auto avg    = [](const std::vector<double>& v){ return v.empty()? 0.0 : std::accumulate(v.begin(), v.end(), 0.0)/v.size(); };
  auto minmax = [](const std::vector<double>& v){ if (v.empty()) return std::make_pair(0.0,0.0); auto mm = std::minmax_element(v.begin(), v.end()); return std::make_pair(*mm.first, *mm.second); };

  metrics.frame_count     = frame_count;
  metrics.total_avg_ms    = avg(frame_times);
  metrics.fps             = (total_time_ms > 0.0) ? ((frame_count * 1000.0) / total_time_ms) : 0.0;
  metrics.latency_avg_ms  = avg(latency_times);
  auto lat                = minmax(latency_times);
  metrics.latency_min_ms  = lat.first;
  metrics.latency_max_ms  = lat.second;
  metrics.cpu_usage_percent = avg(cpu_samples);
  metrics.gpu_usage_percent = avg(gpu_samples);
  metrics.gpu_memory_used_mb= avg(gpu_mem_samples);

  return metrics;
}

// ----------------- CSV helpers -----------------
static inline void printCSVHeader() {
  std::cout
    << "model_type,task_type,InputType,environment,device,threads,precision,"
    << "load_ms,preprocess_ms,inference_ms,postprocess_ms,total_ms,fps,"
    << "memory_mb,system_memory_mb,cpu_usage_%,gpu_usage_%,gpu_memory_mb,"
    << "latency_avg_ms,latency_min_ms,latency_max_ms,map_score,frame_count\n";
}

static inline void printCSVRow(const BenchmarkConfig& config,
                               const PerformanceMetrics& m,
                               const std::string& inputType) {
  std::cout << config.model_type << ","
            << config.task_type << ","
            << inputType << ","
            << m.environment_type << ","
            << (config.use_gpu ? "gpu" : "cpu") << ","
            << config.thread_count << ","
            << config.precision << ","
            << std::fixed << std::setprecision(3)
            << m.load_time_ms << ","
            << m.preprocess_avg_ms << ","
            << m.inference_avg_ms << ","
            << m.postprocess_avg_ms << ","
            << m.total_avg_ms << ","
            << m.fps << ","
            << m.memory_mb << ","
            << m.system_memory_used_mb << ","
            << m.cpu_usage_percent << ","
            << m.gpu_usage_percent << ","
            << m.gpu_memory_used_mb << ","
            << m.latency_avg_ms << ","
            << m.latency_min_ms << ","
            << m.latency_max_ms << ","
            << m.map_score << ","
            << m.frame_count << "\n";
}

// Append a single CSV row to a file
static inline void appendCSVRowToFile(const std::string& filePath,
                                      const BenchmarkConfig& cfg,
                                      const PerformanceMetrics& m,
                                      const std::string& inputType) {
  std::ofstream out(filePath, std::ios::app);
  if (!out) throw std::runtime_error("Cannot append to results file: " + filePath);
  out << cfg.model_type << ","
      << cfg.task_type << ","
      << inputType << ","
      << m.environment_type << ","
      << (cfg.use_gpu ? "gpu" : "cpu") << ","
      << cfg.thread_count << ","
      << cfg.precision << ","
      << std::fixed << std::setprecision(3)
      << m.load_time_ms << ","
      << m.preprocess_avg_ms << ","
      << m.inference_avg_ms << ","
      << m.postprocess_avg_ms << ","
      << m.total_avg_ms << ","
      << m.fps << ","
      << m.memory_mb << ","
      << m.system_memory_used_mb << ","
      << m.cpu_usage_percent << ","
      << m.gpu_usage_percent << ","
      << m.gpu_memory_used_mb << ","
      << m.latency_avg_ms << ","
      << m.latency_min_ms << ","
      << m.latency_max_ms << ","
      << m.map_score << ","
      << m.frame_count << "\n";
}

// ----------------- Arg parse -----------------
static BenchmarkConfig parseConfig(int argc, char** argv) {
  if (argc < 6) {
    std::cerr
      << "Usage: " << argv[0] << " <mode> <model_type> <task_type> <model_path> <labels_path> <input_path> [options]\n"
      << "Modes: image, video, camera, comprehensive\n"
      << "Model types: yolo5..yolo12\n"
      << "Task types: detection, segmentation, obb, pose\n"
      << "Options: --gpu, --cpu, --threads=N, --quantized, --iterations=N, --duration=N\n";
    throw std::runtime_error("Invalid arguments");
  }

  BenchmarkConfig cfg;
  cfg.model_type  = argv[2];
  cfg.task_type   = argv[3];
  cfg.model_path  = argv[4];
  cfg.labels_path = argv[5];

  for (int i = 7; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--gpu" || arg == "gpu") cfg.use_gpu = true;
    else if (arg == "--cpu" || arg == "cpu") cfg.use_gpu = false;
    else if (arg.rfind("--threads=", 0) == 0) cfg.thread_count = std::stoi(arg.substr(10));
    else if (arg == "--quantized") { cfg.quantized = true; cfg.precision = "int8"; }
  }
  return cfg;
}

// ----------------- main -----------------
int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr
        << "Usage: " << argv[0] << " <mode> <model_type> <task_type> <model_path> <labels_path> <input_path> [options]\n"
        << "Modes: image, video, camera, comprehensive\n"
        << "Options: --gpu, --cpu, --threads=N, --quantized, --iterations=N, --duration=N\n";
      return 1;
    }

    std::string mode = argv[1];

    if (mode == "comprehensive") {
      std::cout << "YOLO Performance Analyzer - running comprehensive benchmarks...\n";
      std::filesystem::create_directories("results");

      // candidate models (add freely here)
      std::vector<std::tuple<std::string,std::string,std::string>> test_configs = {
        {"yolo11", "detection", "models/yolo11n.onnx"},
        {"yolo8",  "detection", "models/yolov8n.onnx"},
        {"yolo11_quantized", "detection", "quantized_models/yolo11n_quantized.onnx"},
        {"yolo8_quantized",  "detection", "quantized_models/yolov8n_quantized.onnx"},
      };

      const std::string image_path = "data/dog.jpg";
      const std::string video_path = "data/dogs.mp4";
      const bool has_image = std::filesystem::exists(image_path);
      const bool has_video = std::filesystem::exists(video_path);

      const int image_iterations = 100; // single fixed iterations → no duplicate rows

      std::string results_file = "results/comprehensive_benchmark_" + std::to_string(std::time(nullptr)) + ".csv";
      {
        std::ofstream file(results_file);
        if (!file) throw std::runtime_error("Cannot open results file: " + results_file);
        file << "model_type,task_type,InputType,environment,device,threads,precision,"
                "load_ms,preprocess_ms,inference_ms,postprocess_ms,total_ms,fps,"
                "memory_mb,system_memory_mb,cpu_usage_%,gpu_usage_%,gpu_memory_mb,"
                "latency_avg_ms,latency_min_ms,latency_max_ms,map_score,frame_count\n";
      }

      // For each available model: run CPU(Image,Video) then GPU(Image,Video)
      for (const auto& [model_type, task_type, model_path] : test_configs) {
        if (!std::filesystem::exists(model_path)) {
          std::cerr << "Skipping " << model_type << "/" << task_type << " - model not found: " << model_path << "\n";
          continue;
        }

        for (bool use_gpu : {false, true}) {
          BenchmarkConfig cfg;
          cfg.model_type  = model_type;
          cfg.task_type   = task_type;
          cfg.model_path  = model_path;
          cfg.labels_path = "models/coco.names";
          cfg.use_gpu     = use_gpu;
          cfg.thread_count= 1;

          try {
            if (has_image) {
              auto m_img = benchmark_image_comprehensive(cfg, image_path, image_iterations);
              appendCSVRowToFile(results_file, cfg, m_img, "Image");
            }
            if (has_video) {
              auto m_vid = benchmark_video_comprehensive(cfg, video_path);
              appendCSVRowToFile(results_file, cfg, m_vid, "Video");
            }
            // small breather
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
          } catch (const std::exception& e) {
            std::cerr << "Error benchmarking " << model_type << "/" << task_type
                      << " on " << (use_gpu ? "GPU" : "CPU") << ": " << e.what() << "\n";
          }
        }
      }

      std::cout << "Comprehensive benchmark completed.\n";
      std::cout << "Results saved to: " << results_file << "\n";
      return 0;
    }

    // Single-run modes
    BenchmarkConfig cfg = parseConfig(argc, argv);
    std::string input_path = argv[6];

    int iterations = 100;
    int duration   = 30;
    for (int i = 7; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg.rfind("--iterations=", 0) == 0) iterations = std::stoi(arg.substr(13));
      else if (arg.rfind("--duration=", 0) == 0) duration = std::stoi(arg.substr(11));
    }

    // Print CSV header + one row to stdout
    printCSVHeader();

    PerformanceMetrics m;
    if (mode == "image") {
      m = benchmark_image_comprehensive(cfg, input_path, iterations);
      printCSVRow(cfg, m, "Image");
    } else if (mode == "video") {
      m = benchmark_video_comprehensive(cfg, input_path);
      printCSVRow(cfg, m, "Video");
    } else if (mode == "camera") {
      int cam_id = std::stoi(input_path);
      m = benchmark_camera_comprehensive(cfg, cam_id, duration);
      printCSVRow(cfg, m, "Video"); // treat camera as Video stream
    } else {
      std::cerr << "Error: invalid mode '" << mode << "'. Use image|video|camera|comprehensive.\n";
      return 1;
    }
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
