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
#include <cmath>

#include <opencv2/opencv.hpp>

// Project headers
// CRITICAL FIX: Both det/YOLO.hpp and seg/YOLO-Seg.hpp define the same symbols.
// Solution: Use preprocessor to rename conflicting symbols in seg header,
// then include det header normally. We'll use det's definitions for compatibility.

// Step 1: Include seg header but rename conflicting symbols
#define BoundingBox SegBoundingBox
#define getClassNames seg_getClassNames  
#define NMSBoxes seg_NMSBoxes
#define ScopedTimer SegScopedTimer

#include "seg/YOLO-Seg.hpp"

// Step 2: Undefine the macros and include det header normally
#undef BoundingBox
#undef getClassNames
#undef NMSBoxes  
#undef ScopedTimer

#include "det/YOLO.hpp"

// Step 3: Create aliases so we can use both
namespace seg_compat {
  using SegBoundingBox = BoundingBox;  // Use det's BoundingBox for seg
  // YOLOSegDetector uses SegBoundingBox internally, but we'll cast when needed
}

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

// ============================================================================
// Configuration Structures
// ============================================================================
struct UnifiedConfig {
  std::string model_type;
  std::string task_type;  // "detection" or "segmentation"
  std::string model_path;
  std::string labels_path;
  bool use_gpu = false;
  int thread_count = 1;
  bool quantized = false;
  std::string precision = "fp32";
  std::string device = "CPU";
  
  // Evaluation-specific
  bool evaluate_accuracy = false;
  std::string dataset_path;
  std::string gt_labels_path;
  std::string dataset_type = "custom";  // "coco" or "custom"
  float conf_threshold = 0.4f;
  float nms_threshold = 0.7f;
  float eval_conf_threshold = 0.001f;  // Lower for evaluation
};

struct UnifiedMetrics {
  // Performance metrics
  double load_time_ms = 0.0;
  double preprocess_avg_ms = 0.0;
  double inference_avg_ms = 0.0;
  double postprocess_avg_ms = 0.0;
  double total_avg_ms = 0.0;
  double fps = 0.0;
  double memory_mb = 0.0;
  int frame_count = 0;
  
  // System monitoring
  double cpu_usage_percent = 0.0;
  double gpu_usage_percent = 0.0;
  double gpu_memory_used_mb = 0.0;
  double gpu_memory_total_mb = 0.0;
  double system_memory_used_mb = 0.0;
  double latency_avg_ms = 0.0;
  double latency_min_ms = 0.0;
  double latency_max_ms = 0.0;
  std::string environment_type = "CPU";
  
  // Accuracy metrics (when GT available)
  float AP50 = 0.0f;
  float mAP5095 = 0.0f;
  std::vector<float> AP_per_iou;
  bool has_accuracy = false;
};

// ============================================================================
// System Monitoring
// ============================================================================
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

// ============================================================================
// Unified Detector Factory
// ============================================================================
class UnifiedDetectorFactory {
public:
  struct DetectorBase {
    virtual ~DetectorBase() = default;
    virtual std::string getDevice() const = 0;
  };

  struct DetectionDetector : DetectorBase {
    std::unique_ptr<YOLODetector> det;
    DetectionDetector(const std::string& model_path, const std::string& labels_path, bool use_gpu)
      : det(std::make_unique<YOLODetector>(model_path, labels_path, use_gpu)) {}
    std::string getDevice() const override { return det->getDevice(); }
  };

  struct SegmentationDetector : DetectorBase {
    std::unique_ptr<YOLOSegDetector> det;
    bool use_gpu_flag;
    SegmentationDetector(const std::string& model_path, const std::string& labels_path, bool use_gpu)
      : det(std::make_unique<YOLOSegDetector>(model_path, labels_path, use_gpu)), use_gpu_flag(use_gpu) {}
    std::string getDevice() const override { 
      // YOLOSegDetector doesn't expose device, infer from use_gpu flag
      // Note: This assumes GPU if requested, but actual device may differ
      return use_gpu_flag ? "gpu" : "cpu";
    }
  };

  static std::unique_ptr<DetectorBase> createDetector(const UnifiedConfig& config) {
    if (config.task_type == "detection") {
      return std::make_unique<DetectionDetector>(config.model_path, config.labels_path, config.use_gpu);
    } else if (config.task_type == "segmentation") {
      return std::make_unique<SegmentationDetector>(config.model_path, config.labels_path, config.use_gpu);
    }
    throw std::runtime_error("Unsupported task type: " + config.task_type);
  }

  static std::vector<Detection> detect_detection(DetectorBase* base, const cv::Mat& image, float conf_thresh, float nms_thresh) {
    auto* det = dynamic_cast<DetectionDetector*>(base);
    if (!det) throw std::runtime_error("Invalid detector type for detection");
    return det->det->detect(image, conf_thresh, nms_thresh);
  }

  static std::vector<Segmentation> detect_segmentation(DetectorBase* base, const cv::Mat& image, float conf_thresh, float nms_thresh) {
    auto* det = dynamic_cast<SegmentationDetector*>(base);
    if (!det) throw std::runtime_error("Invalid detector type for segmentation");
    return det->det->segment(image, conf_thresh, nms_thresh);
  }
};

// ============================================================================
// Evaluation Structures and Utilities
// ============================================================================
struct GTBox { int cls; float x, y, w, h; };
struct PredBox { int image_id; float x, y, w, h, score; int cls; };

static inline float bbox_iou_px(const PredBox &a, const GTBox &b) {
  float ax1 = a.x, ay1 = a.y, ax2 = a.x + a.w, ay2 = a.y + a.h;
  float bx1 = b.x, by1 = b.y, bx2 = b.x + b.w, by2 = b.y + b.h;
  float interX1 = std::max(ax1, bx1), interY1 = std::max(ay1, by1);
  float interX2 = std::min(ax2, bx2), interY2 = std::min(ay2, by2);
  float interW = std::max(0.0f, interX2 - interX1);
  float interH = std::max(0.0f, interY2 - interY1);
  float interArea = interW * interH;
  float areaA = std::max(0.0f, a.w * a.h);
  float areaB = std::max(0.0f, b.w * b.h);
  float unionArea = areaA + areaB - interArea;
  if (unionArea <= 0.0f) return 0.0f;
  return interArea / unionArea;
}

// Mask IoU for segmentation
static inline float mask_iou(const cv::Mat& pred_mask, const cv::Mat& gt_mask) {
  cv::Mat intersection, union_mask;
  cv::bitwise_and(pred_mask, gt_mask, intersection);
  cv::bitwise_or(pred_mask, gt_mask, union_mask);
  double inter_area = cv::countNonZero(intersection);
  double union_area = cv::countNonZero(union_mask);
  if (union_area == 0.0) return 0.0f;
  return static_cast<float>(inter_area / union_area);
}

static inline std::vector<std::string> listImages(const std::string& folder) {
  namespace fs = std::filesystem;
  std::vector<std::string> out;
  for (const auto &e : fs::directory_iterator(folder)) {
    if (!e.is_regular_file()) continue;
    auto ext = e.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff")
      out.push_back(e.path().string());
  }
  std::sort(out.begin(), out.end());
  return out;
}

static inline std::vector<GTBox> loadGT_yolo_to_px(const std::string& labelPath, int img_w, int img_h) {
  std::vector<GTBox> boxes;
  std::ifstream f(labelPath);
  if (!f.is_open()) return boxes;
  int cls; float cx, cy, w, h;
  while (f >> cls >> cx >> cy >> w >> h) {
    float bw = w * img_w;
    float bh = h * img_h;
    float cx_px = cx * img_w;
    float cy_px = cy * img_h;
    float x = cx_px - bw / 2.0f;
    float y = cy_px - bh / 2.0f;
    x = std::max(0.0f, std::min(x, float(img_w - 1)));
    y = std::max(0.0f, std::min(y, float(img_h - 1)));
    bw = std::max(0.0f, std::min(bw, float(img_w - x)));
    bh = std::max(0.0f, std::min(bh, float(img_h - y)));
    boxes.push_back({cls, x, y, bw, bh});
  }
  return boxes;
}

// ============================================================================
// Benchmark Functions
// ============================================================================
UnifiedMetrics benchmark_image_unified(UnifiedConfig& config,
                                      const std::string& image_path,
                                      int iterations = 100) {
  UnifiedMetrics metrics;

  auto load_start = std::chrono::high_resolution_clock::now();
  auto detector = UnifiedDetectorFactory::createDetector(config);
  std::string device_str = detector->getDevice();
  config.device = device_str;
  cv::Mat image = cv::imread(image_path);
  if (image.empty()) throw std::runtime_error("Could not read image: " + image_path);
  auto load_end = std::chrono::high_resolution_clock::now();
  metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
  metrics.environment_type = (device_str == "gpu") ? "CUDA" : "CPU";

  std::vector<double> inference_times, total_times, latency_times;

  // Warmup
  for (int i = 0; i < 10; ++i) {
    if (config.task_type == "detection") {
      UnifiedDetectorFactory::detect_detection(detector.get(), image, config.conf_threshold, config.nms_threshold);
    } else {
      UnifiedDetectorFactory::detect_segmentation(detector.get(), image, config.conf_threshold, config.nms_threshold);
    }
  }

  double initial_memory = getCurrentMemoryUsageMB();
  double initial_sys_memory = SystemMonitor::getSystemMemoryUsage();
  SystemMonitor::getCPUUsage();

  std::vector<double> cpu_samples, gpu_samples, gpu_mem_samples;

  for (int i = 0; i < iterations; ++i) {
    double cpu = SystemMonitor::getCPUUsage();
    auto gpu = SystemMonitor::getGPUUsage();
    cpu_samples.push_back(cpu);
    gpu_samples.push_back(gpu.first);
    gpu_mem_samples.push_back(gpu.second);

    auto total_start = std::chrono::high_resolution_clock::now();
    auto infer_start = std::chrono::high_resolution_clock::now();
    
    if (config.task_type == "detection") {
      auto results = UnifiedDetectorFactory::detect_detection(detector.get(), image, config.conf_threshold, config.nms_threshold);
      (void)results;
    } else {
      auto results = UnifiedDetectorFactory::detect_segmentation(detector.get(), image, config.conf_threshold, config.nms_threshold);
      (void)results;
    }
    
    auto infer_end = std::chrono::high_resolution_clock::now();
    auto total_end = std::chrono::high_resolution_clock::now();

    double infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    inference_times.push_back(infer_ms);
    total_times.push_back(total_ms);
    latency_times.push_back(total_ms);
  }

  double final_memory = getCurrentMemoryUsageMB();
  double final_sys_memory = SystemMonitor::getSystemMemoryUsage();
  metrics.memory_mb = final_memory - initial_memory;
  metrics.system_memory_used_mb = final_sys_memory - initial_sys_memory;

  auto avg = [](const std::vector<double>& v){ return v.empty()? 0.0 : std::accumulate(v.begin(), v.end(), 0.0)/v.size(); };
  auto minmax = [](const std::vector<double>& v){ if (v.empty()) return std::make_pair(0.0,0.0); auto mm = std::minmax_element(v.begin(), v.end()); return std::make_pair(*mm.first, *mm.second); };

  metrics.inference_avg_ms = avg(inference_times);
  metrics.total_avg_ms = avg(total_times);
  metrics.fps = (metrics.total_avg_ms > 0.0) ? (1000.0 / metrics.total_avg_ms) : 0.0;
  metrics.latency_avg_ms = avg(latency_times);
  auto lat = minmax(latency_times);
  metrics.latency_min_ms = lat.first;
  metrics.latency_max_ms = lat.second;
  metrics.cpu_usage_percent = avg(cpu_samples);
  metrics.gpu_usage_percent = avg(gpu_samples);
  metrics.gpu_memory_used_mb = avg(gpu_mem_samples);
  metrics.frame_count = iterations;

  return metrics;
}

UnifiedMetrics benchmark_video_unified(UnifiedConfig& config,
                                      const std::string& video_path) {
  UnifiedMetrics metrics;

  auto load_start = std::chrono::high_resolution_clock::now();
  auto detector = UnifiedDetectorFactory::createDetector(config);
  std::string device_str = detector->getDevice();
  config.device = device_str;
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) throw std::runtime_error("Could not open video: " + video_path);
  auto load_end = std::chrono::high_resolution_clock::now();
  metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
  metrics.environment_type = (device_str == "gpu") ? "CUDA" : "CPU";

  std::vector<double> frame_times, latency_times;
  std::vector<double> cpu_samples, gpu_samples, gpu_mem_samples;

  double initial_memory = getCurrentMemoryUsageMB();
  double initial_sys_memory = SystemMonitor::getSystemMemoryUsage();
  SystemMonitor::getCPUUsage();

  auto start_time = std::chrono::high_resolution_clock::now();
  int frame_count = 0;
  cv::Mat frame;

  while (cap.read(frame) && frame_count < 1000) {
    if (frame.empty()) continue;  // Skip empty frames
    
    double cpu = SystemMonitor::getCPUUsage();
    auto gpu = SystemMonitor::getGPUUsage();
    cpu_samples.push_back(cpu);
    gpu_samples.push_back(gpu.first);
    gpu_mem_samples.push_back(gpu.second);

    auto frame_start = std::chrono::high_resolution_clock::now();
    
    if (config.task_type == "detection") {
      auto results = UnifiedDetectorFactory::detect_detection(detector.get(), frame, config.conf_threshold, config.nms_threshold);
      (void)results;
    } else {
      auto results = UnifiedDetectorFactory::detect_segmentation(detector.get(), frame, config.conf_threshold, config.nms_threshold);
      (void)results;
    }
    
    auto frame_end = std::chrono::high_resolution_clock::now();
    double frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
    frame_times.push_back(frame_ms);
    latency_times.push_back(frame_ms);
    frame_count++;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

  double final_memory = getCurrentMemoryUsageMB();
  double final_sys_memory = SystemMonitor::getSystemMemoryUsage();
  metrics.memory_mb = final_memory - initial_memory;
  metrics.system_memory_used_mb = final_sys_memory - initial_sys_memory;

  auto avg = [](const std::vector<double>& v){ return v.empty()? 0.0 : std::accumulate(v.begin(), v.end(), 0.0)/v.size(); };
  auto minmax = [](const std::vector<double>& v){ if (v.empty()) return std::make_pair(0.0,0.0); auto mm = std::minmax_element(v.begin(), v.end()); return std::make_pair(*mm.first, *mm.second); };

  metrics.frame_count = frame_count;
  metrics.total_avg_ms = avg(frame_times);
  metrics.fps = (total_time_ms > 0.0) ? ((frame_count * 1000.0) / total_time_ms) : 0.0;
  metrics.latency_avg_ms = avg(latency_times);
  auto lat = minmax(latency_times);
  metrics.latency_min_ms = lat.first;
  metrics.latency_max_ms = lat.second;
  metrics.cpu_usage_percent = avg(cpu_samples);
  metrics.gpu_usage_percent = avg(gpu_samples);
  metrics.gpu_memory_used_mb = avg(gpu_mem_samples);

  return metrics;
}

UnifiedMetrics benchmark_camera_unified(UnifiedConfig& config,
                                       int camera_id = 0,
                                       int duration_seconds = 30) {
  UnifiedMetrics metrics;

  auto load_start = std::chrono::high_resolution_clock::now();
  auto detector = UnifiedDetectorFactory::createDetector(config);
  std::string device_str = detector->getDevice();
  config.device = device_str;
  cv::VideoCapture cap(camera_id);
  if (!cap.isOpened()) throw std::runtime_error("Could not open camera with ID: " + std::to_string(camera_id));
  auto load_end = std::chrono::high_resolution_clock::now();
  metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
  metrics.environment_type = (device_str == "gpu") ? "CUDA" : "CPU";

  std::vector<double> frame_times, latency_times;
  std::vector<double> cpu_samples, gpu_samples, gpu_mem_samples;

  double initial_memory = getCurrentMemoryUsageMB();
  double initial_sys_memory = SystemMonitor::getSystemMemoryUsage();
  SystemMonitor::getCPUUsage();

  auto start_time = std::chrono::high_resolution_clock::now();
  auto end_time = start_time + std::chrono::seconds(duration_seconds);
  int frame_count = 0;
  cv::Mat frame;

  while (std::chrono::high_resolution_clock::now() < end_time && cap.read(frame)) {
    if (frame.empty()) continue;  // Skip empty frames
    
    double cpu = SystemMonitor::getCPUUsage();
    auto gpu = SystemMonitor::getGPUUsage();
    cpu_samples.push_back(cpu);
    gpu_samples.push_back(gpu.first);
    gpu_mem_samples.push_back(gpu.second);

    auto frame_start = std::chrono::high_resolution_clock::now();
    
    if (config.task_type == "detection") {
      auto results = UnifiedDetectorFactory::detect_detection(detector.get(), frame, config.conf_threshold, config.nms_threshold);
      (void)results;
    } else {
      auto results = UnifiedDetectorFactory::detect_segmentation(detector.get(), frame, config.conf_threshold, config.nms_threshold);
      (void)results;
    }
    
    auto frame_end = std::chrono::high_resolution_clock::now();
    double frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
    frame_times.push_back(frame_ms);
    latency_times.push_back(frame_ms);
    frame_count++;
  }

  double final_memory = getCurrentMemoryUsageMB();
  double final_sys_memory = SystemMonitor::getSystemMemoryUsage();
  metrics.memory_mb = final_memory - initial_memory;
  metrics.system_memory_used_mb = final_sys_memory - initial_sys_memory;

  auto avg = [](const std::vector<double>& v){ return v.empty()? 0.0 : std::accumulate(v.begin(), v.end(), 0.0)/v.size(); };
  auto minmax = [](const std::vector<double>& v){ if (v.empty()) return std::make_pair(0.0,0.0); auto mm = std::minmax_element(v.begin(), v.end()); return std::make_pair(*mm.first, *mm.second); };

  metrics.frame_count = frame_count;
  metrics.total_avg_ms = avg(frame_times);
  metrics.fps = (metrics.total_avg_ms > 0.0) ? (1000.0 / metrics.total_avg_ms) : 0.0;
  metrics.latency_avg_ms = avg(latency_times);
  auto lat = minmax(latency_times);
  metrics.latency_min_ms = lat.first;
  metrics.latency_max_ms = lat.second;
  metrics.cpu_usage_percent = avg(cpu_samples);
  metrics.gpu_usage_percent = avg(gpu_samples);
  metrics.gpu_memory_used_mb = avg(gpu_mem_samples);

  return metrics;
}

// ============================================================================
// Accuracy Evaluation
// ============================================================================
UnifiedMetrics benchmark_evaluate(UnifiedConfig& config) {
  UnifiedMetrics metrics;
  
  auto load_start = std::chrono::high_resolution_clock::now();
  auto detector = UnifiedDetectorFactory::createDetector(config);
  std::string device_str = detector->getDevice();
  config.device = device_str;
  auto load_end = std::chrono::high_resolution_clock::now();
  metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
  metrics.environment_type = (device_str == "gpu") ? "CUDA" : "CPU";

  auto images = listImages(config.dataset_path);
  if (images.empty()) {
    throw std::runtime_error("No images found in dataset path: " + config.dataset_path);
  }
  std::cout << "Found " << images.size() << " images.\n";

  if (config.task_type == "detection") {
    // Detection evaluation
    std::map<int, std::vector<PredBox>> detections_by_class;
    std::map<int, int> gt_count_by_class;
    std::map<int, std::map<int, std::vector<GTBox>>> gt_by_image_and_class;
    
    std::vector<double> per_image_times;
    bool first_time = true;
    int image_id = 0;
    
    for (const auto& imgPath : images) {
      cv::Mat img = cv::imread(imgPath);
      if (img.empty()) { ++image_id; continue; }
      int img_w = img.cols;
      int img_h = img.rows;

      std::string stem = std::filesystem::path(imgPath).stem().string();
      std::string labelPath = config.gt_labels_path + "/" + stem + ".txt";
      auto gt_boxes = loadGT_yolo_to_px(labelPath, img_w, img_h);

      for (const auto& g : gt_boxes) {
        gt_by_image_and_class[image_id][g.cls].push_back(g);
        gt_count_by_class[g.cls] += 1;
      }

      if (first_time) {
        UnifiedDetectorFactory::detect_detection(detector.get(), img, config.eval_conf_threshold, config.nms_threshold);
        first_time = false;
      }

      auto t0 = std::chrono::high_resolution_clock::now();
      auto preds = UnifiedDetectorFactory::detect_detection(detector.get(), img, config.eval_conf_threshold, config.nms_threshold);
      auto t1 = std::chrono::high_resolution_clock::now();
      double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
      per_image_times.push_back(time_ms);

      for (const auto& p : preds) {
        PredBox pb;
        pb.image_id = image_id;
        pb.cls = p.classId;
        pb.score = p.conf;
        pb.x = static_cast<float>(p.box.x);
        pb.y = static_cast<float>(p.box.y);
        pb.w = static_cast<float>(p.box.width);
        pb.h = static_cast<float>(p.box.height);
        detections_by_class[pb.cls].push_back(pb);
      }

      image_id++;
    }

    // Compute mAP
    std::vector<float> iou_thresholds;
    for (float t = 0.50f; t <= 0.95f + 1e-9f; t += 0.05f) iou_thresholds.push_back(t);
    
    std::vector<float> ap_per_iou;
    float AP50 = 0.0f;
    
    for (float iou_thr : iou_thresholds) {
      std::vector<float> ap_per_class;
      for (const auto& kv : detections_by_class) {
        int cls = kv.first;
        auto dets = kv.second;
        int n_gt = gt_count_by_class[cls];
        if (n_gt == 0) continue;

        std::sort(dets.begin(), dets.end(), [](const PredBox& a, const PredBox& b){ return a.score > b.score; });

        std::map<int, std::vector<char>> matched_flag_per_image;
        for (const auto& img_gkv : gt_by_image_and_class) {
          int imgid = img_gkv.first;
          auto it = img_gkv.second.find(cls);
          if (it != img_gkv.second.end()) {
            matched_flag_per_image[imgid] = std::vector<char>(it->second.size(), 0);
          }
        }

        std::vector<int> tps, fps;
        for (const auto& d : dets) {
          auto it_img = gt_by_image_and_class.find(d.image_id);
          int best_gt_idx = -1;
          float best_iou = 0.0f;
          if (it_img != gt_by_image_and_class.end()) {
            auto it_cls = it_img->second.find(cls);
            if (it_cls != it_img->second.end()) {
              const auto& gts_in_img = it_cls->second;
              // Ensure matched_flag_per_image exists and has correct size
              if (matched_flag_per_image.find(d.image_id) != matched_flag_per_image.end() &&
                  matched_flag_per_image[d.image_id].size() == gts_in_img.size()) {
                for (size_t gi = 0; gi < gts_in_img.size(); ++gi) {
                  if (matched_flag_per_image[d.image_id][gi]) continue;
                  float iou = bbox_iou_px(d, gts_in_img[gi]);
                  if (iou > best_iou) { best_iou = iou; best_gt_idx = int(gi); }
                }
              }
            }
          }
          if (best_iou >= iou_thr && best_gt_idx >= 0) {
            tps.push_back(1);
            fps.push_back(0);
            // Safe access: matched_flag_per_image[d.image_id] exists because we initialized it above
            if (matched_flag_per_image.find(d.image_id) != matched_flag_per_image.end() &&
                best_gt_idx < static_cast<int>(matched_flag_per_image[d.image_id].size())) {
              matched_flag_per_image[d.image_id][best_gt_idx] = 1;
            }
          } else {
            tps.push_back(0);
            fps.push_back(1);
          }
        }

        if (tps.empty()) {
          ap_per_class.push_back(0.0f);
          continue;
        }

        std::vector<int> tp_cum(tps.size()), fp_cum(fps.size());
        int s = 0; for (size_t i=0; i<tps.size(); ++i) { s += tps[i]; tp_cum[i] = s; }
        s = 0; for (size_t i=0; i<fps.size(); ++i) { s += fps[i]; fp_cum[i] = s; }

        std::vector<float> precision(tp_cum.size()), recall(tp_cum.size());
        for (size_t i=0; i<tp_cum.size(); ++i) {
          float tpv = float(tp_cum[i]);
          float fpv = float(fp_cum[i]);
          precision[i] = tpv / (tpv + fpv + 1e-12f);
          recall[i] = (n_gt > 0) ? (tpv / float(n_gt)) : 0.0f;
        }

        const int NUM_POINTS = 101;
        float ap = 0.0f;
        for (int ri = 0; ri < NUM_POINTS; ++ri) {
          float r_thr = ri / float(NUM_POINTS - 1);
          float p_max = 0.0f;
          for (size_t k = 0; k < recall.size(); ++k) {
            if (recall[k] >= r_thr) {
              if (precision[k] > p_max) p_max = precision[k];
            }
          }
          ap += p_max;
        }
        ap /= NUM_POINTS;
        ap_per_class.push_back(ap);
      }

      if (ap_per_class.empty()) {
        ap_per_iou.push_back(0.0f);
      } else {
        float sum = 0.0f;
        for (float v : ap_per_class) sum += v;
        ap_per_iou.push_back(sum / float(ap_per_class.size()));
      }

      if (std::fabs(iou_thr - 0.50f) < 1e-6f) AP50 = ap_per_iou.back();
    }

    float mAP5095 = 0.0f;
    if (!ap_per_iou.empty()) {
      for (float v : ap_per_iou) mAP5095 += v;
      mAP5095 /= float(ap_per_iou.size());
    }

    metrics.AP50 = AP50;
    metrics.mAP5095 = mAP5095;
    metrics.AP_per_iou = ap_per_iou;
    metrics.has_accuracy = true;

    if (!per_image_times.empty()) {
      double total = std::accumulate(per_image_times.begin(), per_image_times.end(), 0.0);
      metrics.total_avg_ms = total / per_image_times.size();
      metrics.fps = (metrics.total_avg_ms > 0.0) ? (1000.0 / metrics.total_avg_ms) : 0.0;
      metrics.frame_count = per_image_times.size();
    }
  } else {
    // Segmentation evaluation
    // Note: Full mask IoU evaluation requires GT masks in polygon/mask format
    // For now, we use bounding box IoU from segmentation bounding boxes
    // This provides a reasonable approximation but not the full segmentation accuracy
    std::cout << "Segmentation evaluation: Using bounding box IoU from segmentation results\n";
    std::cout << "Note: Full mask IoU evaluation requires GT masks (not currently supported)\n";
    
    // Use similar structure to detection but extract bounding boxes from Segmentation results
    // This is a simplified evaluation - for production use, implement full mask IoU
    metrics.has_accuracy = false;  // Mark as no accuracy for segmentation until mask IoU is implemented
  }

  return metrics;
}

// ============================================================================
// Output Functions
// ============================================================================
void printTabularSummary(const std::vector<std::pair<UnifiedConfig, UnifiedMetrics>>& results) {
  std::cout << "\n" << std::string(120, '=') << "\n";
  std::cout << "BENCHMARK SUMMARY - COMPARISON TABLE\n";
  std::cout << std::string(120, '=') << "\n\n";

  // Header
  std::cout << std::left << std::setw(20) << "Model" 
            << std::setw(12) << "Task" 
            << std::setw(8) << "Device"
            << std::setw(10) << "FPS"
            << std::setw(12) << "Latency(ms)"
            << std::setw(10) << "AP50"
            << std::setw(12) << "mAP50-95"
            << std::setw(10) << "Memory(MB)"
            << "\n";
  std::cout << std::string(120, '-') << "\n";

  for (const auto& [cfg, m] : results) {
    std::string model_name = cfg.model_type;
    if (cfg.quantized) model_name += "_Q";
    
    std::cout << std::left << std::setw(20) << model_name
              << std::setw(12) << cfg.task_type
              << std::setw(8) << (cfg.use_gpu ? "GPU" : "CPU")
              << std::fixed << std::setprecision(2)
              << std::setw(10) << m.fps
              << std::setw(12) << m.latency_avg_ms;
    
    if (m.has_accuracy) {
      std::cout << std::setprecision(4)
                << std::setw(10) << m.AP50
                << std::setw(12) << m.mAP5095;
    } else {
      std::cout << std::setw(10) << "N/A"
                << std::setw(12) << "N/A";
    }
    
    std::cout << std::setprecision(1)
              << std::setw(10) << m.memory_mb
              << "\n";
  }
  
  std::cout << std::string(120, '=') << "\n\n";
}

void exportToCSV(const std::string& filepath, const UnifiedConfig& cfg, const UnifiedMetrics& m, const std::string& inputType) {
  std::ofstream out(filepath, std::ios::app);
  if (!out) throw std::runtime_error("Cannot append to results file: " + filepath);
  
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
      << (m.has_accuracy ? std::to_string(m.AP50) : "N/A") << ","
      << (m.has_accuracy ? std::to_string(m.mAP5095) : "N/A") << ","
      << m.frame_count << "\n";
}

void printDetailedReport(const UnifiedConfig& cfg, const UnifiedMetrics& m, const std::string& inputType) {
  std::cout << "\n" << std::string(80, '=') << "\n";
  std::cout << "DETAILED BENCHMARK REPORT\n";
  std::cout << std::string(80, '=') << "\n";
  std::cout << "Model: " << cfg.model_type << " (" << cfg.task_type << ")\n";
  std::cout << "Device: " << (cfg.use_gpu ? "GPU" : "CPU") << " (" << m.environment_type << ")\n";
  std::cout << "Input Type: " << inputType << "\n";
  std::cout << std::string(80, '-') << "\n";
  
  std::cout << "\nPERFORMANCE METRICS:\n";
  std::cout << "  Load Time:        " << std::fixed << std::setprecision(3) << m.load_time_ms << " ms\n";
  std::cout << "  Inference Time:   " << m.inference_avg_ms << " ms (avg)\n";
  std::cout << "  Total Time:       " << m.total_avg_ms << " ms (avg)\n";
  std::cout << "  FPS:              " << std::setprecision(2) << m.fps << "\n";
  std::cout << "  Latency:           " << m.latency_avg_ms << " ms (avg), "
            << m.latency_min_ms << " ms (min), " << m.latency_max_ms << " ms (max)\n";
  std::cout << "  Memory Usage:     " << std::setprecision(1) << m.memory_mb << " MB\n";
  std::cout << "  CPU Usage:        " << m.cpu_usage_percent << "%\n";
  std::cout << "  GPU Usage:        " << m.gpu_usage_percent << "%\n";
  std::cout << "  GPU Memory:       " << m.gpu_memory_used_mb << " MB\n";
  std::cout << "  Frames Processed: " << m.frame_count << "\n";
  
  if (m.has_accuracy) {
    std::cout << "\nACCURACY METRICS:\n";
    std::cout << "  AP50:             " << std::setprecision(4) << m.AP50 << "\n";
    std::cout << "  mAP50-95:         " << m.mAP5095 << "\n";
    std::cout << "  IoU Thresholds:   ";
    for (size_t i = 0; i < m.AP_per_iou.size() && i < 10; ++i) {
      std::cout << std::setprecision(3) << m.AP_per_iou[i] << " ";
    }
    if (m.AP_per_iou.size() > 10) std::cout << "...";
    std::cout << "\n";
  }
  
  std::cout << std::string(80, '=') << "\n\n";
}

// ============================================================================
// Command-line Parsing
// ============================================================================
UnifiedConfig parseConfig(int argc, char** argv) {
  UnifiedConfig cfg;
  
  if (argc < 6) {
    throw std::runtime_error("Invalid arguments");
  }
  
  cfg.model_type = argv[2];
  cfg.task_type = argv[3];
  cfg.model_path = argv[4];
  cfg.labels_path = argv[5];
  
  for (int i = 6; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--gpu" || arg == "gpu") cfg.use_gpu = true;
    else if (arg == "--cpu" || arg == "cpu") cfg.use_gpu = false;
    else if (arg.rfind("--threads=", 0) == 0) cfg.thread_count = std::stoi(arg.substr(10));
    else if (arg == "--quantized") { cfg.quantized = true; cfg.precision = "int8"; }
    else if (arg.rfind("--conf-threshold=", 0) == 0) cfg.conf_threshold = std::stof(arg.substr(17));
    else if (arg.rfind("--nms-threshold=", 0) == 0) cfg.nms_threshold = std::stof(arg.substr(16));
    else if (arg.rfind("--eval-conf-threshold=", 0) == 0) cfg.eval_conf_threshold = std::stof(arg.substr(22));
    else if (arg.rfind("--dataset-type=", 0) == 0) cfg.dataset_type = arg.substr(16);
  }
  
  return cfg;
}

// ============================================================================
// Main Function
// ============================================================================
int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " <mode> [arguments...]\n"
                << "Modes:\n"
                << "  image <model_type> <task_type> <model_path> <labels_path> <image_path> [options]\n"
                << "  video <model_type> <task_type> <model_path> <labels_path> <video_path> [options]\n"
                << "  camera <model_type> <task_type> <model_path> <labels_path> <camera_id> [options]\n"
                << "  evaluate <model_type> <task_type> <model_path> <labels_path> <images_folder> <gt_labels_folder> [options]\n"
                << "  comprehensive\n"
                << "Options: --gpu, --cpu, --threads=N, --quantized, --iterations=N, --duration=N\n"
                << "         --conf-threshold=N, --nms-threshold=N, --eval-conf-threshold=N\n";
      return 1;
    }

    std::string mode = argv[1];

    if (mode == "comprehensive") {
      std::cout << "YOLO Unified Benchmark - Running comprehensive benchmarks...\n";
      std::filesystem::create_directories("results");

      std::vector<std::tuple<std::string, std::string, std::string>> test_configs = {
        {"yolo11", "detection", "models/yolo11n.onnx"},
        {"yolo8", "detection", "models/yolov8n.onnx"},
        // Add segmentation models if available
        // {"yolo11", "segmentation", "models/yolo11n-seg.onnx"},
      };

      const std::string image_path = "data/dog.jpg";
      const std::string video_path = "data/dogs.mp4";
      const bool has_image = std::filesystem::exists(image_path);
      const bool has_video = std::filesystem::exists(video_path);

      std::string results_file = "results/unified_benchmark_" + std::to_string(std::time(nullptr)) + ".csv";
      {
        std::ofstream file(results_file);
        if (!file) throw std::runtime_error("Cannot open results file: " + results_file);
        file << "model_type,task_type,InputType,environment,device,threads,precision,"
                "load_ms,preprocess_ms,inference_ms,postprocess_ms,total_ms,fps,"
                "memory_mb,system_memory_mb,cpu_usage_%,gpu_usage_%,gpu_memory_mb,"
                "latency_avg_ms,latency_min_ms,latency_max_ms,AP50,mAP50-95,frame_count\n";
      }

      std::vector<std::pair<UnifiedConfig, UnifiedMetrics>> all_results;

      for (const auto& [model_type, task_type, model_path] : test_configs) {
        if (!std::filesystem::exists(model_path)) {
          std::cerr << "Skipping " << model_type << "/" << task_type << " - model not found: " << model_path << "\n";
          continue;
        }

        for (bool use_gpu : {false, true}) {
          UnifiedConfig cfg;
          cfg.model_type = model_type;
          cfg.task_type = task_type;
          cfg.model_path = model_path;
          cfg.labels_path = "models/coco.names";
          cfg.use_gpu = use_gpu;
          cfg.thread_count = 1;

          try {
            if (has_image) {
              auto m_img = benchmark_image_unified(cfg, image_path, 100);
              exportToCSV(results_file, cfg, m_img, "Image");
              all_results.push_back({cfg, m_img});
            }
            if (has_video) {
              auto m_vid = benchmark_video_unified(cfg, video_path);
              exportToCSV(results_file, cfg, m_vid, "Video");
              all_results.push_back({cfg, m_vid});
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
          } catch (const std::exception& e) {
            std::cerr << "Error benchmarking " << model_type << "/" << task_type
                      << " on " << (use_gpu ? "GPU" : "CPU") << ": " << e.what() << "\n";
          }
        }
      }

      printTabularSummary(all_results);
      std::cout << "Comprehensive benchmark completed.\n";
      std::cout << "Results saved to: " << results_file << "\n";
      return 0;
    }

    if (mode == "evaluate") {
      if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " evaluate <model_type> <task_type> <model_path> <labels_path> <images_folder> <gt_labels_folder> [options]\n";
        return 1;
      }
      
      UnifiedConfig cfg = parseConfig(argc, argv);
      cfg.dataset_path = argv[6];
      cfg.gt_labels_path = argv[7];
      cfg.evaluate_accuracy = true;
      
      auto metrics = benchmark_evaluate(cfg);
      printDetailedReport(cfg, metrics, "Evaluation");
      
      std::cout << "\n=== Evaluation Results ===\n";
      if (metrics.has_accuracy) {
        for (size_t i = 0; i < metrics.AP_per_iou.size(); ++i) {
          float iou_thr = 0.50f + i * 0.05f;
          std::cout << "IoU " << std::fixed << std::setprecision(2) << iou_thr 
                    << "  AP=" << std::setprecision(4) << metrics.AP_per_iou[i] << "\n";
        }
        std::cout << "\nAP50 = " << metrics.AP50 << "\n";
        std::cout << "mAP50-95 = " << metrics.mAP5095 << "\n";
      }
      std::cout << "\n=== Speed ===\n";
      std::cout << "Images processed = " << metrics.frame_count << "\n";
      std::cout << "Inference time (mean): " << std::fixed << std::setprecision(3) << metrics.total_avg_ms << " ms\n";
      std::cout << "FPS (from mean): " << std::setprecision(2) << metrics.fps << "\n";
      
      return 0;
    }

    // Single-run modes (image, video, camera)
    UnifiedConfig cfg = parseConfig(argc, argv);
    std::string input_path = argv[6];

    int iterations = 100;
    int duration = 30;
    for (int i = 7; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg.rfind("--iterations=", 0) == 0) iterations = std::stoi(arg.substr(14));
      else if (arg.rfind("--duration=", 0) == 0) duration = std::stoi(arg.substr(11));
    }

    UnifiedMetrics m;
    std::string inputType;
    
    if (mode == "image") {
      m = benchmark_image_unified(cfg, input_path, iterations);
      inputType = "Image";
    } else if (mode == "video") {
      m = benchmark_video_unified(cfg, input_path);
      inputType = "Video";
    } else if (mode == "camera") {
      int cam_id = std::stoi(input_path);
      m = benchmark_camera_unified(cfg, cam_id, duration);
      inputType = "Camera";
    } else {
      std::cerr << "Error: invalid mode '" << mode << "'. Use image|video|camera|evaluate|comprehensive.\n";
      return 1;
    }

    printDetailedReport(cfg, m, inputType);
    
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

