// ============================================================================
// YOLO Unified Benchmark Suite
// ============================================================================
// Comprehensive benchmarking for YOLO models across all task types:
// - Detection, Segmentation, Pose Estimation, OBB, Classification
//
// Features:
// - Multi-task support with unified interface
// - Detailed performance metrics (latency, FPS, memory, percentiles)
// - mAP evaluation for detection tasks
// - CSV and JSON export
// - Progress reporting with colored output
//
// Author: YOLOs-CPP Team, https://github.com/Geekgineer/YOLOs-CPP
// ============================================================================

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
#include <optional>

#ifdef _WIN32
    #include <io.h>
    #define isatty _isatty
    #define fileno _fileno
#else
    #include <unistd.h>
#endif

#include <opencv2/opencv.hpp>

// Clean unified YOLO headers
#include "yolos/yolos.hpp"

namespace fs = std::filesystem;
using namespace yolos;

// ============================================================================
// Version Info
// ============================================================================
constexpr const char* BENCHMARK_VERSION = "2.0.0";

// ============================================================================
// ANSI Color Codes for Terminal Output
// ============================================================================
namespace colors {
    const std::string RESET   = "\033[0m";
    const std::string RED     = "\033[31m";
    const std::string GREEN   = "\033[32m";
    const std::string YELLOW  = "\033[33m";
    const std::string BLUE    = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN    = "\033[36m";
    const std::string BOLD    = "\033[1m";
    const std::string DIM     = "\033[2m";
    
    inline bool isTerminal() {
        #ifdef _WIN32
            return false;  // Windows terminal handling is complex
        #else
            return isatty(fileno(stdout));
        #endif
    }
    
    inline std::string colorize(const std::string& text, const std::string& color) {
        if (isTerminal()) return color + text + RESET;
        return text;
    }
}

// ============================================================================
// Progress Bar
// ============================================================================
class ProgressBar {
public:
    ProgressBar(int total, int width = 40, const std::string& prefix = "")
        : total_(total), width_(width), prefix_(prefix), current_(0) {}
    
    void update(int current) {
        current_ = current;
        print();
    }
    
    void increment() {
        ++current_;
        print();
    }
    
    void finish() {
        current_ = total_;
        print();
        std::cout << "\n";
    }
    
private:
    void print() const {
        if (!colors::isTerminal()) return;
        
        float progress = static_cast<float>(current_) / total_;
        int filled = static_cast<int>(progress * width_);
        
        std::cout << "\r" << prefix_ << " [";
        for (int i = 0; i < width_; ++i) {
            if (i < filled) std::cout << "█";
            else if (i == filled) std::cout << "▓";
            else std::cout << "░";
        }
        std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100) << "% "
                  << "(" << current_ << "/" << total_ << ")" << std::flush;
    }
    
    int total_;
    int width_;
    std::string prefix_;
    int current_;
};

// ============================================================================
// Configuration Structures
// ============================================================================
struct BenchmarkConfig {
    // Model configuration
    std::string model_type;
    std::string task_type;    // detection, segmentation, pose, obb, classification
    std::string model_path;
    std::string labels_path;
    bool use_gpu = false;
    int thread_count = 0;     // 0 = auto
    bool quantized = false;
    std::string precision = "fp32";
    std::string device = "CPU";
    
    // Benchmark parameters
    int warmup_iterations = 10;
    int iterations = 100;
    int max_video_frames = 1000;
    int camera_duration_seconds = 30;
    
    // Inference thresholds
    float conf_threshold = 0.4f;
    float nms_threshold = 0.45f;
    float eval_conf_threshold = 0.001f;  // Lower for evaluation
    
    // Evaluation-specific
    bool evaluate_accuracy = false;
    std::string dataset_path;
    std::string gt_labels_path;
    std::string dataset_type = "custom";  // "coco" or "custom"
    
    // Output options
    bool verbose = false;
    bool json_output = false;
    std::string output_dir = "results";
};

struct LatencyStats {
    double avg = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    
    static LatencyStats compute(std::vector<double>& samples) {
        LatencyStats stats;
        if (samples.empty()) return stats;
        
        std::sort(samples.begin(), samples.end());
        size_t n = samples.size();
        
        // Basic stats
        double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
        stats.avg = sum / n;
        stats.min = samples.front();
        stats.max = samples.back();
        
        // Standard deviation
        double sq_sum = 0.0;
        for (double v : samples) sq_sum += (v - stats.avg) * (v - stats.avg);
        stats.stddev = std::sqrt(sq_sum / n);
        
        // Percentiles
        auto percentile = [&](double p) -> double {
            double idx = p * (n - 1);
            size_t lower = static_cast<size_t>(idx);
            size_t upper = std::min(lower + 1, n - 1);
            double frac = idx - lower;
            return samples[lower] * (1 - frac) + samples[upper] * frac;
        };
        
        stats.p50 = percentile(0.50);
        stats.p90 = percentile(0.90);
        stats.p95 = percentile(0.95);
        stats.p99 = percentile(0.99);
        
        return stats;
    }
};

struct BenchmarkMetrics {
    // Timing metrics
    double load_time_ms = 0.0;
    double warmup_time_ms = 0.0;
    LatencyStats latency;
    double fps = 0.0;
    double throughput = 0.0;  // images/second
    int frame_count = 0;
    
    // Memory metrics
    double peak_memory_mb = 0.0;
    double memory_delta_mb = 0.0;
    
    // System metrics
    double cpu_usage_avg = 0.0;
    double gpu_usage_avg = 0.0;
    double gpu_memory_mb = 0.0;
    
    // Accuracy metrics (when GT available)
    float AP50 = 0.0f;
    float mAP5095 = 0.0f;
    std::vector<float> AP_per_iou;
    bool has_accuracy = false;
    
    // Metadata
    std::string environment_type = "CPU";
    std::string model_input_shape;
    int detected_objects_count = 0;
};

// ============================================================================
// System Monitoring Utilities
// ============================================================================
namespace sysmon {

#ifdef _WIN32
    #define NOMINMAX
    #include <windows.h>
    #include <psapi.h>
    #pragma comment(lib, "psapi.lib")
#else
    #include <sys/resource.h>
    #include <unistd.h>
#endif

double getProcessMemoryMB() {
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

double getCPUUsage() {
#ifdef _WIN32
    return 0.0;
#else
    static unsigned long long lastUser = 0, lastUserLow = 0, lastSys = 0, lastIdle = 0;
    std::ifstream file("/proc/stat");
    std::string line;
    if (!std::getline(file, line)) return 0.0;
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

std::pair<double, double> getGPUStats() {
#ifdef _WIN32
    return {0.0, 0.0};
#else
    FILE* pipe = popen("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null", "r");
    if (!pipe) return {0.0, 0.0};
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof buffer, pipe) != nullptr) result += buffer;
    pclose(pipe);
    if (result.empty()) return {0.0, 0.0};
    double gpu_util = 0.0, gpu_mem = 0.0;
    if (std::sscanf(result.c_str(), "%lf, %lf", &gpu_util, &gpu_mem) != 2) return {0.0, 0.0};
    return {gpu_util, gpu_mem};
#endif
}

} // namespace sysmon

// ============================================================================
// Unified Detector Factory with All Task Types
// ============================================================================
class UnifiedDetector {
public:
    enum class TaskType { Detection, Segmentation, Pose, OBB, Classification };
    
    virtual ~UnifiedDetector() = default;
    virtual std::string getDevice() const = 0;
    virtual cv::Size getInputShape() const = 0;
    virtual TaskType getTaskType() const = 0;
    virtual int runInference(const cv::Mat& image, float conf, float nms) = 0;
    virtual void drawResults(cv::Mat& image) const = 0;
};

class DetectionWrapper : public UnifiedDetector {
public:
    DetectionWrapper(const std::string& model_path, const std::string& labels_path, bool use_gpu)
        : detector_(std::make_unique<det::YOLODetector>(model_path, labels_path, use_gpu)) {}
    
    std::string getDevice() const override { return detector_->getDevice(); }
    cv::Size getInputShape() const override { return detector_->getInputShape(); }
    TaskType getTaskType() const override { return TaskType::Detection; }
    
    int runInference(const cv::Mat& image, float conf, float nms) override {
        results_ = detector_->detect(image, conf, nms);
        return static_cast<int>(results_.size());
    }
    
    void drawResults(cv::Mat& image) const override {
        detector_->drawDetections(image, results_);
    }
    
private:
    std::unique_ptr<det::YOLODetector> detector_;
    std::vector<det::Detection> results_;
};

class SegmentationWrapper : public UnifiedDetector {
public:
    SegmentationWrapper(const std::string& model_path, const std::string& labels_path, bool use_gpu)
        : detector_(std::make_unique<seg::YOLOSegDetector>(model_path, labels_path, use_gpu)) {}
    
    std::string getDevice() const override { return detector_->getDevice(); }
    cv::Size getInputShape() const override { return detector_->getInputShape(); }
    TaskType getTaskType() const override { return TaskType::Segmentation; }
    
    int runInference(const cv::Mat& image, float conf, float nms) override {
        results_ = detector_->segment(image, conf, nms);
        return static_cast<int>(results_.size());
    }
    
    void drawResults(cv::Mat& image) const override {
        detector_->drawSegmentations(image, results_);
    }
    
private:
    std::unique_ptr<seg::YOLOSegDetector> detector_;
    std::vector<seg::Segmentation> results_;
};

class PoseWrapper : public UnifiedDetector {
public:
    PoseWrapper(const std::string& model_path, const std::string& labels_path, bool use_gpu)
        : detector_(std::make_unique<pose::YOLOPoseDetector>(model_path, labels_path, use_gpu)) {}
    
    std::string getDevice() const override { return detector_->getDevice(); }
    cv::Size getInputShape() const override { return detector_->getInputShape(); }
    TaskType getTaskType() const override { return TaskType::Pose; }
    
    int runInference(const cv::Mat& image, float conf, float nms) override {
        results_ = detector_->detect(image, conf, nms);
        return static_cast<int>(results_.size());
    }
    
    void drawResults(cv::Mat& image) const override {
        detector_->drawPoses(image, results_);
    }
    
private:
    std::unique_ptr<pose::YOLOPoseDetector> detector_;
    std::vector<pose::PoseResult> results_;
};

class OBBWrapper : public UnifiedDetector {
public:
    OBBWrapper(const std::string& model_path, const std::string& labels_path, bool use_gpu)
        : detector_(std::make_unique<obb::YOLOOBBDetector>(model_path, labels_path, use_gpu)) {}
    
    std::string getDevice() const override { return detector_->getDevice(); }
    cv::Size getInputShape() const override { return detector_->getInputShape(); }
    TaskType getTaskType() const override { return TaskType::OBB; }
    
    int runInference(const cv::Mat& image, float conf, float nms) override {
        results_ = detector_->detect(image, conf, nms);
        return static_cast<int>(results_.size());
    }
    
    void drawResults(cv::Mat& image) const override {
        detector_->drawDetections(image, results_);
    }
    
private:
    std::unique_ptr<obb::YOLOOBBDetector> detector_;
    std::vector<obb::OBBResult> results_;
};

class ClassificationWrapper : public UnifiedDetector {
public:
    ClassificationWrapper(const std::string& model_path, const std::string& labels_path, bool use_gpu)
        : classifier_(std::make_unique<cls::YOLOClassifier>(model_path, labels_path, use_gpu)) {}
    
    std::string getDevice() const override { return "cpu"; }  // Classification doesn't track this
    cv::Size getInputShape() const override { return classifier_->getInputShape(); }
    TaskType getTaskType() const override { return TaskType::Classification; }
    
    int runInference(const cv::Mat& image, float /*conf*/, float /*nms*/) override {
        result_ = classifier_->classify(image);
        return (result_.classId >= 0) ? 1 : 0;
    }
    
    void drawResults(cv::Mat& image) const override {
        classifier_->drawResult(image, result_);
    }
    
private:
    std::unique_ptr<cls::YOLOClassifier> classifier_;
    cls::ClassificationResult result_;
};

std::unique_ptr<UnifiedDetector> createDetector(const BenchmarkConfig& config) {
    if (config.task_type == "detection" || config.task_type == "det") {
        return std::make_unique<DetectionWrapper>(config.model_path, config.labels_path, config.use_gpu);
    } else if (config.task_type == "segmentation" || config.task_type == "seg") {
        return std::make_unique<SegmentationWrapper>(config.model_path, config.labels_path, config.use_gpu);
    } else if (config.task_type == "pose") {
        return std::make_unique<PoseWrapper>(config.model_path, config.labels_path, config.use_gpu);
    } else if (config.task_type == "obb") {
        return std::make_unique<OBBWrapper>(config.model_path, config.labels_path, config.use_gpu);
    } else if (config.task_type == "classification" || config.task_type == "cls") {
        return std::make_unique<ClassificationWrapper>(config.model_path, config.labels_path, config.use_gpu);
    }
    throw std::runtime_error("Unsupported task type: " + config.task_type);
}

// ============================================================================
// File Utilities
// ============================================================================
std::vector<std::string> listImages(const std::string& folder) {
    std::vector<std::string> out;
    if (!fs::exists(folder) || !fs::is_directory(folder)) return out;
    
    for (const auto& e : fs::directory_iterator(folder)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" || ext == ".tiff") {
            out.push_back(e.path().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// ============================================================================
// Benchmark Functions
// ============================================================================
BenchmarkMetrics benchmarkImage(BenchmarkConfig& config, const std::string& image_path) {
    BenchmarkMetrics metrics;
    
    // Load model and image
    auto load_start = std::chrono::high_resolution_clock::now();
    auto detector = createDetector(config);
    config.device = detector->getDevice();
    
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) throw std::runtime_error("Could not read image: " + image_path);
    
    auto load_end = std::chrono::high_resolution_clock::now();
    metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    metrics.environment_type = (config.device == "gpu") ? "CUDA" : "CPU";
    
    auto input_shape = detector->getInputShape();
    metrics.model_input_shape = std::to_string(input_shape.width) + "x" + std::to_string(input_shape.height);
    
    double initial_memory = sysmon::getProcessMemoryMB();
    
    // Warmup
    auto warmup_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < config.warmup_iterations; ++i) {
        detector->runInference(image, config.conf_threshold, config.nms_threshold);
    }
    auto warmup_end = std::chrono::high_resolution_clock::now();
    metrics.warmup_time_ms = std::chrono::duration<double, std::milli>(warmup_end - warmup_start).count();
    
    // Benchmark iterations
    std::vector<double> latencies;
    latencies.reserve(config.iterations);
    std::vector<double> cpu_samples, gpu_samples;
    int total_detections = 0;
    
    ProgressBar progress(config.iterations, 40, "Benchmarking");
    
    sysmon::getCPUUsage();  // Initialize
    
    for (int i = 0; i < config.iterations; ++i) {
        cpu_samples.push_back(sysmon::getCPUUsage());
        auto [gpu_util, gpu_mem] = sysmon::getGPUStats();
        gpu_samples.push_back(gpu_util);
        if (i == 0) metrics.gpu_memory_mb = gpu_mem;
        
        auto start = std::chrono::high_resolution_clock::now();
        int count = detector->runInference(image, config.conf_threshold, config.nms_threshold);
        auto end = std::chrono::high_resolution_clock::now();
        
        latencies.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        total_detections += count;
        
        if (config.verbose) progress.increment();
    }
    
    if (config.verbose) progress.finish();
    
    // Compute metrics
    metrics.latency = LatencyStats::compute(latencies);
    metrics.fps = (metrics.latency.avg > 0.0) ? (1000.0 / metrics.latency.avg) : 0.0;
    metrics.throughput = metrics.fps;
    metrics.frame_count = config.iterations;
    metrics.detected_objects_count = total_detections / config.iterations;
    
    double final_memory = sysmon::getProcessMemoryMB();
    metrics.peak_memory_mb = final_memory;
    metrics.memory_delta_mb = final_memory - initial_memory;
    
    if (!cpu_samples.empty()) {
        metrics.cpu_usage_avg = std::accumulate(cpu_samples.begin(), cpu_samples.end(), 0.0) / cpu_samples.size();
    }
    if (!gpu_samples.empty()) {
        metrics.gpu_usage_avg = std::accumulate(gpu_samples.begin(), gpu_samples.end(), 0.0) / gpu_samples.size();
    }
    
    return metrics;
}

BenchmarkMetrics benchmarkVideo(BenchmarkConfig& config, const std::string& video_path) {
    BenchmarkMetrics metrics;
    
    auto load_start = std::chrono::high_resolution_clock::now();
    auto detector = createDetector(config);
    config.device = detector->getDevice();
    
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) throw std::runtime_error("Could not open video: " + video_path);
    
    int total_frames = std::min(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT)), config.max_video_frames);
    
    auto load_end = std::chrono::high_resolution_clock::now();
    metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    metrics.environment_type = (config.device == "gpu") ? "CUDA" : "CPU";
    
    auto input_shape = detector->getInputShape();
    metrics.model_input_shape = std::to_string(input_shape.width) + "x" + std::to_string(input_shape.height);
    
    double initial_memory = sysmon::getProcessMemoryMB();
    
    std::vector<double> latencies;
    int frame_count = 0;
    int total_detections = 0;
    cv::Mat frame;
    
    ProgressBar progress(total_frames, 40, "Processing video");
    
    auto bench_start = std::chrono::high_resolution_clock::now();
    
    while (cap.read(frame) && frame_count < config.max_video_frames) {
        if (frame.empty()) continue;
        
        auto start = std::chrono::high_resolution_clock::now();
        int count = detector->runInference(frame, config.conf_threshold, config.nms_threshold);
        auto end = std::chrono::high_resolution_clock::now();
        
        latencies.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        total_detections += count;
        frame_count++;
        
        if (config.verbose) progress.update(frame_count);
    }
    
    auto bench_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(bench_end - bench_start).count();
    
    if (config.verbose) progress.finish();
    
    metrics.latency = LatencyStats::compute(latencies);
    metrics.fps = (total_time_ms > 0.0) ? ((frame_count * 1000.0) / total_time_ms) : 0.0;
    metrics.throughput = metrics.fps;
    metrics.frame_count = frame_count;
    metrics.detected_objects_count = (frame_count > 0) ? (total_detections / frame_count) : 0;
    
    double final_memory = sysmon::getProcessMemoryMB();
    metrics.peak_memory_mb = final_memory;
    metrics.memory_delta_mb = final_memory - initial_memory;
    
    return metrics;
}

BenchmarkMetrics benchmarkCamera(BenchmarkConfig& config, int camera_id) {
    BenchmarkMetrics metrics;
    
    auto load_start = std::chrono::high_resolution_clock::now();
    auto detector = createDetector(config);
    config.device = detector->getDevice();
    
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) throw std::runtime_error("Could not open camera: " + std::to_string(camera_id));
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    
    auto load_end = std::chrono::high_resolution_clock::now();
    metrics.load_time_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    metrics.environment_type = (config.device == "gpu") ? "CUDA" : "CPU";
    
    auto input_shape = detector->getInputShape();
    metrics.model_input_shape = std::to_string(input_shape.width) + "x" + std::to_string(input_shape.height);
    
    std::vector<double> latencies;
    int frame_count = 0;
    cv::Mat frame;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time + std::chrono::seconds(config.camera_duration_seconds);
    
    std::cout << "Benchmarking camera for " << config.camera_duration_seconds << " seconds...\n";
    
    while (std::chrono::high_resolution_clock::now() < end_time && cap.read(frame)) {
        if (frame.empty()) continue;
        
        auto start = std::chrono::high_resolution_clock::now();
        detector->runInference(frame, config.conf_threshold, config.nms_threshold);
        auto end = std::chrono::high_resolution_clock::now();
        
        latencies.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        frame_count++;
    }
    
    metrics.latency = LatencyStats::compute(latencies);
    metrics.fps = (metrics.latency.avg > 0.0) ? (1000.0 / metrics.latency.avg) : 0.0;
    metrics.throughput = metrics.fps;
    metrics.frame_count = frame_count;
    
    return metrics;
}

// ============================================================================
// Output Functions
// ============================================================================
std::string repeatStr(const std::string& s, int n) {
    std::string result;
    result.reserve(s.size() * n);
    for (int i = 0; i < n; ++i) result += s;
    return result;
}

void printMetrics(const BenchmarkConfig& config, const BenchmarkMetrics& m, const std::string& input_type) {
    using namespace colors;
    
    std::cout << "\n" << colorize(std::string(80, '='), CYAN) << "\n";
    std::cout << colorize("  BENCHMARK RESULTS", BOLD) << "\n";
    std::cout << colorize(std::string(80, '='), CYAN) << "\n\n";
    
    // Model info
    std::cout << colorize("MODEL CONFIGURATION", YELLOW) << "\n";
    std::cout << colorize(std::string(40, '-'), DIM) << "\n";
    std::cout << "  Model:       " << colorize(config.model_type, GREEN) << "\n";
    std::cout << "  Task:        " << config.task_type << "\n";
    std::cout << "  Device:      " << colorize(m.environment_type, (m.environment_type == "CUDA" ? GREEN : YELLOW)) << "\n";
    std::cout << "  Input Shape: " << m.model_input_shape << "\n";
    std::cout << "  Input Type:  " << input_type << "\n";
    std::cout << "\n";
    
    // Performance metrics
    std::cout << colorize("PERFORMANCE METRICS", YELLOW) << "\n";
    std::cout << colorize(std::string(40, '-'), DIM) << "\n";
    std::cout << std::fixed;
    std::cout << "  Load Time:      " << std::setprecision(2) << m.load_time_ms << " ms\n";
    std::cout << "  Warmup Time:    " << std::setprecision(2) << m.warmup_time_ms << " ms\n";
    std::cout << "  Frames:         " << m.frame_count << "\n";
    std::cout << "  FPS:            " << colorize(std::to_string(static_cast<int>(m.fps)), GREEN) << "\n";
    std::cout << "\n";
    
    // Latency breakdown
    std::cout << colorize("LATENCY STATISTICS (ms)", YELLOW) << "\n";
    std::cout << colorize(std::string(40, '-'), DIM) << "\n";
    std::cout << "  Average:   " << std::setprecision(3) << m.latency.avg << "\n";
    std::cout << "  Std Dev:   " << std::setprecision(3) << m.latency.stddev << "\n";
    std::cout << "  Min:       " << std::setprecision(3) << m.latency.min << "\n";
    std::cout << "  Max:       " << std::setprecision(3) << m.latency.max << "\n";
    std::cout << "  P50:       " << std::setprecision(3) << m.latency.p50 << "\n";
    std::cout << "  P90:       " << std::setprecision(3) << m.latency.p90 << "\n";
    std::cout << "  P95:       " << std::setprecision(3) << m.latency.p95 << "\n";
    std::cout << "  P99:       " << std::setprecision(3) << m.latency.p99 << "\n";
    std::cout << "\n";
    
    // Memory
    std::cout << colorize("RESOURCE USAGE", YELLOW) << "\n";
    std::cout << colorize(std::string(40, '-'), DIM) << "\n";
    std::cout << "  Peak Memory:    " << std::setprecision(1) << m.peak_memory_mb << " MB\n";
    std::cout << "  Memory Delta:   " << std::setprecision(1) << m.memory_delta_mb << " MB\n";
    std::cout << "  CPU Usage:      " << std::setprecision(1) << m.cpu_usage_avg << "%\n";
    std::cout << "  GPU Usage:      " << std::setprecision(1) << m.gpu_usage_avg << "%\n";
    std::cout << "  GPU Memory:     " << std::setprecision(1) << m.gpu_memory_mb << " MB\n";
    std::cout << "  Avg Detections: " << m.detected_objects_count << "\n";
    
    if (m.has_accuracy) {
        std::cout << "\n";
        std::cout << colorize("ACCURACY METRICS", YELLOW) << "\n";
        std::cout << colorize(std::string(40, '-'), DIM) << "\n";
        std::cout << "  AP50:      " << std::setprecision(4) << m.AP50 << "\n";
        std::cout << "  mAP50-95:  " << std::setprecision(4) << m.mAP5095 << "\n";
    }
    
    std::cout << "\n" << colorize(std::string(80, '='), CYAN) << "\n\n";
}

void exportCSV(const std::string& filepath, const BenchmarkConfig& cfg, const BenchmarkMetrics& m, const std::string& input_type) {
    bool exists = fs::exists(filepath);
    std::ofstream out(filepath, std::ios::app);
    if (!out) throw std::runtime_error("Cannot open CSV file: " + filepath);
    
    // Write header if new file
    if (!exists) {
        out << "timestamp,model_type,task_type,input_type,device,precision,input_shape,"
               "load_ms,warmup_ms,frames,"
               "latency_avg,latency_stddev,latency_min,latency_max,latency_p50,latency_p90,latency_p95,latency_p99,"
               "fps,throughput,memory_peak_mb,memory_delta_mb,cpu_usage,gpu_usage,gpu_memory_mb,"
               "avg_detections,AP50,mAP5095\n";
    }
    
    auto now = std::time(nullptr);
    out << now << ","
        << cfg.model_type << ","
        << cfg.task_type << ","
        << input_type << ","
        << m.environment_type << ","
        << cfg.precision << ","
        << m.model_input_shape << ","
        << std::fixed << std::setprecision(3)
        << m.load_time_ms << ","
        << m.warmup_time_ms << ","
        << m.frame_count << ","
        << m.latency.avg << ","
        << m.latency.stddev << ","
        << m.latency.min << ","
        << m.latency.max << ","
        << m.latency.p50 << ","
        << m.latency.p90 << ","
        << m.latency.p95 << ","
        << m.latency.p99 << ","
        << std::setprecision(2) << m.fps << ","
        << m.throughput << ","
        << std::setprecision(1) << m.peak_memory_mb << ","
        << m.memory_delta_mb << ","
        << m.cpu_usage_avg << ","
        << m.gpu_usage_avg << ","
        << m.gpu_memory_mb << ","
        << m.detected_objects_count << ","
        << (m.has_accuracy ? std::to_string(m.AP50) : "N/A") << ","
        << (m.has_accuracy ? std::to_string(m.mAP5095) : "N/A") << "\n";
}

void exportJSON(const std::string& filepath, const BenchmarkConfig& cfg, const BenchmarkMetrics& m, const std::string& input_type) {
    std::ofstream out(filepath);
    if (!out) throw std::runtime_error("Cannot open JSON file: " + filepath);
    
    out << "{\n";
    out << "  \"timestamp\": " << std::time(nullptr) << ",\n";
    out << "  \"benchmark_version\": \"" << BENCHMARK_VERSION << "\",\n";
    out << "  \"config\": {\n";
    out << "    \"model_type\": \"" << cfg.model_type << "\",\n";
    out << "    \"task_type\": \"" << cfg.task_type << "\",\n";
    out << "    \"model_path\": \"" << cfg.model_path << "\",\n";
    out << "    \"device\": \"" << m.environment_type << "\",\n";
    out << "    \"precision\": \"" << cfg.precision << "\",\n";
    out << "    \"input_shape\": \"" << m.model_input_shape << "\",\n";
    out << "    \"warmup_iterations\": " << cfg.warmup_iterations << ",\n";
    out << "    \"iterations\": " << cfg.iterations << "\n";
    out << "  },\n";
    out << "  \"input_type\": \"" << input_type << "\",\n";
    out << "  \"performance\": {\n";
    out << std::fixed << std::setprecision(3);
    out << "    \"load_time_ms\": " << m.load_time_ms << ",\n";
    out << "    \"warmup_time_ms\": " << m.warmup_time_ms << ",\n";
    out << "    \"frames_processed\": " << m.frame_count << ",\n";
    out << std::setprecision(2);
    out << "    \"fps\": " << m.fps << ",\n";
    out << "    \"throughput\": " << m.throughput << "\n";
    out << "  },\n";
    out << "  \"latency_ms\": {\n";
    out << std::setprecision(3);
    out << "    \"avg\": " << m.latency.avg << ",\n";
    out << "    \"stddev\": " << m.latency.stddev << ",\n";
    out << "    \"min\": " << m.latency.min << ",\n";
    out << "    \"max\": " << m.latency.max << ",\n";
    out << "    \"p50\": " << m.latency.p50 << ",\n";
    out << "    \"p90\": " << m.latency.p90 << ",\n";
    out << "    \"p95\": " << m.latency.p95 << ",\n";
    out << "    \"p99\": " << m.latency.p99 << "\n";
    out << "  },\n";
    out << "  \"resources\": {\n";
    out << std::setprecision(1);
    out << "    \"peak_memory_mb\": " << m.peak_memory_mb << ",\n";
    out << "    \"memory_delta_mb\": " << m.memory_delta_mb << ",\n";
    out << "    \"cpu_usage_percent\": " << m.cpu_usage_avg << ",\n";
    out << "    \"gpu_usage_percent\": " << m.gpu_usage_avg << ",\n";
    out << "    \"gpu_memory_mb\": " << m.gpu_memory_mb << "\n";
    out << "  },\n";
    out << "  \"detection\": {\n";
    out << "    \"avg_objects_per_frame\": " << m.detected_objects_count << "\n";
    out << "  }";
    if (m.has_accuracy) {
        out << ",\n  \"accuracy\": {\n";
        out << std::setprecision(4);
        out << "    \"AP50\": " << m.AP50 << ",\n";
        out << "    \"mAP50_95\": " << m.mAP5095 << "\n";
        out << "  }";
    }
    out << "\n}\n";
}

// ============================================================================
// Command-line Parsing
// ============================================================================
void printUsage(const char* prog) {
    std::cout << colors::colorize("YOLO Unified Benchmark v" + std::string(BENCHMARK_VERSION), colors::BOLD) << "\n\n";
    std::cout << "Usage: " << prog << " <mode> [arguments...]\n\n";
    std::cout << colors::colorize("Modes:", colors::YELLOW) << "\n";
    std::cout << "  image     Benchmark on a single image\n";
    std::cout << "  video     Benchmark on a video file\n";
    std::cout << "  camera    Benchmark using camera input\n";
    std::cout << "  quick     Quick benchmark with default settings\n\n";
    std::cout << colors::colorize("Arguments:", colors::YELLOW) << "\n";
    std::cout << "  image <model_type> <task_type> <model_path> <labels_path> <image_path> [options]\n";
    std::cout << "  video <model_type> <task_type> <model_path> <labels_path> <video_path> [options]\n";
    std::cout << "  camera <model_type> <task_type> <model_path> <labels_path> <camera_id> [options]\n";
    std::cout << "  quick <model_path> <image_path>\n\n";
    std::cout << colors::colorize("Task Types:", colors::YELLOW) << "\n";
    std::cout << "  detection (det), segmentation (seg), pose, obb, classification (cls)\n\n";
    std::cout << colors::colorize("Options:", colors::YELLOW) << "\n";
    std::cout << "  --gpu                   Use GPU for inference\n";
    std::cout << "  --cpu                   Use CPU for inference (default)\n";
    std::cout << "  --iterations=N          Number of benchmark iterations (default: 100)\n";
    std::cout << "  --warmup=N              Number of warmup iterations (default: 10)\n";
    std::cout << "  --duration=N            Camera benchmark duration in seconds (default: 30)\n";
    std::cout << "  --conf-threshold=N      Confidence threshold (default: 0.4)\n";
    std::cout << "  --nms-threshold=N       NMS threshold (default: 0.45)\n";
    std::cout << "  --output-dir=PATH       Output directory for results (default: results)\n";
    std::cout << "  --json                  Export results as JSON\n";
    std::cout << "  --verbose               Show progress during benchmark\n";
    std::cout << "  --help                  Show this help message\n\n";
    std::cout << colors::colorize("Examples:", colors::YELLOW) << "\n";
    std::cout << "  " << prog << " quick models/yolo11n.onnx data/dog.jpg\n";
    std::cout << "  " << prog << " image yolo11 det models/yolo11n.onnx models/coco.names data/dog.jpg --gpu\n";
    std::cout << "  " << prog << " video yolo11 det models/yolo11n.onnx models/coco.names data/video.mp4 --iterations=500\n";
}

BenchmarkConfig parseArgs(int argc, char** argv) {
    BenchmarkConfig cfg;
    cfg.verbose = true;  // Default to verbose for better UX
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--gpu" || arg == "gpu") cfg.use_gpu = true;
        else if (arg == "--cpu" || arg == "cpu") cfg.use_gpu = false;
        else if (arg == "--verbose" || arg == "-v") cfg.verbose = true;
        else if (arg == "--quiet" || arg == "-q") cfg.verbose = false;
        else if (arg == "--json") cfg.json_output = true;
        else if (arg.rfind("--iterations=", 0) == 0) cfg.iterations = std::stoi(arg.substr(13));
        else if (arg.rfind("--warmup=", 0) == 0) cfg.warmup_iterations = std::stoi(arg.substr(9));
        else if (arg.rfind("--duration=", 0) == 0) cfg.camera_duration_seconds = std::stoi(arg.substr(11));
        else if (arg.rfind("--conf-threshold=", 0) == 0) cfg.conf_threshold = std::stof(arg.substr(17));
        else if (arg.rfind("--nms-threshold=", 0) == 0) cfg.nms_threshold = std::stof(arg.substr(16));
        else if (arg.rfind("--output-dir=", 0) == 0) cfg.output_dir = arg.substr(13);
    }
    
    return cfg;
}

// ============================================================================
// Main Function
// ============================================================================
int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            printUsage(argv[0]);
            return 1;
        }
        
        std::string mode = argv[1];
        
        if (mode == "--help" || mode == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        
        // Quick mode - minimal arguments
        if (mode == "quick") {
            if (argc < 4) {
                std::cerr << "Usage: " << argv[0] << " quick <model_path> <image_path> [--gpu]\n";
                return 1;
            }
            
            BenchmarkConfig cfg = parseArgs(argc, argv);
            cfg.model_path = argv[2];
            
            // Try multiple possible label paths
            std::vector<std::string> label_candidates = {
                "models/coco.names",
                "../models/coco.names",
                fs::path(cfg.model_path).parent_path().string() + "/coco.names"
            };
            cfg.labels_path = "models/coco.names";  // Default
            for (const auto& path : label_candidates) {
                if (fs::exists(path)) {
                    cfg.labels_path = path;
                    break;
                }
            }
            
            cfg.model_type = fs::path(cfg.model_path).stem().string();
            cfg.task_type = "detection";
            
            std::string image_path = argv[3];
            
            std::cout << colors::colorize("\n🚀 Quick Benchmark Mode\n", colors::BOLD);
            std::cout << "Model: " << cfg.model_path << "\n";
            std::cout << "Image: " << image_path << "\n";
            std::cout << "Device: " << (cfg.use_gpu ? "GPU" : "CPU") << "\n\n";
            
            auto metrics = benchmarkImage(cfg, image_path);
            printMetrics(cfg, metrics, "Image");
            
            return 0;
        }
        
        // Standard modes
        if (mode == "image" || mode == "video" || mode == "camera") {
            if (argc < 7) {
                std::cerr << "Usage: " << argv[0] << " " << mode << " <model_type> <task_type> <model_path> <labels_path> <input_path> [options]\n";
                return 1;
            }
            
            BenchmarkConfig cfg = parseArgs(argc, argv);
            cfg.model_type = argv[2];
            cfg.task_type = argv[3];
            cfg.model_path = argv[4];
            cfg.labels_path = argv[5];
            std::string input_path = argv[6];
            
            fs::create_directories(cfg.output_dir);
            
            BenchmarkMetrics metrics;
            std::string input_type;
            
            if (mode == "image") {
                metrics = benchmarkImage(cfg, input_path);
                input_type = "Image";
            } else if (mode == "video") {
                metrics = benchmarkVideo(cfg, input_path);
                input_type = "Video";
            } else if (mode == "camera") {
                int cam_id = std::stoi(input_path);
                metrics = benchmarkCamera(cfg, cam_id);
                input_type = "Camera";
            }
            
            printMetrics(cfg, metrics, input_type);
            
            // Export results
            std::string csv_path = cfg.output_dir + "/benchmark_results.csv";
            exportCSV(csv_path, cfg, metrics, input_type);
            std::cout << "Results appended to: " << csv_path << "\n";
            
            if (cfg.json_output) {
                std::string json_path = cfg.output_dir + "/" + cfg.model_type + "_" + std::to_string(std::time(nullptr)) + ".json";
                exportJSON(json_path, cfg, metrics, input_type);
                std::cout << "JSON exported to: " << json_path << "\n";
            }
            
            return 0;
        }
        
        std::cerr << "Unknown mode: " << mode << "\n";
        printUsage(argv[0]);
        return 1;
        
    } catch (const std::exception& e) {
        std::cerr << colors::colorize("Error: ", colors::RED) << e.what() << "\n";
        return 1;
    }
}
