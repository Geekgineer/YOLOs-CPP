/*
 * YOLO Benchmark Suite
 * Professional multi-backend benchmarking tool for YOLO models
 * Supports ONNX Runtime and OpenCV DNN backend comparison
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>

// Use existing project headers
#include "det/YOLO11.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Simple benchmark result structure
struct SimpleBenchmarkResult {
    std::string backend_name;
    bool success = false;
    double mean_time_ms = 0.0;
    double std_time_ms = 0.0;
    double fps = 0.0;
    int detection_count = 0;
    std::string error_message;
};

// Simple statistics calculation
class SimpleStats {
public:
    static SimpleBenchmarkResult calculate_stats(const std::string& name, 
                                                const std::vector<double>& times,
                                                int detection_count = 0) {
        SimpleBenchmarkResult result;
        result.backend_name = name;
        result.detection_count = detection_count;
        
        if (times.empty()) {
            result.success = false;
            result.error_message = "No timing data";
            return result;
        }
        
        result.success = true;
        result.mean_time_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        
        // Calculate standard deviation
        if (times.size() > 1) {
            double sum_sq_diff = 0.0;
            for (double time : times) {
                double diff = time - result.mean_time_ms;
                sum_sq_diff += diff * diff;
            }
            result.std_time_ms = std::sqrt(sum_sq_diff / (times.size() - 1));
        }
        
        result.fps = (result.mean_time_ms > 0) ? (1000.0 / result.mean_time_ms) : 0.0;
        
        return result;
    }
};

// Simple benchmark runner
class SimpleBenchmarkRunner {
private:
    std::string model_path_;
    std::string labels_path_;
    cv::Mat test_image_;
    int num_runs_;
    int warmup_runs_;
    
public:
    SimpleBenchmarkRunner(const std::string& model_path, 
                         const std::string& labels_path,
                         const std::string& input_path = "",
                         int image_size = 640, 
                         int num_runs = 20, 
                         int warmup_runs = 5)
        : model_path_(model_path), labels_path_(labels_path), 
          num_runs_(num_runs), warmup_runs_(warmup_runs) {
        
        load_test_image(input_path, image_size);
    }
    
    void load_test_image(const std::string& input_path, int image_size) {
        if (input_path.empty()) {
            // Generate random input
            test_image_ = cv::Mat(image_size, image_size, CV_8UC3);
            cv::randu(test_image_, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
            std::cout << "📷 Using random input image (" << image_size << "x" << image_size << ")" << std::endl;
            return;
        }
        
        // Check if input is a video file
        std::string extension = input_path.substr(input_path.find_last_of(".") + 1);
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        
        if (extension == "mp4" || extension == "avi" || extension == "mov" || extension == "mkv") {
            // Load from video file
            cv::VideoCapture cap(input_path);
            if (!cap.isOpened()) {
                std::cout << "⚠️  Could not open video " << input_path << ", using random input" << std::endl;
                test_image_ = cv::Mat(image_size, image_size, CV_8UC3);
                cv::randu(test_image_, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
                return;
            }
            
            cv::Mat frame;
            // Skip to middle frame for better representative sample
            int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
            int middle_frame = total_frames / 2;
            cap.set(cv::CAP_PROP_POS_FRAMES, middle_frame);
            
            if (!cap.read(frame) || frame.empty()) {
                std::cout << "⚠️  Could not read frame from " << input_path << ", using random input" << std::endl;
                test_image_ = cv::Mat(image_size, image_size, CV_8UC3);
                cv::randu(test_image_, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
                return;
            }
            
            cv::resize(frame, test_image_, cv::Size(image_size, image_size));
            std::cout << "🎬 Loaded frame from video: " << input_path 
                      << " (frame " << middle_frame << "/" << total_frames 
                      << ", resized to " << image_size << "x" << image_size << ")" << std::endl;
            cap.release();
            return;
        }
        
        // Try to load as image
        cv::Mat img = cv::imread(input_path);
        if (img.empty()) {
            std::cout << "⚠️  Could not load " << input_path << ", using random input" << std::endl;
            test_image_ = cv::Mat(image_size, image_size, CV_8UC3);
            cv::randu(test_image_, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
            return;
        }
        
        cv::resize(img, test_image_, cv::Size(image_size, image_size));
        std::cout << "🖼️  Loaded image: " << input_path 
                  << " (resized to " << image_size << "x" << image_size << ")" << std::endl;
    }
    
    SimpleBenchmarkResult benchmark_yolo11() {
        std::cout << "\n🚀 Benchmarking YOLO11 (CPU)..." << std::endl;
        
        try {
            // Create YOLO11 detector (CPU only for stability)
            YOLO11Detector detector(model_path_, labels_path_, false);
            
            std::vector<double> run_times;
            int total_detections = 0;
            
            // Warmup runs
            std::cout << "Warming up..." << std::endl;
            for (int i = 0; i < warmup_runs_; ++i) {
                try {
                    auto detections = detector.detect(test_image_);
                } catch (const std::exception& e) {
                    std::cout << "Warmup " << i << " failed: " << e.what() << std::endl;
                }
            }
            
            // Timed runs
            std::cout << "Running " << num_runs_ << " timed iterations..." << std::endl;
            int successful_runs = 0;
            
            for (int i = 0; i < num_runs_; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                
                try {
                    auto detections = detector.detect(test_image_);
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    
                    run_times.push_back(duration.count() / 1000.0); // Convert to milliseconds
                    total_detections += detections.size();
                    successful_runs++;
                    
                    if ((i + 1) % 5 == 0) {
                        std::cout << "Completed " << (i + 1) << "/" << num_runs_ << " runs" << std::endl;
                    }
                    
                } catch (const std::exception& e) {
                    std::cout << "Run " << i << " failed: " << e.what() << std::endl;
                    break;
                }
            }
            
            int avg_detections = (successful_runs > 0) ? (total_detections / successful_runs) : 0;
            auto result = SimpleStats::calculate_stats("YOLO11 (Integrated)", run_times, avg_detections);
            
            if (result.success) {
                std::cout << "✅ YOLO11 completed: " << std::fixed << std::setprecision(1) 
                          << result.mean_time_ms << "±" << result.std_time_ms 
                          << " ms, " << result.fps << " FPS, " 
                          << avg_detections << " detections/image" << std::endl;
            }
            
            return result;
            
        } catch (const std::exception& e) {
            SimpleBenchmarkResult result;
            result.backend_name = "YOLO11 (Integrated)";
            result.success = false;
            result.error_message = e.what();
            std::cout << "❌ YOLO11 failed: " << e.what() << std::endl;
            return result;
        }
    }
    
    SimpleBenchmarkResult benchmark_opencv_dnn() {
        std::cout << "\n🚀 Benchmarking OpenCV DNN..." << std::endl;
        
        try {
            cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path_);
            
            if (net.empty()) {
                SimpleBenchmarkResult result;
                result.backend_name = "OpenCV DNN";
                result.success = false;
                result.error_message = "Failed to load model";
                return result;
            }
            
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            
            std::vector<double> run_times;
            
            // Warmup runs
            std::cout << "Warming up..." << std::endl;
            for (int i = 0; i < warmup_runs_; ++i) {
                cv::Mat blob;
                cv::dnn::blobFromImage(test_image_, blob, 1.0/255.0, 
                                     cv::Size(640, 640), cv::Scalar(0,0,0), 
                                     true, false, CV_32F);
                net.setInput(blob);
                cv::Mat output = net.forward();
            }
            
            // Timed runs
            std::cout << "Running " << num_runs_ << " timed iterations..." << std::endl;
            for (int i = 0; i < num_runs_; ++i) {
                cv::Mat blob;
                cv::dnn::blobFromImage(test_image_, blob, 1.0/255.0, 
                                     cv::Size(640, 640), cv::Scalar(0,0,0), 
                                     true, false, CV_32F);
                
                auto start = std::chrono::high_resolution_clock::now();
                
                net.setInput(blob);
                cv::Mat output = net.forward();
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                run_times.push_back(duration.count() / 1000.0);
                
                if ((i + 1) % 5 == 0) {
                    std::cout << "Completed " << (i + 1) << "/" << num_runs_ << " runs" << std::endl;
                }
            }
            
            auto result = SimpleStats::calculate_stats("OpenCV DNN", run_times);
            
            if (result.success) {
                std::cout << "✅ OpenCV DNN completed: " << std::fixed << std::setprecision(1) 
                          << result.mean_time_ms << "±" << result.std_time_ms 
                          << " ms, " << result.fps << " FPS" << std::endl;
            }
            
            return result;
            
        } catch (const std::exception& e) {
            SimpleBenchmarkResult result;
            result.backend_name = "OpenCV DNN";
            result.success = false;
            result.error_message = e.what();
            std::cout << "❌ OpenCV DNN failed: " << e.what() << std::endl;
            return result;
        }
    }
    
    void run_benchmark() {
        std::cout << "\n🚀 YOLO Benchmark Suite - Multi-Backend Performance Analysis" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << "Model: " << model_path_ << std::endl;
        std::cout << "Labels: " << labels_path_ << std::endl;
        std::cout << "Runs: " << num_runs_ << " (Warmup: " << warmup_runs_ << ")" << std::endl;
        
        std::vector<SimpleBenchmarkResult> results;
        
        // Run benchmarks
        results.push_back(benchmark_yolo11());
        results.push_back(benchmark_opencv_dnn());
        
        // Print summary table
        print_results_table(results);
    }
    
private:
    void print_results_table(const std::vector<SimpleBenchmarkResult>& results) {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "📋 Benchmark Results Summary" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        // Table header
        std::cout << std::left 
                  << std::setw(20) << "Backend"
                  << std::setw(10) << "Status"
                  << std::setw(15) << "Time (ms)"
                  << std::setw(10) << "FPS"
                  << std::setw(15) << "Detections"
                  << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        // Results rows
        for (const auto& result : results) {
            std::string status = result.success ? "✅" : "❌";
            
            std::cout << std::left 
                      << std::setw(20) << result.backend_name
                      << std::setw(10) << status;
            
            if (result.success) {
                std::cout << std::setw(15) << (std::to_string(static_cast<int>(result.mean_time_ms)) + 
                                              "±" + std::to_string(static_cast<int>(result.std_time_ms)))
                          << std::setw(10) << static_cast<int>(result.fps)
                          << std::setw(15) << result.detection_count;
            } else {
                std::cout << std::setw(15) << "-"
                          << std::setw(10) << "-"
                          << std::setw(15) << "-";
            }
            std::cout << std::endl;
        }
        
        std::cout << std::string(80, '=') << std::endl;
        
        // Find best performer
        auto best_result = std::min_element(results.begin(), results.end(),
            [](const SimpleBenchmarkResult& a, const SimpleBenchmarkResult& b) {
                if (!a.success) return false;
                if (!b.success) return true;
                return a.mean_time_ms < b.mean_time_ms;
            });
        
        if (best_result != results.end() && best_result->success) {
            std::cout << "\n🏆 Best Performance: " << best_result->backend_name 
                      << " - " << std::fixed << std::setprecision(1) 
                      << best_result->mean_time_ms << " ms ("
                      << static_cast<int>(best_result->fps) << " FPS)" << std::endl;
        }
    }
};

// Main function
int main(int argc, char* argv[]) {
    std::cout << "🚀 YOLO Benchmark Suite - Professional Multi-Backend Benchmarking" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <labels_path> [options]" << std::endl;
        std::cout << "\nOptions:" << std::endl;
        std::cout << "  --input <path>        Input image file (optional)" << std::endl;
        std::cout << "  --image-size <size>   Input image size (default: 640)" << std::endl;
        std::cout << "  --runs <num>          Number of benchmark runs (default: 20)" << std::endl;
        std::cout << "  --warmup <num>        Number of warmup runs (default: 5)" << std::endl;
        std::cout << "\nExamples:" << std::endl;
        std::cout << "  " << argv[0] << " models/yolo11n.onnx models/coco.names" << std::endl;
        std::cout << "  " << argv[0] << " models/yolo11n.onnx models/coco.names --input data/dog.jpg --runs 10" << std::endl;
        return 1;
    }
    
    // Parse arguments
    std::string model_path = argv[1];
    std::string labels_path = argv[2];
    std::string input_path = "";
    int image_size = 640;
    int num_runs = 20;
    int warmup_runs = 5;
    
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--image-size" && i + 1 < argc) {
            image_size = std::stoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            num_runs = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup_runs = std::stoi(argv[++i]);
        }
    }
    
    try {
        SimpleBenchmarkRunner runner(model_path, labels_path, input_path, 
                                   image_size, num_runs, warmup_runs);
        runner.run_benchmark();
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
