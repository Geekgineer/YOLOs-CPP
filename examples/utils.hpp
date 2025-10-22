/**
 * @file utils.hpp
 * @brief Utility functions for examples (timestamps, file saving, argument parsing)
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace utils {

/**
 * @brief Get current timestamp string in format: YYYYMMDD_HHMMSS
 * @return Timestamp string
 */
inline std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
    
    // Cross-platform localtime
    #ifdef _WIN32
        localtime_s(&tm_now, &time_t_now);  // Windows
    #else
        localtime_r(&time_t_now, &tm_now);  // POSIX (Linux/macOS)
    #endif
    
    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y%m%d_%H%M%S");
    return oss.str();
}

/**
 * @brief Generate output filename with timestamp
 * @param inputPath Original input file path
 * @param outputDir Output directory (e.g., "outputs/det/")
 * @param suffix Additional suffix (e.g., "_result")
 * @return Full output path with timestamp
 */
inline std::string getOutputPath(const std::string& inputPath, 
                                 const std::string& outputDir,
                                 const std::string& suffix = "_result") {
    namespace fs = std::filesystem;
    
    // Ensure output directory exists
    fs::create_directories(outputDir);
    
    // Get input filename without extension
    fs::path inputFilePath(inputPath);
    std::string baseName = inputFilePath.stem().string();
    std::string extension = inputFilePath.extension().string();
    
    // Create output filename: basename_timestamp_suffix.ext
    std::string timestamp = getTimestamp();
    std::string outputFilename = baseName + "_" + timestamp + suffix + extension;
    
    return (fs::path(outputDir) / outputFilename).string();
}

/**
 * @brief Save image with timestamp
 * @param image Image to save
 * @param inputPath Original input path
 * @param outputDir Output directory
 * @return Path where image was saved
 */
inline std::string saveImage(const cv::Mat& image, 
                             const std::string& inputPath,
                             const std::string& outputDir) {
    std::string outputPath = getOutputPath(inputPath, outputDir);
    cv::imwrite(outputPath, image);
    return outputPath;
}

/**
 * @brief Save video writer output path with timestamp
 * @param inputPath Original input path
 * @param outputDir Output directory
 * @return Output path for video
 */
inline std::string getVideoOutputPath(const std::string& inputPath,
                                      const std::string& outputDir) {
    return getOutputPath(inputPath, outputDir, "_result");
}

/**
 * @brief Print performance metrics
 * @param taskName Name of the task
 * @param durationMs Duration in milliseconds
 * @param fps Frames per second (optional)
 */
inline void printMetrics(const std::string& taskName, 
                        int64_t durationMs, 
                        double fps = -1) {
    std::cout << "\n═══════════════════════════════════════" << std::endl;
    std::cout << "  " << taskName << " Metrics" << std::endl;
    std::cout << "═══════════════════════════════════════" << std::endl;
    std::cout << "  Inference Time: " << durationMs << " ms" << std::endl;
    if (fps > 0) {
        std::cout << "  FPS: " << std::fixed << std::setprecision(2) << fps << std::endl;
    }
    std::cout << "═══════════════════════════════════════\n" << std::endl;
}

/**
 * @brief Check if file extension is supported image format
 */
inline bool isImageFile(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || 
            ext == ".bmp" || ext == ".tiff" || ext == ".tif");
}

/**
 * @brief Check if file extension is supported video format
 */
inline bool isVideoFile(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".mp4" || ext == ".avi" || ext == ".mov" || 
            ext == ".mkv" || ext == ".flv" || ext == ".wmv");
}

/**
 * @brief Print usage information
 */
inline void printUsage(const std::string& programName, 
                      const std::string& taskType,
                      const std::string& defaultModel,
                      const std::string& defaultInput,
                      const std::string& defaultLabels) {
    std::cout << "\n╔════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  YOLOs-CPP " << taskType << " Example" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nUsage: " << programName << " [model_path] [input_path] [labels_path]" << std::endl;
    std::cout << "\nArguments:" << std::endl;
    std::cout << "  model_path   : Path to ONNX model (default: " << defaultModel << ")" << std::endl;
    std::cout << "  input_path   : Image/video file or directory (default: " << defaultInput << ")" << std::endl;
    std::cout << "  labels_path  : Path to class labels file (default: " << defaultLabels << ")" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << programName << std::endl;
    std::cout << "  " << programName << " ../models/yolo11n.onnx" << std::endl;
    std::cout << "  " << programName << " ../models/yolo11n.onnx ../data/image.jpg" << std::endl;
    std::cout << "  " << programName << " ../models/yolo11n.onnx ../data/ ../models/coco.names" << std::endl;
    std::cout << std::endl;
}

} // namespace utils

#endif // UTILS_HPP
