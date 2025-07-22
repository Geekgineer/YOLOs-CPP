#pragma once

// ===================================
// RTX 4090 Optimized YOLOv11 Detector
// ===================================
//
// Optimized configuration for RTX 4090 with maximum GPU utilization
// Addresses ONNX Runtime warnings and improves GPU performance
//

#include "det/YOLO11.hpp"
#include <cuda_provider_factory.h>

/**
 * @brief RTX 4090 optimized YOLO11 detector with enhanced CUDA configuration
 */
class YOLO11DetectorRTX4090 : public YOLO11Detector {
public:
    YOLO11DetectorRTX4090(const std::string &modelPath, const std::string &labelsPath, bool useGPU = true);
    
private:
    void setupOptimizedCUDA();
};

// Optimized implementation for RTX 4090
inline YOLO11DetectorRTX4090::YOLO11DetectorRTX4090(const std::string &modelPath, const std::string &labelsPath, bool useGPU) {
    // Initialize ONNX Runtime environment with minimal logging for performance
    env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "ONNX_DETECTION_RTX4090");
    sessionOptions = Ort::SessionOptions();

    // RTX 4090 specific optimizations
    sessionOptions.SetIntraOpNumThreads(1);  // Let CUDA handle parallelism
    sessionOptions.SetInterOpNumThreads(1);  // Reduce CPU-GPU synchronization
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // Disable CPU fallback for better error detection
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    
    // Enable memory pattern optimization
    sessionOptions.EnableMemPattern();
    sessionOptions.EnableCpuMemArena();
    
    // Optimize for throughput over latency
    sessionOptions.AddConfigEntry("session.disable_prepacking", "0");
    sessionOptions.AddConfigEntry("session.use_env_allocators", "1");

    // RTX 4090 specific CUDA configuration
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");

    if (useGPU && cudaAvailable != availableProviders.end()) {
        std::cout << "Inference device: RTX 4090 GPU (Optimized)" << std::endl;
        
        // Advanced CUDA configuration for RTX 4090
        OrtCUDAProviderOptions cudaOptions;
        cudaOptions.device_id = 0;
        cudaOptions.arena_extend_strategy = 1;  // Extend by larger chunks
        cudaOptions.gpu_mem_limit = SIZE_MAX;   // Use all available GPU memory
        cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cudaOptions.do_copy_in_default_stream = 1;
        cudaOptions.has_user_compute_stream = 0;
        cudaOptions.default_memory_arena_cfg = nullptr;
        
        // RTX 4090 specific optimizations
        std::unordered_map<std::string, std::string> provider_options = {
            {"enable_cuda_graph", "1"},                    // Enable CUDA graphs for RTX 4090
            {"cuda_graph_enable_building", "1"},           // Build CUDA graphs
            {"enable_skip_layer_norm_strict_mode", "0"},   // Allow optimizations
            {"prefer_nhwc", "1"},                          // Use NHWC layout for better performance
            {"enable_cuda_graph_capture", "1"},            // Capture CUDA graphs
            {"cuda_graph_capture_allow_build", "1"}        // Allow graph building
        };
        
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        
        // Add provider options
        for (const auto& option : provider_options) {
            sessionOptions.AddConfigEntry(("ep.cuda." + option.first).c_str(), option.second.c_str());
        }
        
    } else {
        if (useGPU) {
            std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        }
        std::cout << "Inference device: CPU" << std::endl;
    }

    // Load the model
#ifdef _WIN32
    std::wstring w_modelPath(modelPath.begin(), modelPath.end());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    // Continue with standard initialization
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShapeVec = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    
    // Set up input/output tensor information
    modelInputHeight = inputTensorShapeVec[2];
    modelInputWidth = inputTensorShapeVec[3];
    
    // Get input and output names
    inputName = std::string(session.GetInputNameAllocated(0, allocator).get());
    outputName = std::string(session.GetOutputNameAllocated(0, allocator).get());
    
    // Load class labels
    std::ifstream ifs(labelsPath.c_str());
    std::string line;
    while (getline(ifs, line)) {
        classNames.push_back(line);
    }
    
    std::cout << "Model loaded successfully with " << session.GetInputCount() 
              << " input nodes and " << session.GetOutputCount() << " output nodes." << std::endl;
}
