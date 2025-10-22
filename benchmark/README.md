# YOLO Benchmarking Tools

This directory contains professional benchmarking tools for YOLO models with advanced system monitoring and performance analysis.

## Platform Support

- **Linux**: Full support with GPU acceleration (CUDA + cuDNN)
- **macOS**: CPU-only support (Apple Silicon and Intel)
- **Windows**: Experimental support with CPU/GPU
- **Cloud/Colab**: Supported with GPU runtime enabled

## Prerequisites

### System Requirements
- CMake 3.16+
- OpenCV 4.6+
- ONNX Runtime 1.20.1+
- C++17 compatible compiler

### For GPU Benchmarking on Linux/Colab:
- NVIDIA GPU with CUDA Toolkit 12.x
- cuDNN library (optional, for optimized performance)
- Enable GPU runtime in Colab: Runtime > Change runtime type > GPU

### Models and Data
- Models: Place in `../models/` (e.g., `yolo11n.onnx`, `yolov8n.onnx`)
- Quantized models: Place in `../quantized_models/`
- Test data: Place in `../data/` (e.g., `dog.jpg`, `dogs.mp4`)
- Labels: `../models/coco.names`

## Building

### Main Project Build
```bash
# CPU build (default)
./build.sh

# GPU build (downloads GPU ONNX Runtime)
./build.sh 1.20.1 1
```

### Benchmark-Specific Build
The benchmark is built separately to avoid modifying the main CMake:

```bash
# Navigate to benchmark directory
cd benchmark

# Create build directory
mkdir build && cd build

# Configure with CMake (use GPU ONNX Runtime if built)
cmake .. -D ONNXRUNTIME_DIR=../../onnxruntime-linux-x64-1.20.1  # CPU
# or
cmake .. -D ONNXRUNTIME_DIR=../../onnxruntime-linux-x64-gpu-1.20.1  # GPU

# Build
make

# Copy executable to main build directory
cp yolo_performance_analyzer ../build/
```

### Permission Fix (if needed)
```bash
chmod +x ../build/yolo_performance_analyzer
```

## Tools

### 1. YOLO Performance Analyzer (`yolo_performance_analyzer`)
**Professional comprehensive benchmarking tool with advanced system monitoring**

**Features:**
- Multiple modes: `image`, `video`, `camera`, `comprehensive`
- Advanced system monitoring (CPU, GPU, memory usage)
- Detailed performance metrics and CSV output
- Automated comprehensive testing
- Latency analysis (min/max/average)
- Real-time resource monitoring
- Dynamic GPU/CPU detection (uses GPU if available, falls back to CPU)
- Suppressed debug output for clean runs

**Usage:**
```bash
# Always run from project root for correct paths
cd /path/to/YOLOs-CPP

# Single image benchmark
./build/yolo_performance_analyzer image yolo11 detection models/yolo11n.onnx models/coco.names data/dog.jpg --iterations=100

# Video processing benchmark
./build/yolo_performance_analyzer video yolo11 detection models/yolo11n.onnx models/coco.names data/dogs.mp4

# Camera benchmark (if camera available)
./build/yolo_performance_analyzer camera yolo11 detection models/yolo11n.onnx models/coco.names 0

# Automated comprehensive testing (all combinations)
./build/yolo_performance_analyzer comprehensive
```

### 2. YOLO Benchmark Suite (`yolo_benchmark_suite`)
**Professional multi-backend benchmarking tool for quick performance comparison**

**Features:**
- Multi-backend support (ONNX Runtime + OpenCV DNN)
- Quick performance comparison
- Clean tabular output
- Statistical analysis (mean, std deviation)
- Flexible configuration options

**Usage:**
```bash
# Basic benchmark
./build/yolo_benchmark_suite models/yolo11n.onnx models/coco.names

# Custom configuration
./build/yolo_benchmark_suite models/yolo11n.onnx models/coco.names --input data/dog.jpg --runs 50 --warmup 10
```

## Build Targets

```bash
# Build all tools
./build.sh

# Individual targets (after CMake configuration)
make yolo_performance_analyzer
make yolo_benchmark_suite
```

## Output

- **Performance Analyzer**: Generates timestamped CSV files in `results/` directory with device-specific rows (cpu/gpu)
- **Benchmark Suite**: Displays formatted results table in terminal

## Troubleshooting

### Common Issues
- **"No such file or directory"**: Ensure executable is in `build/` and run from project root
- **"Permission denied"**: Run `chmod +x build/yolo_performance_analyzer`
- **"Model not found"**: Check model paths; run from project root
- **GPU not detected**: Ensure CUDA is installed and GPU runtime is enabled in Colab
- **Build fails**: Verify ONNX Runtime and OpenCV are installed

### Colab-Specific Notes
- Mount Google Drive: `from google.colab import drive; drive.mount('/content/drive')`
- Build may take longer; use GPU runtime for faster execution
- Models/data must be in mounted drive for persistence

Both tools support professional workflows and provide detailed performance insights for YOLO model optimization.
