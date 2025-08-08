# YOLO Benchmarking Tools

This directory contains a professional benchmarking tool for YOLO models with advanced system monitoring and performance analysis.

## Platform Support

- **Linux**: Full support with GPU acceleration (CUDA + cuDNN)
- **macOS**: CPU-only support (Apple Silicon and Intel)
- **Windows**: Experimental support with CPU/GPU

## Prerequisites

For GPU benchmarking on Linux:
- NVIDIA GPU with CUDA Toolkit 12.x
- cuDNN library (optional, for optimized performance)
- See `../BENCHMARK.md` for detailed installation instructions

## Tool

### YOLO Performance Analyzer (`yolo_performance_analyzer`)
**Professional comprehensive benchmarking tool with advanced system monitoring**

**Description:**
This tool consolidates all benchmarking functionality into a single, powerful application. It replaces the need for multiple tools by providing comprehensive performance analysis with various modes and backend support.

**Features:**
- Multiple modes: `image`, `video`, `camera`, `comprehensive`
- Advanced system monitoring (CPU, GPU, memory usage)
- Detailed performance metrics and CSV output
- Automated comprehensive testing
- Latency analysis (min/max/average)
- Real-time resource monitoring
- Multi-backend support (ONNX Runtime + OpenCV DNN)
- Statistical analysis (mean, std deviation)
- Flexible configuration options
- Cross-platform support (Linux, macOS, Windows)

**Usage:**
```bash
# Single image benchmark
./build/yolo_performance_analyzer image yolo11 detection models/yolo11n.onnx models/coco.names data/dog.jpg --cpu --iterations=100

# Video processing benchmark
./build/yolo_performance_analyzer video yolo11 detection models/yolo11n.onnx models/coco.names data/dogs.mp4 --cpu

# Automated comprehensive testing (all combinations)
./build/yolo_performance_analyzer comprehensive

# Quick benchmark with custom parameters
./build/yolo_performance_analyzer image yolo11 detection models/yolo11n.onnx models/coco.names data/dog.jpg --gpu --iterations=50
```


## Build Target

```bash
# Build the benchmarking tool
./build.sh

# Individual target (after CMake configuration)
make yolo_performance_analyzer
```

## Output

**Performance Analyzer**: Generates timestamped CSV files in `results/` directory with comprehensive performance metrics and detailed analysis for YOLO model optimization.
