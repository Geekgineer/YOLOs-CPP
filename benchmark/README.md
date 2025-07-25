# YOLO Benchmarking Tools

This directory contains professional benchmarking tools for YOLO models.

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

**Usage:**
```bash
# Single image benchmark
./build/yolo_performance_analyzer image yolo11 detection models/yolo11n.onnx models/coco.names data/dog.jpg --cpu --iterations=100

# Video processing benchmark
./build/yolo_performance_analyzer video yolo11 detection models/yolo11n.onnx models/coco.names data/dogs.mp4 --cpu

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

- **Performance Analyzer**: Generates timestamped CSV files in `results/` directory
- **Benchmark Suite**: Displays formatted results table in terminal

Both tools support professional workflows and provide detailed performance insights for YOLO model optimization.
