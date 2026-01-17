# YOLO Unified Benchmark Tool

A comprehensive benchmarking tool that combines **performance metrics** (FPS, latency, memory) and **accuracy metrics** (mAP, AP50) for YOLO detection and segmentation models.

## Features

### Holistic Benchmarking
- **Performance Metrics**: FPS, latency (min/max/avg), memory usage, CPU/GPU utilization
- **Accuracy Metrics**: mAP, AP50, AP50-95 (when ground truth available)
- **Unified Output**: CSV files + beautiful tabular terminal reports

### Supported Modes
- `image`: Single image performance benchmarking
- `video`: Video file performance benchmarking
- `camera`: Real-time camera performance benchmarking
- `evaluate`: Accuracy evaluation on dataset with ground truth
- `comprehensive`: Automated multi-model/config testing

### Task Support
- **Detection**: Object detection models (YOLOv5-v12, YOLO26)
- **Segmentation**: Instance segmentation models
- **Pose**: Human pose estimation models
- **OBB**: Oriented bounding box detection
- **Classification**: Image classification models

### Dataset Support
- **COCOVal2017**: Standard COCO validation dataset
- **Custom Datasets**: Any folder with images + YOLO-format labels

## Prerequisites

### System Requirements
- CMake 3.16+
- OpenCV 4.6+
- ONNX Runtime 1.20.1+
- C++17 compatible compiler

### For GPU Benchmarking
- NVIDIA GPU with CUDA Toolkit 12.x
- cuDNN library (optional, for optimized performance)

### Models and Data
- Models: Place in `../models/` (e.g., `yolo11n.onnx`, `yolov8n.onnx`, `yolo26n.onnx`)
- Test data: Place in `../data/` (e.g., `dog.jpg`, `dogs.mp4`)
- Labels: `../models/coco.names`
- Evaluation dataset: COCOVal2017 or custom dataset with ground truth labels

## Building

### Quick Build (Recommended)
```bash
# Navigate to benchmarks directory
cd benchmarks

# Run auto_bench.sh (downloads ONNX Runtime, builds, and runs benchmarks)
./auto_bench.sh [ONNXRUNTIME_VERSION] [ONNXRUNTIME_GPU] [MODELS] [EVAL_DATASET]

# Examples:
./auto_bench.sh 1.20.1 0                    # CPU build
./auto_bench.sh 1.20.1 1                    # GPU build
./auto_bench.sh 1.20.1 0 yolo11n,yolov8n,yolo26n   # Test specific models
```

### Manual Build
```bash
# Navigate to benchmarks directory
cd benchmarks

# Create build directory
mkdir build && cd build

# Configure with CMake (use GPU ONNX Runtime if built)
cmake .. -D ONNXRUNTIME_DIR=../../onnxruntime-linux-x64-1.20.1  # CPU
# or
cmake .. -D ONNXRUNTIME_DIR=../../onnxruntime-linux-x64-gpu-1.20.1  # GPU

# Build
make

# Copy executable to main build directory (optional)
cp yolo_unified_benchmark ../../build/
```

## Usage

### Performance Benchmarking (No Ground Truth Required)

#### Single Image Benchmark
```bash
# Always run from project root for correct paths
cd /path/to/YOLOs-CPP

./benchmarks/build/yolo_unified_benchmark image yolo11 detection \
  models/yolo11n.onnx models/coco.names data/dog.jpg \
  --iterations=100 --gpu
```

#### Video Processing Benchmark
```bash
./benchmarks/build/yolo_unified_benchmark video yolo11 detection \
  models/yolo11n.onnx models/coco.names data/dogs.mp4 --gpu
```

#### Camera Benchmark
```bash
./benchmarks/build/yolo_unified_benchmark camera yolo11 detection \
  models/yolo11n.onnx models/coco.names 0 \
  --duration=30 --gpu
```

#### Segmentation Benchmark
```bash
./benchmarks/build/yolo_unified_benchmark image yolo11 segmentation \
  models/yolo11n-seg.onnx models/coco.names data/dog.jpg \
  --iterations=100 --gpu
```

### Accuracy Evaluation (Requires Ground Truth)

#### COCOVal2017 Evaluation
```bash
./benchmarks/build/yolo_unified_benchmark evaluate yolo11 detection \
  models/yolo11n.onnx models/coco.names \
  ../val2017 ../labels_val2017 \
  --gpu --dataset-type=coco
```

#### Custom Dataset Evaluation
```bash
./benchmarks/build/yolo_unified_benchmark evaluate yolo11 detection \
  models/yolo11n.onnx models/coco.names \
  /path/to/images /path/to/labels \
  --gpu --dataset-type=custom
```

### Comprehensive Automated Testing
```bash
# Tests all configured models on CPU and GPU
./benchmarks/build/yolo_unified_benchmark comprehensive
```

This will:
- Test all models in the configuration list
- Run on both CPU and GPU
- Generate CSV results file
- Display tabular comparison summary

## Command-Line Options

### General Options
- `--gpu`: Use GPU acceleration
- `--cpu`: Force CPU execution
- `--threads=N`: Number of threads (default: 1)
- `--quantized`: Indicate quantized model

### Performance Options
- `--iterations=N`: Number of iterations for image mode (default: 100)
- `--duration=N`: Duration in seconds for video/camera (default: 30)
- `--conf-threshold=N`: Confidence threshold for inference (default: 0.4)
- `--nms-threshold=N`: NMS threshold (default: 0.7)

### Evaluation Options
- `--eval-conf-threshold=N`: Confidence threshold for evaluation (default: 0.001)
- `--dataset-type=coco|custom`: Dataset type (default: custom)

## Output

### CSV Output
Results are saved to `results/unified_benchmark_TIMESTAMP.csv` with the following columns:
- Model configuration (model_type, task_type, device, precision)
- Performance metrics (load_time, inference_time, fps, latency, memory)
- System metrics (CPU usage, GPU usage, GPU memory)
- Accuracy metrics (AP50, mAP50-95) - when available

### Terminal Output

#### Detailed Report (Single Run)
```
================================================================================
DETAILED BENCHMARK REPORT
================================================================================
Model: yolo11 (detection)
Device: GPU (CUDA)
Input Type: Image
--------------------------------------------------------------------------------

PERFORMANCE METRICS:
  Load Time:        245.123 ms
  Inference Time:   12.456 ms (avg)
  Total Time:       12.789 ms (avg)
  FPS:              78.23
  Latency:          12.789 ms (avg), 10.123 ms (min), 15.456 ms (max)
  Memory Usage:     125.3 MB
  CPU Usage:        45.2%
  GPU Usage:        78.5%
  GPU Memory:       1024.5 MB
  Frames Processed: 100

ACCURACY METRICS:
  AP50:             0.5234
  mAP50-95:         0.4021
  IoU Thresholds:   0.5234 0.5142 0.4956 ...
================================================================================
```

#### Tabular Summary (Multiple Runs)
```
========================================================================================================================
BENCHMARK SUMMARY - COMPARISON TABLE
========================================================================================================================

Model                Task        Device   FPS        Latency(ms) AP50       mAP50-95   Memory(MB)
--------------------------------------------------------------------------------
yolo11               detection   CPU      45.23      22.12       N/A        N/A        98.5
yolo11               detection   GPU      78.45      12.75       0.5234     0.4021     125.3
yolo8                detection   CPU      42.11      23.75       N/A        N/A        95.2
yolo8                detection   GPU      75.23      13.29       0.5123     0.3956     120.1
========================================================================================================================
```

## Examples

### Example 1: Quick Performance Test
```bash
# Test YOLO11n on GPU with 200 iterations
./benchmarks/build/yolo_unified_benchmark image yolo11 detection \
  models/yolo11n.onnx models/coco.names data/dog.jpg \
  --iterations=200 --gpu
```

### Example 2: Full Accuracy Evaluation
```bash
# Evaluate YOLO11n on COCOVal2017
./benchmarks/build/yolo_unified_benchmark evaluate yolo11 detection \
  models/yolo11n.onnx models/coco.names \
  ../val2017 ../labels_val2017 \
  --gpu --dataset-type=coco
```

### Example 3: Compare Multiple Models
```bash
# Run comprehensive benchmark
./benchmarks/build/yolo_unified_benchmark comprehensive

# Results will show comparison table and save CSV
```

### Example 4: Segmentation Benchmark
```bash
# Benchmark segmentation model
./benchmarks/build/yolo_unified_benchmark image yolo11 segmentation \
  models/yolo11n-seg.onnx models/coco.names data/dog.jpg \
  --iterations=100 --gpu
```

### Example 5: YOLO26 Benchmark (End-to-End NMS-Free)
```bash
# YOLO26 detection - faster inference with built-in NMS
./benchmarks/build/yolo_unified_benchmark image yolo26 detection \
  models/yolo26n.onnx models/coco.names data/dog.jpg \
  --iterations=100 --gpu

# YOLO26 pose estimation
./benchmarks/build/yolo_unified_benchmark image yolo26 pose \
  models/yolo26n-pose.onnx models/coco.names data/person.jpg \
  --iterations=100 --gpu
```

## CSV Schema

The CSV output includes the following columns:

| Column | Description |
|--------|-------------|
| `model_type` | Model identifier (e.g., yolo11, yolo8) |
| `task_type` | Task type (detection, segmentation) |
| `InputType` | Input type (Image, Video, Camera) |
| `environment` | Runtime environment (CPU, CUDA) |
| `device` | Device used (cpu, gpu) |
| `threads` | Number of threads |
| `precision` | Model precision (fp32, int8) |
| `load_ms` | Model loading time (ms) |
| `preprocess_ms` | Preprocessing time (ms) |
| `inference_ms` | Inference time (ms) |
| `postprocess_ms` | Postprocessing time (ms) |
| `total_ms` | Total processing time (ms) |
| `fps` | Frames per second |
| `memory_mb` | Memory usage (MB) |
| `system_memory_mb` | System memory usage (MB) |
| `cpu_usage_%` | CPU usage percentage |
| `gpu_usage_%` | GPU usage percentage |
| `gpu_memory_mb` | GPU memory usage (MB) |
| `latency_avg_ms` | Average latency (ms) |
| `latency_min_ms` | Minimum latency (ms) |
| `latency_max_ms` | Maximum latency (ms) |
| `AP50` | Average Precision at IoU 0.5 (N/A if no GT) |
| `mAP50-95` | Mean Average Precision IoU 0.5-0.95 (N/A if no GT) |
| `frame_count` | Number of frames processed |

## Troubleshooting

### Common Issues

**"No such file or directory"**
- Ensure executable is in `build/` and run from project root
- Check that model paths are correct relative to project root

**"Permission denied"**
- Run `chmod +x benchmarks/build/yolo_unified_benchmark`

**"Model not found"**
- Verify model files exist in `models/` directory
- Check model path in command arguments

**"GPU not detected"**
- Ensure CUDA is installed and accessible
- Check `nvidia-smi` output
- Verify GPU ONNX Runtime is used (not CPU version)

**"Ground truth labels not found"**
- For COCOVal2017: Ensure `labels_val2017` folder exists
- For custom datasets: Ensure labels folder structure matches images
- Labels should be in YOLO format: `class_id cx cy w h` (normalized)

**Build fails**
- Verify ONNX Runtime and OpenCV are installed
- Check CMake version (3.16+)
- Ensure C++17 compiler is available

### Performance Tips

1. **Warmup**: First inference is slower due to initialization. The tool includes automatic warmup.

2. **Iterations**: For accurate performance metrics, use at least 100 iterations for image mode.

3. **GPU vs CPU**: GPU provides significant speedup but requires CUDA setup.

4. **Memory**: Large models may require more system memory. Monitor with system tools.

5. **Evaluation**: Accuracy evaluation on full COCOVal2017 (5000 images) takes time. Be patient.

## Integration with CI/CD

The unified benchmark tool can be integrated into CI/CD pipelines:

```bash
# Example CI script
cd benchmarks
./auto_bench.sh 1.20.1 0 yolo11n

# Check results
if [ -f "results/unified_benchmark_*.csv" ]; then
    echo "Benchmark completed successfully"
    # Parse CSV and check thresholds
else
    echo "Benchmark failed"
    exit 1
fi
```

## Advanced Usage

### Custom Model Configurations

Edit the `comprehensive` mode configuration in the source code to add custom models:

```cpp
std::vector<std::tuple<std::string, std::string, std::string>> test_configs = {
  {"yolo11", "detection", "models/yolo11n.onnx"},
  {"yolo11", "segmentation", "models/yolo11n-seg.onnx"},
  {"yolo26", "detection", "models/yolo26n.onnx"},
  {"yolo26", "pose", "models/yolo26n-pose.onnx"},
  {"yolo26", "obb", "models/yolo26n-obb.onnx"},
  // Add your models here
};
```

### Batch Evaluation

For batch evaluation of multiple models:

```bash
for model in yolo11n yolov8n yolo26n yolo11s; do
  ./benchmarks/build/yolo_unified_benchmark evaluate yolo11 detection \
    models/${model}.onnx models/coco.names \
    ../val2017 ../labels_val2017 --gpu
done
```

## License

Same as YOLOs-CPP project.

## Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- New features include tests
- Documentation is updated

## Support

For issues and questions:
- Check existing issues in the project repository`
- Review documentation in `doc/` directory
- Ensure prerequisites are met

