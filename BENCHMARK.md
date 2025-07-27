# YOLOs-CPP Guide for Benchmarking 

## Quick Start - Basic Setup

### 1. Install Dependencies
```bash
# Update system and install required packages
apt update && apt install -y cmake git libopencv-dev python3-pip
```

### 2. Install cuDNN (GPU Support)
```bash
# Install cuDNN from .deb package
dpkg -i cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb
cp /var/cudnn-local-repo-ubuntu2204-9.10.2/*.gpg /usr/share/keyrings/
apt-key add /usr/share/keyrings/*.gpg
apt update
apt install -y libcudnn9-cuda-12=9.10.2.21-1 libcudnn9-dev-cuda-12=9.10.2.21-1 libcudnn9-headers-cuda-12=9.10.2.21-1
ldconfig
```

### 3. Download ONNX Runtime
```bash
# Download and extract ONNX Runtime GPU
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.20.1.tgz
```

### 4. Clone and Build Project
```bash
# Clone repository
git clone https://github.com/Elbhnasy/YOLOs-CPP.git
cd YOLOs-CPP

# Build the project
./build.sh
```

### 5. Run Comprehensive Benchmark
```bash
# Run full benchmark suite (tests all model configurations)
./build/yolo_performance_analyzer comprehensive
```

**Output**: 
- Comprehensive CSV results saved in `results/` directory
- File format: `comprehensive_benchmark_[timestamp].csv`
- Contains detailed performance metrics including FPS, latency, CPU/GPU usage, memory consumption
- Benchmarks both original and quantized models automatically
- Professional-grade performance analysis suitable for production environments

**Check Results**:
```bash
# View results directory
ls results/

# Sample CSV output
head -5 results/comprehensive_benchmark_*.csv
```

