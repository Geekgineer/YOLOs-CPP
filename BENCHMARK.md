# YOLOs-CPP Guide for Benchmarking 

## Server Deployment Guide

### Prerequisites Check
```bash
# Check if you're on a server
uname -a
lscpu | head -20
free -h

# Check for GPU (if available)
nvidia-smi || echo "No NVIDIA GPU detected"

# Check internet connectivity
curl -I https://github.com 2>/dev/null | head -1 || echo "No internet connection"
```

## Quick Start - Server Setup

### 1. Install Dependencies (Root/Sudo Required)
```bash
# Update system and install required packages
apt update && apt upgrade -y
apt install -y cmake git libopencv-dev python3-pip curl wget build-essential

# Install additional dependencies for server environment
apt install -y pkg-config libssl-dev ca-certificates
```

### 2. Install CUDA and GPU Support (For GPU Servers)

#### For Cloud/Server GPU Instances (Ubuntu 20.04/22.04)
```bash
# Check if NVIDIA driver is already installed
nvidia-smi

# If not installed, install NVIDIA driver
apt install -y nvidia-driver-535 nvidia-utils-535

# Install CUDA Toolkit via package manager (recommended for servers)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt-get install -y cuda-toolkit-12-4

# Add CUDA to PATH (for current session and future sessions)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Reboot to ensure driver is properly loaded
reboot
```

#### For RunPod/Vast.ai/Paperspace (Pre-configured GPU)
```bash
# Most GPU cloud providers come with CUDA pre-installed
# Verify installation
nvcc --version
nvidia-smi

# If missing, install just the essentials
apt update && apt install -y cmake git libopencv-dev python3-pip curl wget
```
```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA toolkit
sudo apt-get install cuda-toolkit-12-4
```

#### Install cuDNN (Optional, for optimized GPU performance)
```bash
# Download cuDNN from NVIDIA Developer Portal
# Register at https://developer.nvidia.com/cudnn
# Download cuDNN Library for Linux (x86_64)

# Example installation for cuDNN 9.x
dpkg -i cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb
cp /var/cudnn-local-repo-ubuntu2204-9.10.2/*.gpg /usr/share/keyrings/
apt-key add /usr/share/keyrings/*.gpg
apt update
apt install -y libcudnn9-cuda-12=9.10.2.21-1 libcudnn9-dev-cuda-12=9.10.2.21-1 libcudnn9-headers-cuda-12=9.10.2.21-1
ldconfig
```

#### Windows
```bash
# Download and install CUDA Toolkit from NVIDIA:
# https://developer.nvidia.com/cuda-downloads

# Download and install cuDNN:
# https://developer.nvidia.com/cudnn
# Extract to CUDA installation directory
```

#### macOS
```bash
# CUDA is not supported on macOS with Apple Silicon
# For Intel Macs, download legacy CUDA from NVIDIA
# Recommended: Use CPU-only mode
```

### 3. Deploy and Build YOLOs-CPP
```bash
# Clone repository
cd /opt  # or /home/username - choose a persistent location
git clone https://github.com/Elbhnasy/YOLOs-CPP.git
cd YOLOs-CPP

# Build the project (automatically downloads ONNX Runtime)
chmod +x build.sh
./build.sh

# Verify build success
ls -la build/
# Should show: yolo_performance_analyzer, image_inference, video_inference, camera_inference
```

### 4. Prepare Test Data (Server Environment)
```bash
# Check if sample data exists
ls -la data/

# If no sample data, download test images/videos
mkdir -p data
cd data

# Download sample image
wget https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg -O dog.jpg

# Download sample video (optional)
wget https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4 -O dogs.mp4

cd ..
```

### 5. Run Comprehensive Benchmark (Server Optimized)
```bash
```bash
# Quick benchmark test (5 iterations for fast results)
./build/yolo_performance_analyzer image yolo11 detection models/yolo11n.onnx models/coco.names data/dog.jpg --cpu --iterations=5

# Full comprehensive benchmark (all configurations)
./build/yolo_performance_analyzer comprehensive

# GPU benchmark (if GPU available)
nvidia-smi && ./build/yolo_performance_analyzer image yolo11 detection models/yolo11n.onnx models/coco.names data/dog.jpg --gpu --iterations=50

# Monitor system resources during benchmark
htop &  # or top
./build/yolo_performance_analyzer comprehensive
```

## Server-Specific Commands

### For Headless Servers (No GUI)
```bash
# All benchmarks run in terminal - no GUI needed
# Results saved to CSV files in results/ directory

# Check results
ls -la results/
head -10 results/comprehensive_benchmark_*.csv

# Copy results to local machine (if needed)
# From your local machine:
# scp user@server:/opt/YOLOs-CPP/results/*.csv ./local_results/
```

### For Docker Containers
```bash
# If running in Docker, mount results directory
# docker run -v $(pwd)/results:/opt/YOLOs-CPP/results ...

# Inside container:
cd /opt/YOLOs-CPP
./build/yolo_performance_analyzer comprehensive

# Results will be available in mounted volume
```

### For Cloud GPU Instances (RunPod/Vast.ai/etc.)
```bash
# Start persistent session
screen -S yolo_benchmark

# Run long benchmarks
./build/yolo_performance_analyzer comprehensive

# Detach from screen: Ctrl+A then D
# Reattach later: screen -r yolo_benchmark

# Download results via Jupyter (if available)
# Or use file manager in cloud interface
```

## Performance Monitoring

### Monitor GPU Usage
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Log GPU usage during benchmark
nvidia-smi dmon -s u -o T &
./build/yolo_performance_analyzer comprehensive
pkill nvidia-smi
```

### Monitor CPU and Memory
```bash
# Install htop if not available
apt install -y htop

# Monitor during benchmark
htop &
./build/yolo_performance_analyzer comprehensive
```

### System Information Collection
```bash
# Collect system info for benchmark context
echo "=== System Information ===" > system_info.txt
uname -a >> system_info.txt
lscpu >> system_info.txt
free -h >> system_info.txt
df -h >> system_info.txt
nvidia-smi >> system_info.txt 2>/dev/null || echo "No GPU" >> system_info.txt

# Include with benchmark results
cat system_info.txt
```

**Note**: The build script automatically downloads and configures ONNX Runtime, so no manual installation is required.

## Local Development Setup (Alternative)

### 1. Install Dependencies
```

**Note**: The build script automatically downloads and configures ONNX Runtime GPU, so no manual installation is required.
```bash
# Clone repository
git clone https://github.com/Elbhnasy/YOLOs-CPP.git
cd YOLOs-CPP

# Build the project
./build.sh
```

### 4. Run Comprehensive Benchmark
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

## Server Troubleshooting

### Common Issues and Solutions

#### 1. Permission Denied Errors
```bash
# If you get permission errors
sudo chown -R $USER:$USER /opt/YOLOs-CPP
chmod +x build.sh

# Or run with sudo (not recommended for production)
sudo ./build.sh
```

#### 2. CUDA Not Found
```bash
# Check CUDA installation
which nvcc
echo $PATH
echo $LD_LIBRARY_PATH

# Fix PATH if needed
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Test CUDA
nvcc --version
nvidia-smi
```

#### 3. Out of Memory (GPU)
```bash
# Check GPU memory
nvidia-smi

# Run with smaller batch sizes or CPU mode
./build/yolo_performance_analyzer image yolo11 detection models/yolo11n.onnx models/coco.names data/dog.jpg --cpu --iterations=10

# Clear GPU memory
nvidia-smi --gpu-reset-clocks=1
```

#### 4. Network/Download Issues
```bash
# If ONNX Runtime download fails
wget --spider https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz

# Manual download if needed
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.20.1.tgz
```

#### 5. Missing Models
```bash
# Check if models exist
ls -la models/

# Download missing models (example)
cd models/
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.onnx
cd ..
```

### Resource Monitoring Commands
```bash
# Monitor all resources during benchmark
watch -n 2 "echo '=== CPU ==='; top -n 1 | head -10; echo '=== Memory ==='; free -h; echo '=== GPU ==='; nvidia-smi | head -20"

# Log resource usage to file
(while true; do date; nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv; sleep 5; done) > gpu_usage.log &

# Run benchmark
./build/yolo_performance_analyzer comprehensive

# Stop logging
pkill -f gpu_usage.log
```

### Quick Server Validation
```bash
# Full system check before benchmarking
echo "=== System Validation ==="
echo "OS: $(lsb_release -d 2>/dev/null || uname -a)"
echo "CPU: $(nproc) cores"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'None')"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'Not installed')"
echo "Disk Space: $(df -h . | tail -1 | awk '{print $4}') available"
echo "Internet: $(curl -s --max-time 5 https://github.com && echo 'OK' || echo 'Failed')"
```

