# YOLOs-CPP Guide for Benchmarking


## Quick Start - Setup

### 1. Install Dependencies (Root/Sudo Required)

```bash
# Update system and install required packages
apt update && apt upgrade -y
apt install -y cmake git libopencv-dev python3-pip curl wget build-essential

# Install additional dependencies for server environment
apt install -y pkg-config libssl-dev ca-certificates
```

---

### 2. Install CUDA and GPU Support (For GPU Servers)

#### For Cloud/Server GPU Instances (Ubuntu 20.04/22.04)

```bash
# Check if NVIDIA driver is already installed
nvidia-smi

# If not installed, install NVIDIA driver
apt install -y nvidia-driver-535 nvidia-utils-535

# Install CUDA Toolkit via package manager
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt-get install -y cuda-toolkit-12-4

# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Reboot to ensure driver is loaded
reboot
```

#### For RunPod/Vast.ai/Paperspace

```bash
# Verify installation
nvcc --version
nvidia-smi

# If missing, install essentials
apt update && apt install -y cmake git libopencv-dev python3-pip curl wget
```

---

### 3. Windows

```powershell
# Download and install CUDA Toolkit:
# https://developer.nvidia.com/cuda-downloads

# Download and install cuDNN:
# https://developer.nvidia.com/cudnn
# Extract to CUDA installation directory
```

---

## 4. Export YOLO Models to ONNX

Before running benchmarks, export your YOLO models to ONNX format.

### Export with ONNX Opset 11

```bash
python3 models/export_onnx_11.py
```

### Export with ONNX Opset 8

```bash
python3 models/export_onnx_8.py
```

> **Note:**
>
> * The generated `.onnx` models will be in the `models/` directory.
> * Install dependencies first (`ultralytics`, `torch`, etc.).
> * These models are required for benchmarking.

---

## 5. Prepare Test Data

You must ensure the following files exist before running benchmarks:

* `data/dog.jpg` — sample image
* `data/dogs.mp4` — sample video

If missing, download them:

---

## 6. Deploy and Build YOLOs-CPP

### Linux / macOS

```bash
# Clone repository
git clone https://github.com/Elbhnasy/YOLOs-CPP.git
cd YOLOs-CPP

# Make build script executable and run
chmod +x build.sh
./build.sh
```

### Windows (PowerShell / CMD)

```powershell
# Clone repository
git clone https://github.com/Elbhnasy/YOLOs-CPP.git
cd YOLOs-CPP

# Run build script (downloads ONNX Runtime automatically)
bash build.sh
```

> On Windows, ensure you have **CMake**, **Visual Studio Build Tools**, and **OpenCV** installed.

---

## 7. Run Benchmarks

### Quick benchmark (image, CPU mode)

```bash
./build/yolo_performance_analyzer image yolo11 detection models/yolo11n.onnx models/coco.names data/dog.jpg --cpu --iterations=5
```

### GPU benchmark (image)

```bash
nvidia-smi && ./build/yolo_performance_analyzer image yolo11 detection models/yolo11n.onnx models/coco.names data/dog.jpg --gpu --iterations=50
```

### Full benchmark suite

```bash
./build/yolo_performance_analyzer comprehensive
```

---

## 8. Viewing Results

```bash
ls results/
head -5 results/comprehensive_benchmark_*.csv
```

---
