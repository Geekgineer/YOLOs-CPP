# YOLOs-CPP – Benchmarking & Setup Guide

This guide explains how to set up and run **YOLOs-CPP** benchmarks on **Linux**, **Windows**, and **GPU-enabled environments** such as **RunPod**, **Vast.ai**, and **Paperspace**.

---

## 🚀 1. Install Dependencies

### Common (All Environments)
```bash
apt update && apt upgrade -y
apt install -y cmake git libopencv-dev python3-pip curl wget build-essential pkg-config libssl-dev ca-certificates
````

---

## ⚡ 2. Install CUDA Toolkit & cuDNN

> **Important:**
>
> * **Bare Metal / VM** → You may install NVIDIA driver + CUDA Toolkit.
> * **Containers** (RunPod/Vast/Paperspace/Docker) → **Do not install NVIDIA driver**. Use the host driver with `--gpus all`.

---

### **A) Bare Metal / VM (Ubuntu 20.04 / 22.04)**

```bash
# Install NVIDIA driver (skip inside containers)
apt install -y nvidia-driver-535 nvidia-utils-535

# Install CUDA Toolkit 12.4
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y cuda-toolkit-12-4

# Install cuDNN 9 for CUDA 12
apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12

# Add CUDA paths
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Reboot to apply driver changes
reboot
```

---

### **B) Containers (RunPod / Vast.ai / Paperspace / Docker)**

```bash
# Install CUDA Toolkit without NVIDIA driver
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y --no-install-recommends cuda-toolkit-12-4

# Install cuDNN 9 for CUDA 12
apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12

# Configure library paths
printf "/usr/local/cuda/lib64\n/usr/local/cuda/targets/x86_64-linux/lib\n/usr/lib/x86_64-linux-gnu\n" > /etc/ld.so.conf.d/cuda.conf
ldconfig

# Export environment variables
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

Run container with GPU:

```bash
docker run --gpus all -e NVIDIA_VISIBLE_DEVICES=all --runtime=nvidia -it <image> bash
```

---

## 🧩 3. Export YOLO Models to ONNX

Export models with the required ONNX opsets:

```bash
python3 models/export_onnx_11.py  # Opset 11
python3 models/export_onnx_8.py   # Opset 8
```

> The `.onnx` files will be saved in the `models/` directory.
> Ensure you have installed `ultralytics` and `torch`.

---

## 📂 4. Prepare Test Data

Required files:

```
data/dog.jpg
data/dogs.mp4
```

Download them if missing.

---

## 🛠 5. Build YOLOs-CPP

### Linux / macOS

```bash
git clone https://github.com/Elbhnasy/YOLOs-CPP.git
cd YOLOs-CPP
chmod +x build.sh
./build.sh
```

### Windows (PowerShell / CMD)

```powershell
git clone https://github.com/Elbhnasy/YOLOs-CPP.git
cd YOLOs-CPP
bash build.sh
```

---

## 📊 6. Run Benchmarks

### Quick CPU Benchmark (Image)

```bash
./build/yolo_performance_analyzer image yolo11 detection models/yolo11n.onnx models/coco.names data/dog.jpg --cpu --iterations=5
```

### GPU Benchmark (Image)

```bash
nvidia-smi && ./build/yolo_performance_analyzer image yolo11 detection models/yolo11n.onnx models/coco.names data/dog.jpg --gpu --iterations=50
```

### Full Benchmark Suite

```bash
./build/yolo_performance_analyzer comprehensive
```

---

## 📑 7. View Results

```bash
ls results/
head -5 results/comprehensive_benchmark_*.csv
```

---


## 💡 Notes

* Inside containers, **never install NVIDIA drivers**; always use host GPU driver.
* Always match cuDNN version with your CUDA version.
* To run GPU workloads inside Docker, start the container with:
