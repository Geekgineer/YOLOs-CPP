# Docker Guide for YOLOs-CPP

This guide covers building and running YOLOs-CPP with Docker for consistent, portable deployments.

## Prerequisites

### Install Docker

1. **Download Docker Desktop**: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

2. **Verify installation**:
   ```bash
   docker --version
   docker run hello-world
   ```

3. **For GPU support** (NVIDIA only):
   ```bash
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

---

## Quick Start

### Build Images

```bash
# GPU version (requires NVIDIA GPU)
docker build -t yolos-cpp:gpu -f Dockerfile .

# CPU version (works everywhere)
docker build -t yolos-cpp:cpu -f Dockerfile.cpu .
```

### Run Inference

```bash
# CPU inference
docker run --rm -it yolos-cpp:cpu

# GPU inference
docker run --gpus all --rm -it yolos-cpp:gpu
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `INFERENCE_TARGET` | Executable to run | `image_inference` |
| `MODEL_PATH` | Path to ONNX model | `models/yolo11n.onnx` |
| `INPUT_PATH` | Path to input file | `data/dog.jpg` |
| `LABELS_PATH` | Path to class labels | `models/coco.names` |

### Examples

```bash
# Video inference
docker run --rm -it \
    -e INFERENCE_TARGET=video_inference \
    -e INPUT_PATH=data/sample.mp4 \
    yolos-cpp:cpu

# Camera inference (requires device access)
docker run --rm -it \
    --device=/dev/video0 \
    -e INFERENCE_TARGET=camera_inference \
    -e INPUT_PATH=0 \
    yolos-cpp:cpu

# Custom model
docker run --rm -it \
    -v /path/to/your/model.onnx:/app/models/custom.onnx \
    -e MODEL_PATH=models/custom.onnx \
    yolos-cpp:cpu
```

---

## GUI Support

### Linux

```bash
docker run --rm -it \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    yolos-cpp:cpu
```

### macOS

```bash
# Install XQuartz first: https://www.xquartz.org/
xhost +localhost
docker run --rm -it \
    -e DISPLAY=host.docker.internal:0 \
    yolos-cpp:cpu
```

### Windows

1. Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/)
2. Launch VcXsrv with "Disable access control" checked
3. Run:
   ```powershell
   docker run --rm -it -e DISPLAY=host.docker.internal:0.0 yolos-cpp:cpu
   ```

---

## Development

### Build with Custom ONNX Runtime Version

```bash
docker build \
    --build-arg ONNXRUNTIME_VERSION=1.19.0 \
    -t yolos-cpp:custom .
```

### Interactive Development

```bash
docker run --rm -it \
    -v $(pwd):/workspace \
    -w /workspace \
    yolos-cpp:cpu \
    /bin/bash
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `permission denied` | Run with `--privileged` or fix permissions |
| `GPU not found` | Ensure NVIDIA Container Toolkit is installed |
| `Display not found` | Configure X11 forwarding (see GUI Support) |
| `Model not found` | Mount your models with `-v` flag |

### Verify GPU Access

```bash
docker run --gpus all --rm nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

---

## Image Sizes

| Image | Size | Features |
|-------|------|----------|
| `yolos-cpp:gpu` | ~4.5 GB | CUDA + cuDNN |
| `yolos-cpp:cpu` | ~800 MB | CPU only |

---

## Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  yolos:
    build: .
    environment:
      - INFERENCE_TARGET=image_inference
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Run with:
```bash
docker-compose up
```
