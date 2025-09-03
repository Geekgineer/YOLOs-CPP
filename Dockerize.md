# 🚀 Dockerize YOLOs-CPP for Easy Deployment and Reproducibility

Easily run the YOLOs-CPP project using Docker for consistent and portable builds.

---

## 📦 Prerequisites: Install Docker

Before using Docker, ensure Docker Desktop is installed on your system.

### 🧰 1. Download Docker Desktop

👉 [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)

Choose the correct version for your operating system:
- **Windows**: `.exe`
- **macOS**: `.dmg`
- **Linux**: Follow distro-specific guide

### 🛠️ 2. Install Docker Desktop

- **Windows**: Run the `.exe` installer
- **macOS**: Drag Docker into `Applications`
- **Linux**: Follow [Linux installation guide](https://docs.docker.com/engine/install/)

### ▶️ 3. Start Docker

- Launch Docker Desktop from your applications menu
- Wait for it to start (Docker icon appears in tray)

### ✅ 4. Verify Installation

Run these commands in your terminal to verify:
```bash
docker --version
docker-compose --version
```

---

## 🏗️ Build and Run YOLOs-CPP with Docker

### 🔨 1. Build Docker Image

From the project root directory:
```bash
docker build -t yolos-cpp -f Dockerfile .
```

---

## 🖥️ Run the Project (Linux/macOS)

### 🧪 Inference Modes (CPU)

Choose your target mode:

**Image Inference**:
```bash
docker run --rm -it -e INFERENCE_TARGET=image_inference yolos-cpp
```

**Video Inference**:
```bash
docker run --rm -it -e INFERENCE_TARGET=video_inference yolos-cpp
```

### ⚡ Enable GPU Acceleration (NVIDIA GPU Required)

**Image Inference with GPU**:
```bash
docker run --gpus all --rm -it -e INFERENCE_TARGET=image_inference yolos-cpp
```

---

## 🪟 Run the Project (Windows with OpenCV GUI support)

### 🧩 2. Install VcXsrv for GUI (imshow)

Download and install [VcXsrv](https://sourceforge.net/projects/vcxsrv/), then:

- Launch **VcXsrv**
- Set:
  - **Display number**: `0`
  - ✅ Check **"Disable access control"**

### 🧰 3. Set Display Variable in PowerShell

```powershell
$env:DISPLAY = "host.docker.internal:0.0"
```

---

### 🚀 4. Run Docker with GUI (Windows)

**Image Inference**:
```powershell
docker run --rm -it -e DISPLAY=host.docker.internal:0.0 -e INFERENCE_TARGET=image_inference yolos-cpp
```

**Video Inference**:
```powershell
docker run --rm -it -e INFERENCE_TARGET=video_inference yolos-cpp
```

**Camera Inference**:
```powershell
docker run --rm -it -e INFERENCE_TARGET=camera_inference yolos-cpp
```

---

## ✅ Tips

- You can override `INFERENCE_TARGET` with any of:  
  `image_inference`, `video_inference`, `camera_inference`
- Use `--gpus all` **only if** your system supports NVIDIA GPU containers
- On Linux/macOS, GUI display is typically native; on Windows, VcXsrv is required for `imshow()`

---