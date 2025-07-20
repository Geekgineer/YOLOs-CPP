# Use a CUDA 12.x and cuDNN 9 base image for the builder stage
FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04 AS builder

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set NVIDIA environment variables for GPU access
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install system dependencies. libopencv-dev on Ubuntu 22.04 installs version 4.5.x
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

# --- Install ONNX Runtime v1.20.0 ---
WORKDIR /opt
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-gpu-1.20.0.tgz && \
    tar -xvf onnxruntime-linux-x64-gpu-1.20.0.tgz && \
    mv onnxruntime-linux-x64-gpu-1.20.0 onnxruntime && \
    rm onnxruntime-linux-x64-gpu-1.20.0.tgz

# Set the library path so the system can find the ONNX Runtime .so files
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# --- Build the YOLOs-CPP Project ---
WORKDIR /app
COPY . .

# Build the project with Release configuration
RUN cmake -S . -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D ONNXRUNTIME_DIR=/opt/onnxruntime && \
    cmake --build build --config Release -j$(nproc)

# --- Final Production Image ---
# Use the smaller -runtime image for the final stage to reduce image size.
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

# Set environment variables again for the final image
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libopencv-highgui4.5d \
    libopencv-videoio4.5d && \
    rm -rf /var/lib/apt/lists/*

# Copy the pre-built ONNX Runtime and the compiled project from the builder stage
WORKDIR /app
COPY --from=builder /opt/onnxruntime /opt/onnxruntime
COPY --from=builder /app/build ./build
COPY --from=builder /app/data ./data
COPY --from=builder /app/models ./models
COPY --from=builder /app/run_*.sh .

# Create a non-root user for security
RUN useradd -ms /bin/bash yolouser && \
    usermod -aG video yolouser && \
    chown -R yolouser:yolouser /app

USER yolouser

CMD ["/bin/bash"]
