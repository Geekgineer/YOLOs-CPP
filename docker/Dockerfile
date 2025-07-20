# =================================================================================================
# Stage 1: Build Stage - Installs all dependencies and compiles the project
# =================================================================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set NVIDIA environment variables for GPU access
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install system dependencies in a single layer to keep the image lean
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

# --- Install ONNX Runtime ---
WORKDIR /opt
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-gpu-1.18.0.tgz && \
    tar -xvf onnxruntime-linux-x64-gpu-1.18.0.tgz && \
    mv onnxruntime-linux-x64-gpu-1.18.0 onnxruntime && \
    rm onnxruntime-linux-x64-gpu-1.18.0.tgz

# Set the library path so the system can find the ONNX Runtime .so files
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# --- Build the YOLOs-CPP Project ---
WORKDIR /app
COPY . .

RUN cmake -S . -B build \
      -D CMAKE_BUILD_TYPE=Release \
      -D ONNXRUNTIME_DIR=/opt/onnxruntime && \
    cmake --build build --config Release -j$(nproc)

# =================================================================================================
# Stage 2: Final Stage - Creates the final runnable image
# =================================================================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables again for the final image
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install OpenCV runtime libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the pre-built ONNX Runtime and the compiled project from the builder stage
WORKDIR /app
COPY --from=builder /opt/onnxruntime /opt/onnxruntime
COPY --from=builder /app/build ./build
COPY --from=builder /app/data ./data
COPY --from=builder /app/models ./models
COPY --from=builder /app/run_*.sh .

# Security: Create a non-root user, add to video group, and give ownership of the app directory
RUN useradd -ms /bin/bash yolouser && \
    usermod -aG video yolouser && \
    chown -R yolouser:yolouser /app

# Switch to the non-root user
USER yolouser

# Set the default command to an interactive bash shell
CMD ["/bin/bash"]