##nvidia/cuda for GPU support or ubuntu:22.04 for CPU
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# Install essential tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    curl \
    pkg-config \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    # OpenCV runtime dependencies (w/o CUDA build)
    libopencv-dev \
    libgtk-3-dev \
    libcanberra-gtk3-module \
    # Media support
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    # Optimization libraries
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Copy source code into the image
COPY . /workspace

# Make sure build.sh is executable
RUN chmod +x /workspace/build.sh

# Environment variable to choose which target to run
ENV INFERENCE_TARGET=camera_inference

# Default command
CMD bash -c "./build.sh && cd build && ./${INFERENCE_TARGET}"
