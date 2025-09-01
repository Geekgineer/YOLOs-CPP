##nvidia/cuda for GPU support or ubuntu:22.04 for CPU
FROM ubuntu:22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# Install essential tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    ca-certificates \
    # OpenCV (headless)
    libopencv-dev \
    # Media support
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    # Math libraries (choose one approach)
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


# Set the working directory
WORKDIR /workspace
