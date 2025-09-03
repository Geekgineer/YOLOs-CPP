# Use NVIDIA's CUDA base image with Python
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip \
    && pip install --no-cache-dir ultralytics[export]


# Set working directory
WORKDIR /workspace
