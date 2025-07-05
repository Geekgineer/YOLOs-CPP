FROM ubuntu:22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        curl \
        pkg-config \
        libopencv-dev \
        ca-certificates \
        python3 \
        python3-pip \
        # GUI dependencies for OpenCV imshow
        libgtk-3-dev \
        libcanberra-gtk3-module \
        && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Copy the entire project (for initial build; docker-compose will mount for dev)
COPY . /workspace

# Make sure build.sh is executable
RUN chmod +x /workspace/build.sh

# Build the project (will be rebuilt on container start for live code updates)
# RUN ./build.sh

# Default command (can be overridden by docker-compose)
CMD bash -c "./build.sh && cd build && ./image_inference"   
