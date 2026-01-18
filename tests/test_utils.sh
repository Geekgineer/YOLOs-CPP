#!/bin/bash
# ============================================================================
# YOLOs-CPP Test Utilities
# Shared functions for all test scripts
# ============================================================================

# Colors for output
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export BLUE='\033[0;34m'
export NC='\033[0m' # No Color

# Virtual environment directory
VENV_DIR="${TEST_VENV_DIR:-$HOME/.yolos-cpp-test-venv}"

# ============================================================================
# Install uv (fast Python package installer)
# ============================================================================
install_uv() {
    if command -v uv &> /dev/null; then
        echo -e "${GREEN}uv is already installed${NC}"
        return 0
    fi
    
    echo -e "${YELLOW}Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add to PATH
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    
    if command -v uv &> /dev/null; then
        echo -e "${GREEN}uv installed successfully${NC}"
        return 0
    else
        echo -e "${RED}Failed to install uv${NC}"
        return 1
    fi
}

# ============================================================================
# Setup virtual environment
# ============================================================================
setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${BLUE}Creating virtual environment at $VENV_DIR${NC}"
        if command -v uv &> /dev/null; then
            uv venv "$VENV_DIR"
        else
            python3 -m venv "$VENV_DIR"
        fi
    fi
    
    # Activate the virtual environment
    source "$VENV_DIR/bin/activate"
    echo -e "${GREEN}Virtual environment activated${NC}"
}

# ============================================================================
# Install Python packages using uv (with pip fallback)
# ============================================================================
install_python_packages() {
    local packages="$@"
    
    echo -e "${BLUE}Installing Python packages: $packages${NC}"
    
    # Ensure venv is set up and activated
    setup_venv
    
    # Try uv first (within the venv)
    if command -v uv &> /dev/null; then
        echo "Using uv..."
        uv pip install $packages --quiet && return 0
    fi
    
    # Fallback to pip (venv's pip)
    echo "Using pip..."
    pip install -q $packages && return 0
    
    echo -e "${RED}Failed to install packages${NC}"
    return 1
}

# ============================================================================
# Export PyTorch models to ONNX
# ============================================================================
export_models_to_onnx() {
    local models_dir="$1"
    local export_script="$2"
    
    cd "$models_dir" || return 1
    
    echo -e "${BLUE}Exporting models to ONNX...${NC}"
    
    # Ensure venv is activated
    setup_venv
    
    # Try the export script first
    if [ -f "$export_script" ]; then
        python3 "$export_script" cpu 2>&1 && return 0
    fi
    
    # Fallback: export each .pt file individually
    echo "Using individual export fallback..."
    for pt_file in *.pt; do
        if [ -f "$pt_file" ]; then
            onnx_file="${pt_file%.pt}.onnx"
            if [ ! -f "$onnx_file" ]; then
                echo "Exporting $pt_file -> $onnx_file"
                python3 -c "
from ultralytics import YOLO
model = YOLO('$pt_file')
model.export(format='onnx', opset=12, simplify=True, imgsz=320)
" 2>&1 || echo -e "${YELLOW}Warning: Failed to export $pt_file${NC}"
            else
                echo "Skipping $pt_file (ONNX already exists)"
            fi
        fi
    done
    
    # Verify we have ONNX models
    local onnx_count=$(ls -1 *.onnx 2>/dev/null | wc -l)
    if [ "$onnx_count" -eq 0 ]; then
        echo -e "${RED}ERROR: No ONNX models available${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Found $onnx_count ONNX model(s)${NC}"
    return 0
}

# ============================================================================
# Print section header
# ============================================================================
print_header() {
    local title="$1"
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  $title${NC}"
    echo -e "${BLUE}============================================${NC}"
}

# ============================================================================
# Print success message
# ============================================================================
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# ============================================================================
# Print error message
# ============================================================================
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ============================================================================
# Download test images if missing
# Downloads sample images from Ultralytics assets for testing
# ============================================================================
download_test_images() {
    local images_dir="$1"
    local test_type="$2"  # detection, pose, obb, segmentation, classification
    local original_dir="$(pwd)"
    
    mkdir -p "$images_dir"
    cd "$images_dir" || return 1
    
    # Remove any .REMOVED.git-id placeholder files
    rm -f *.REMOVED.git-id 2>/dev/null || true
    
    # Check if we have valid images (not placeholder files)
    local valid_images=$(find . -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) ! -name "*.REMOVED*" 2>/dev/null | wc -l)
    
    if [ "$valid_images" -gt 0 ]; then
        echo -e "${GREEN}Found $valid_images existing test image(s)${NC}"
        cd "$original_dir"
        return 0
    fi
    
    echo -e "${YELLOW}Downloading test images for $test_type...${NC}"
    
    # Ultralytics sample images (public domain / freely available)
    case "$test_type" in
        "detection"|"segmentation")
            # Standard COCO-style test images
            curl -sL "https://ultralytics.com/images/bus.jpg" -o "bus.jpg" 2>/dev/null || true
            curl -sL "https://ultralytics.com/images/zidane.jpg" -o "zidane.jpg" 2>/dev/null || true
            ;;
        "pose")
            # Images with people for pose estimation
            curl -sL "https://ultralytics.com/images/zidane.jpg" -o "test1.jpg" 2>/dev/null || true
            curl -sL "https://ultralytics.com/images/bus.jpg" -o "test2.jpg" 2>/dev/null || true
            ;;
        "obb")
            # Aerial/rotated images for OBB
            curl -sL "https://ultralytics.com/images/bus.jpg" -o "image.png" 2>/dev/null || true
            ;;
        "classification")
            # Any image works for classification
            curl -sL "https://ultralytics.com/images/bus.jpg" -o "bus.jpg" 2>/dev/null || true
            ;;
        *)
            echo -e "${YELLOW}Unknown test type: $test_type${NC}"
            curl -sL "https://ultralytics.com/images/bus.jpg" -o "test.jpg" 2>/dev/null || true
            ;;
    esac
    
    # Verify we have images
    valid_images=$(find . -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) ! -name "*.REMOVED*" 2>/dev/null | wc -l)
    
    cd "$original_dir"
    
    if [ "$valid_images" -eq 0 ]; then
        echo -e "${RED}ERROR: Failed to download test images${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Downloaded $valid_images test image(s)${NC}"
    return 0
}
