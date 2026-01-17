#!/bin/bash

# ============================================================================
# YOLOs-CPP Release Preparation Script
# ============================================================================
# This script packages EXISTING validated models from the tests directory
# for a GitHub release. These are the same models used in testing.
#
# Usage:
#   ./scripts/prepare_release.sh [output_dir]
#
# This will create zip files ready for upload to GitHub releases:
#   - yolo-detection-models.zip
#   - yolo-segmentation-models.zip
#   - yolo-pose-models.zip
#   - yolo-obb-models.zip
#   - yolo-classification-models.zip
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${1:-$PROJECT_ROOT/release_assets}"
TESTS_DIR="$PROJECT_ROOT/tests"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  YOLOs-CPP Release Preparation${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}Output directory: $OUTPUT_DIR${NC}"
echo -e "${YELLOW}Source: $TESTS_DIR (existing validated models)${NC}"
echo ""

# ============================================================================
# Package existing models from tests directory
# ============================================================================

package_models() {
    local task="$1"
    local zip_name="$2"
    local models_dir="$TESTS_DIR/$task/models"
    
    echo -e "${BLUE}Packaging $task models...${NC}"
    
    if [[ ! -d "$models_dir" ]]; then
        echo -e "${RED}  ✗ Directory not found: $models_dir${NC}"
        return 1
    fi
    
    # Find .pt files
    local pt_files=$(find "$models_dir" -maxdepth 1 -name "*.pt" 2>/dev/null)
    
    if [[ -z "$pt_files" ]]; then
        echo -e "${YELLOW}  ! No .pt files found in $models_dir${NC}"
        return 0
    fi
    
    # Create zip
    cd "$models_dir"
    local count=$(ls -1 *.pt 2>/dev/null | wc -l)
    
    if [[ $count -gt 0 ]]; then
        zip -j "$OUTPUT_DIR/$zip_name" *.pt
        echo -e "${GREEN}  ✓ $zip_name created ($count models)${NC}"
        ls -1 *.pt | sed 's/^/      - /'
    fi
    
    cd "$PROJECT_ROOT"
}

# ============================================================================
# Main
# ============================================================================

echo -e "${YELLOW}Packaging existing validated models from tests/...${NC}"
echo ""

package_models "detection" "yolo-detection-models.zip"
package_models "segmentation" "yolo-segmentation-models.zip"
package_models "pose" "yolo-pose-models.zip"
package_models "obb" "yolo-obb-models.zip"
package_models "classification" "yolo-classification-models.zip"

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Release Assets Summary${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "Assets created in: ${GREEN}$OUTPUT_DIR${NC}"
echo ""

if ls "$OUTPUT_DIR"/*.zip 1> /dev/null 2>&1; then
    ls -lh "$OUTPUT_DIR"/*.zip
else
    echo -e "${RED}No zip files created. Make sure models exist in tests/*/models/${NC}"
    echo ""
    echo "To download models first, run the test scripts:"
    echo "  cd tests && ./test_detection.sh"
    echo "  cd tests && ./test_segmentation.sh"
    echo "  etc."
fi

echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Go to GitHub → Releases → Create new release"
echo "2. Tag version: v1.0.0-models"
echo "3. Upload all .zip files from $OUTPUT_DIR"
echo ""
