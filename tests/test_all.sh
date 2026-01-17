#!/bin/bash
# ============================================================================
# YOLOs-CPP - Run All Tests
# ============================================================================
# Note: Not using set -e so we can continue running other tests even if one fails

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/test_utils.sh"

print_header "YOLOs-CPP Complete Test Suite"

# Track results
PASSED=0
FAILED=0
SKIPPED=0

run_test() {
    local name="$1"
    local script="$2"
    
    echo ""
    echo -e "${BLUE}>>> Running $name tests...${NC}"
    
    if [ -f "$script" ]; then
        # Run the test script and capture exit code
        bash "$script"
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            print_success "$name tests passed"
            PASSED=$((PASSED + 1))
        else
            print_error "$name tests failed (exit code: $exit_code)"
            FAILED=$((FAILED + 1))
        fi
    else
        echo -e "${YELLOW}Skipped: $script not found${NC}"
        SKIPPED=$((SKIPPED + 1))
    fi
}

# ============================================================================
# Run All Tests
# ============================================================================

run_test "Detection" "$SCRIPT_DIR/test_detection.sh"
run_test "Classification" "$SCRIPT_DIR/test_classification.sh"
run_test "Segmentation" "$SCRIPT_DIR/test_segmentation.sh"
run_test "Pose" "$SCRIPT_DIR/test_pose.sh"
run_test "OBB" "$SCRIPT_DIR/test_obb.sh"

# ============================================================================
# Summary
# ============================================================================
print_header "Test Summary"

echo -e "${GREEN}Passed:  $PASSED${NC}"
echo -e "${RED}Failed:  $FAILED${NC}"
echo -e "${YELLOW}Skipped: $SKIPPED${NC}"

if [ "$FAILED" -gt 0 ]; then
    print_error "Some tests failed!"
    exit 1
else
    print_success "All tests passed!"
    exit 0
fi
