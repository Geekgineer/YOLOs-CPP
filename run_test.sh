#! /bin/bash

# Default values
TEST_TASK="${1:-0}" # 0: detection, 1: classification, 2: segmentation, 3: pose, 4: obb, 5: all


# Function to display usage
usage() {
    echo "Usage: $0 [TEST_TASK] "
    echo
    echo "This script runs the specified test task for YOLOs-CPP."
    echo
    echo "Arguments:"
    echo "  TEST_TASK            The test task to build (0 for detection, 1 for classification, 2 for segmentation, 3 for pose, 4 for obb, 5 for all, default: 0)."
    echo
    echo "Examples:"
    echo "  $0 0          # Runs detection tests."
    echo "  $0 5        # Runs all tests."
    echo
    exit 1
}

# Show usage if help is requested
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

cd tests

case "$TEST_TASK" in
    0)
        echo "Running detection tests..."
        ./test_detection.sh
        ;;
    1)
        echo "Running classification tests..."
        ./test_classification.sh
        ;;
    2)
        echo "Running segmentation tests..."
        ./test_segmentation.sh
        ;;
    3)
        echo "Running pose tests..."
        ./test_pose.sh
        ;;
    4)
        echo "Running obb tests..."
        ./test_obb.sh
        ;;
    5)
        echo "Running all tests..."
        ./test_detection.sh
        ./test_classification.sh
        ./test_segmentation.sh
        ./test_pose.sh
        ./test_obb.sh
        ;;
    *)
        echo "Invalid TEST_TASK: $TEST_TASK"
        usage
        ;;
esac

