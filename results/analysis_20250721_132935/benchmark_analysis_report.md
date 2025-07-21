# YOLOs-CPP Benchmark Analysis Report

**Generated**: 2025-07-21 13:29:39.538369
**Total Benchmarks**: 4

## Model Performance Summary

| Model | FPS (avg) | Latency (avg) | CPU % (avg) | GPU % (avg) | Memory (avg) |
|-------|-----------|---------------|-------------|-------------|-------------|
| yolo11 | 10.32 | 73.92 | 53.87 | 0.00 | 132.14 |
| yolo8 | 10.37 | 72.87 | 53.80 | 0.00 | 108.02 |


## Environment Comparison

| Environment | FPS (avg) | FPS (std) | Latency (avg) | Latency (std) | Load Time (avg) |
|-------------|-----------|-----------|---------------|---------------|----------------|
| CPU | 10.23 | 0.84 | 74.27 | 6.19 | 75.53 |
| GPU | 10.45 | 0.92 | 72.53 | 7.67 | 69.57 |


## Key Insights

- **Best FPS Performance**: yolo8 on GPU with 11.11 FPS
- **Lowest Latency**: yolo8 on GPU with 67.10ms
- **Most Resource Efficient**: yolo8 on GPU
- **GPU Performance Delta**: 2.2% improvement over CPU

## Recommendations

- For **maximum throughput**: Use GPU environment with the best performing model
- For **cost efficiency**: Consider CPU environment for non-critical applications
- For **real-time applications**: Focus on models with lowest latency
- For **resource-constrained environments**: Use the most efficient model/environment combination

## Charts Generated

- `performance_comparison.png`: Overall performance comparison
- `detailed_analysis.png`: Detailed performance analysis
- `cost_efficiency.png`: Cost-efficiency analysis
