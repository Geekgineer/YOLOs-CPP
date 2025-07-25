# YOLOs-CPP Benchmark Analysis Report

**Generated**: 2025-07-25 16:23:45.617077
**Total Benchmarks**: 4

## Model Performance Summary

| Model | FPS (avg) | Latency (avg) | CPU % (avg) | GPU % (avg) | Memory (avg) |
|-------|-----------|---------------|-------------|-------------|-------------|
| yolo11 | 15.10 | 47.85 | 3.61 | 2.45 | 587.32 |
| yolo8 | 14.00 | 60.87 | 3.88 | 2.52 | 564.31 |


## Environment Comparison

| Environment | FPS (avg) | FPS (std) | Latency (avg) | Latency (std) | Load Time (avg) |
|-------------|-----------|-----------|---------------|---------------|----------------|
| CPU | 8.29 | 1.26 | 89.68 | 18.24 | 63.47 |
| GPU | 20.81 | 0.30 | 19.05 | 0.17 | 393.24 |


## Key Insights

- **Best FPS Performance**: yolo11 on GPU with 21.02 FPS
- **Lowest Latency**: yolo11 on GPU with 18.93ms
- **Most Resource Efficient**: yolo11 on GPU
- **GPU Performance Delta**: 151.0% improvement over CPU

## Recommendations

- For **maximum throughput**: Use GPU environment with the best performing model
- For **cost efficiency**: Consider CPU environment for non-critical applications
- For **real-time applications**: Focus on models with lowest latency
- For **resource-constrained environments**: Use the most efficient model/environment combination

## Charts Generated

- `performance_comparison.png`: Overall performance comparison
- `detailed_analysis.png`: Detailed performance analysis
- `cost_efficiency.png`: Cost-efficiency analysis
