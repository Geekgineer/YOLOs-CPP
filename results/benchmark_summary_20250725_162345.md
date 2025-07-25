# YOLOs-CPP Benchmark Summary

**Date**: Fri Jul 25 16:23:45 UTC 2025
**Environment**: AMD_EPYC_7543_32-Core_Processor, 120 cores, 967511MB RAM
**GPU**: NVIDIA_GeForce_RTX_5090

## System Specifications
- **CPU**: AMD_EPYC_7543_32-Core_Processor
- **Cores**: 120
- **Memory**: 967511MB
- **GPU**: NVIDIA_GeForce_RTX_5090

## Benchmark Results

The following files contain the detailed benchmark results:

- image_benchmark_20250725_161429.csv (2025-07-25 16:21)
- video_benchmark_20250725_162134.csv (2025-07-25 16:23)

## Key Findings

- Performance measurements include FPS, latency, and resource utilization
- Both CPU and GPU environments tested (where available)
- Multiple iteration counts tested for statistical significance
- System monitoring included during benchmarks

## Files Generated

drwxr-xr-x  3 root root   188 Jul 25 16:23 .
drwxr-xr-x 13 root root  4096 Jul 25 16:12 ..
drwxr-xr-x  2 root root   152 Jul 25 16:23 analysis_20250725_162342
-rw-r--r--  1 root root   519 Jul 25 16:23 benchmark_summary_20250725_162345.md
-rw-r--r--  1 root root 12545 Jul 25 16:21 image_benchmark_20250725_161429.csv
-rw-r--r--  1 root root  2363 Jul 25 16:23 video_benchmark_20250725_162134.csv

## Usage

To view detailed analysis, examine the CSV files or run:
```bash
python3 benchmark/analyze_results.py results/benchmark_results.csv
```

