# YOLOs-CPP Benchmark Summary

**Date**: Mon Jul 21 01:29:39 PM EEST 2025
**Environment**: Intel(R)_Core(TM)_i7-8850H_CPU_@_2.60GHz, 12 cores, 15672MB RAM
**GPU**: Quadro_P1000

## System Specifications
- **CPU**: Intel(R)_Core(TM)_i7-8850H_CPU_@_2.60GHz
- **Cores**: 12
- **Memory**: 15672MB
- **GPU**: Quadro_P1000

## Benchmark Results

The following files contain the detailed benchmark results:

- video_benchmark_20250721_132716.csv (2025-07-21 13:29)
- image_benchmark_20250721_132150.csv (2025-07-21 13:27)

## Key Findings

- Performance measurements include FPS, latency, and resource utilization
- Both CPU and GPU environments tested (where available)
- Multiple iteration counts tested for statistical significance
- System monitoring included during benchmarks

## Files Generated

drwxrwxr-x  3 elbahnasy elbahnasy  4096 Jul 21 13:29 .
drwxrwxr-x 14 elbahnasy elbahnasy  4096 Jul 21 12:59 ..
drwxrwxr-x  2 elbahnasy elbahnasy  4096 Jul 21 13:29 analysis_20250721_132935
-rw-rw-r--  1 elbahnasy elbahnasy   515 Jul 21 13:29 benchmark_summary_20250721_132939.md
-rw-rw-r--  1 elbahnasy elbahnasy 13000 Jul 21 13:27 image_benchmark_20250721_132150.csv
-rw-rw-r--  1 elbahnasy elbahnasy  2554 Jul 21 13:29 video_benchmark_20250721_132716.csv

## Usage

To view detailed analysis, examine the CSV files or run:
```bash
python3 benchmark/analyze_results.py results/benchmark_results.csv
```

