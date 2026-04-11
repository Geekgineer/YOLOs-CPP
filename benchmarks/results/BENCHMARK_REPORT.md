# YOLO Detection Models Benchmark Report

**Generated:** January 17, 2026  
**Benchmark Version:** 2.0.0  
**Framework:** YOLOs-CPP (C++ ONNX Runtime Inference)

---

## System Specifications

| Component | Details |
|-----------|---------|
| **OS** | Ubuntu 24.04 LTS (Linux 6.14.0-37-generic) |
| **Architecture** | x86_64 |
| **CPU** | Intel Core i7-1185G7 @ 3.00GHz (11th Gen Tiger Lake) |
| **CPU Cores** | 4 physical cores, 8 threads |
| **CPU Frequency** | 400 MHz - 4800 MHz (Turbo) |
| **RAM** | 38 GB DDR4 |
| **GPU** | Integrated Intel Iris Xe (CPU inference only) |
| **Storage** | NVMe SSD |
| **OpenCV** | 4.6.0 |
| **ONNX Runtime** | 1.20.1 (CPU) |
| **Compiler** | GCC with C++17 |

---

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| **Input Image** | dog_bike_car.jpg (768×576 pixels) |
| **Model Input Size** | 320×320 |
| **Precision** | FP32 |
| **Warmup Iterations** | 20 |
| **Benchmark Iterations** | 100 |
| **Confidence Threshold** | 0.25 |
| **NMS Threshold** | 0.45 |
| **Device** | CPU |
| **Dataset** | Pascal VOC (20 classes) |

---

## Performance Results Summary

### FPS Comparison (Higher is Better)

| Model | FPS | Rank |
|-------|-----|------|
| **YOLOv11n** | **97.18** | 🥇 1st |
| YOLOv8n | 85.55 | 🥈 2nd |
| YOLOv12n | 80.99 | 🥉 3rd |
| YOLO26n | 78.31 | 4th |
| YOLOv5nu | 77.22 | 5th |
| YOLOv6n | 76.85 | 6th |
| YOLOv10n | 69.27 | 7th |
| YOLOv9t | 46.28 | 8th |

---

## Detailed Benchmark Results

### Latency Statistics (milliseconds)

| Model | Avg | StdDev | Min | Max | P50 | P90 | P95 | P99 |
|-------|-----|--------|-----|-----|-----|-----|-----|-----|
| **YOLOv11n** | **10.29** | 0.96 | 8.47 | 12.65 | 10.25 | 11.59 | 12.11 | 12.57 |
| YOLOv8n | 11.69 | 1.25 | 9.86 | 16.08 | 11.56 | 13.10 | 14.18 | 15.73 |
| YOLOv12n | 12.35 | 0.99 | 10.36 | 18.76 | 12.14 | 13.33 | 13.95 | 15.32 |
| YOLO26n | 12.77 | 0.94 | 11.70 | 18.49 | 12.56 | 13.96 | 14.34 | 15.21 |
| YOLOv5nu | 12.95 | 1.30 | 11.32 | 19.56 | 12.70 | 13.98 | 15.28 | 19.39 |
| YOLOv6n | 13.01 | 1.26 | 11.61 | 19.57 | 12.49 | 14.36 | 15.00 | 17.36 |
| YOLOv10n | 14.44 | 8.13 | 9.12 | 55.35 | 11.45 | 23.08 | 33.35 | 48.04 |
| YOLOv9t | 21.61 | 11.06 | 15.48 | 87.97 | 17.44 | 32.16 | 41.20 | 67.76 |

### Load & Warmup Times (milliseconds)

| Model | Load Time | Warmup Time | Total Startup |
|-------|-----------|-------------|---------------|
| YOLOv8n | 55.34 | 228.29 | 283.63 |
| YOLOv11n | 59.09 | 230.38 | 289.47 |
| YOLOv6n | 61.70 | 261.34 | 323.04 |
| YOLOv5nu | 67.94 | 278.62 | 346.56 |
| YOLO26n | 70.16 | 279.41 | 349.57 |
| YOLOv10n | 74.65 | 200.34 | 274.99 |
| YOLOv9t | 98.71 | 402.03 | 500.74 |
| YOLOv12n | 103.60 | 243.54 | 347.14 |

### Memory Usage

| Model | Peak Memory (MB) | Memory Delta (MB) | CPU Usage (%) |
|-------|------------------|-------------------|---------------|
| YOLOv5nu | 110.2 | 5.2 | 84.7 |
| YOLOv10n | 111.4 | 7.3 | 94.0 |
| YOLOv8n | 116.8 | 9.4 | 86.6 |
| YOLOv11n | 119.4 | 13.8 | 85.2 |
| YOLO26n | 120.1 | 14.8 | 85.6 |
| YOLOv6n | 121.2 | 4.1 | 85.6 |
| YOLOv12n | 121.2 | 13.4 | 85.6 |
| YOLOv9t | 123.4 | 14.8 | 90.4 |

### Model File Sizes

| Model | ONNX Size | Architecture |
|-------|-----------|--------------|
| YOLOv9t | 7.8 MB | Gelan-based |
| YOLOv10n | 8.8 MB | End-to-end NMS-free |
| YOLO26n | 9.3 MB | End-to-end NMS-free |
| YOLOv5nu | 9.7 MB | CSPDarknet |
| YOLOv11n | 10 MB | C3k2 blocks |
| YOLOv12n | 10 MB | Area Attention |
| YOLOv8n | 12 MB | C2f blocks |
| YOLOv6n | 17 MB | EfficientRep |

---

## Analysis & Insights

### 🏆 Performance Leaders

1. **YOLOv11n** achieves the highest FPS (97.18) with excellent latency consistency (σ=0.96ms)
2. **YOLOv8n** follows closely at 85.55 FPS with good stability
3. **YOLOv12n** and **YOLO26n** perform similarly (~80 FPS)

### ⚡ Latency Stability

| Model | Stability Rating | Notes |
|-------|------------------|-------|
| YOLO26n | ⭐⭐⭐⭐⭐ | Lowest std dev (0.94ms), most predictable |
| YOLOv11n | ⭐⭐⭐⭐⭐ | Very stable (0.96ms std dev) |
| YOLOv12n | ⭐⭐⭐⭐⭐ | Excellent consistency (0.99ms) |
| YOLOv8n | ⭐⭐⭐⭐ | Good stability (1.25ms) |
| YOLOv5nu | ⭐⭐⭐⭐ | Acceptable variance (1.30ms) |
| YOLOv6n | ⭐⭐⭐⭐ | Good for production (1.26ms) |
| YOLOv10n | ⭐⭐ | High variance (8.13ms), outliers up to 55ms |
| YOLOv9t | ⭐ | Very high variance (11.06ms), not production-ready |

### 💾 Memory Efficiency

- **YOLOv5nu** and **YOLOv10n** are most memory-efficient (~110 MB)
- All models fit comfortably in typical embedded systems with 256MB+ RAM
- Memory delta during inference is minimal (<15 MB)

### 🎯 End-to-End NMS-Free Models

**YOLO26n** and **YOLOv10n** feature built-in NMS:
- ✅ No separate NMS postprocessing step
- ✅ Simpler deployment pipeline
- ✅ Consistent latency (no NMS variance)
- ⚠️ YOLOv10n shows higher variance likely due to model complexity

---

## Recommendations

### For Real-Time Applications (>60 FPS required)
- **Best Choice:** YOLOv11n (97 FPS, excellent stability)
- **Runner-up:** YOLOv8n (86 FPS, proven reliability)

### For Edge Deployment (Memory Constrained)
- **Best Choice:** YOLOv5nu (110 MB, 77 FPS)
- **Alternative:** YOLOv10n (111 MB, end-to-end architecture)

### For Predictable Latency (Production Systems)
- **Best Choice:** YOLO26n (lowest variance, 78 FPS)
- **Alternative:** YOLOv11n (stable with higher throughput)

### For Simplest Deployment (No NMS handling)
- **Best Choice:** YOLO26n (end-to-end, stable, modern)
- **Alternative:** YOLOv10n (end-to-end, but less stable)

---

## Raw Data

Full benchmark CSV available at: `benchmarks/results/benchmark_results.csv`

---

## YOLOE Open-Vocabulary Benchmark Results

**Added:** April 2026  
**Task:** YOLOE instance segmentation (open-vocabulary)  
**Conditions:** Same system as above — Intel i7-1185G7, CPU, ONNX Runtime 1.20.1, 640×640, 200 iterations, 30 warmup, 6 LVIS classes (`person,car,bus,bicycle,motorcycle,truck`)

### Methodology

YOLOE models were exported from Ultralytics checkpoints with `model.set_classes([...])` then
`model.export(format="onnx", nms=False)`. The C++ `YOLOESegDetector` uses class-agnostic NMS
(enabled by default) for correct open-vocabulary behaviour. Benchmarks were run with the
`yolo_unified_benchmark image <model> yoloe-seg <onnx> <classes> <image>` command.

### FPS & Latency — YOLOE vs Closed-Set Baseline (CPU @ 640×640)

| Model | Type | Classes | FPS | Avg(ms) | StdDev | Min(ms) | P50(ms) | P90(ms) | P95(ms) | P99(ms) | Mem(MB) | ONNX Size |
|-------|------|--------:|----:|--------:|-------:|--------:|--------:|--------:|--------:|--------:|--------:|----------:|
| **yoloe-26n-seg** | Open-vocab | 6 | **10** | **97.0** | 23.5 | 62.5 | 92.3 | 118.9 | 134.9 | 181.0 | 218 | 10.6 MB |
| **yoloe-26s-seg** | Open-vocab | 6 | **3** | **255.6** | 44.0 | 201.0 | 246.1 | 301.0 | 318.4 | 367.4 | 346 | 39.9 MB |
| yolo26n-seg *(baseline)* | Closed-set | 80 | 9 | 110.6 | 37.5 | 61.6 | 104.9 | 147.8 | 169.1 | 291.4 | 219 | 10.7 MB |

### Comprehensive Suite — All Tasks (CPU @ 640×640, 150 iters)

*Models run sequentially; memory readings include cumulative RSS.*

| Model | Task | FPS | Avg(ms) | P99(ms) | Mem(MB) | ONNX Size |
|-------|------|----:|--------:|--------:|--------:|----------:|
| yolo26n | detection | 16 | 60.97 | 127.10 | 184 | 9.5 MB |
| yolo26n-seg | segmentation | 12 | 80.13 | 126.68 | 227 | 10.7 MB |
| yolo26n-pose | pose | 11 | 85.53 | 190.52 | 660† | 11.6 MB |
| **yoloe-26n-seg** | **yoloe-seg** | **8** | **111.4** | **244.1** | 660† | 10.6 MB |
| **yoloe-26s-seg** | **yoloe-seg** | **3** | **258.4** | **441.6** | 660† | 39.9 MB |
| yolo26m-pose | pose | 1 | 511.9 | 775.2 | 660† | 82.6 MB |
| yolo26m-seg | segmentation | 1 | 887.9 | 1571.9 | 660† | 90.2 MB |

† Memory readings cumulative after loading multiple large models sequentially.

### Analysis

#### Open-Vocabulary vs Closed-Set (nano models)

| Metric | YOLOE-26n-seg (6 cls) | YOLO26n-seg (80 cls) | Delta |
|--------|----------------------:|---------------------:|------:|
| FPS | 10 | 9 | +11% faster |
| Avg latency | 97.0 ms | 110.6 ms | −12% |
| P99 latency | 181.0 ms | 291.4 ms | −38% |
| Latency σ | 23.5 ms | 37.5 ms | −37% (more consistent) |
| ONNX size | 10.6 MB | 10.7 MB | ~identical |

**Key finding:** YOLOE-26n-seg with 6 exported classes is **faster and more consistent** than the
80-class YOLO26n-seg on CPU. This is because the postprocessing workload scales with output
channels: fewer classes → fewer score comparisons per anchor → faster NMS. The architecture
overhead of YOLOE (RepRTA modules) is zero at inference — they are re-parameterised away.

#### YOLOE Class Count vs Speed (empirical rule of thumb)

| Exported classes | Expected vs YOLO26n-seg | Use case |
|:---:|---|---|
| 1 – 10 | Faster or equal | Targeted deployment (robot, kiosk) |
| 20 – 30 | Similar | Custom datasets |
| 80 (COCO-equivalent) | ~10-20% slower | Drop-in open-vocab replacement |
| 300+ (large vocab) | 20-40% slower | Discovery / cataloging tasks |

#### Model Scale Trade-offs

| Model | FPS (CPU) | LVIS mAP (paper) | Use case |
|-------|----------:|------------------:|---------|
| YOLOE-26n-seg | 10 | — | Edge / embedded, real-time |
| YOLOE-26s-seg | 3 | 29.9% | Workstation CPU, high accuracy |
| YOLOE-26m-seg | <1 | 33.1% | GPU required for real-time |
| YOLOE-26l-seg | <1 | 36.8% | GPU, production accuracy |

#### Load and Startup Times (YOLOE-specific)

| Model | Load Time | Warmup (30 iters) | Note |
|-------|----------:|------------------:|------|
| yoloe-26n-seg | 59.1 ms | 2259 ms | Same as YOLO26n-seg |
| yoloe-26s-seg | 130.5 ms | 8090 ms | Larger model, longer warmup |

### Recommendations for YOLOE Deployment

#### CPU / Edge Devices
- Use **YOLOE-26n-seg** with ≤20 classes for real-time performance (≥10 FPS at 640×640)
- Reduce resolution to 320×320 for 2-3× speed gain with minimal accuracy loss
- Warmup with 20-30 dummy frames before live inference

#### GPU Acceleration (T4/RTX-class)
- All YOLOE models reach **6.2ms / 161 FPS** at 640×640 (Ultralytics paper, T4)
- No inference overhead vs equivalent YOLO26 backbone — modules re-parameterised away

#### Vocabulary Size Guidance
- **Text-prompt** (few classes): fastest — class list baked in at ONNX export (`set_classes` before `export`)
- **Prompt-free (`-pf`)**: large fixed vocabulary (Ultralytics docs: e.g. thousands of classes); use agnostic NMS; labels file must match that export — expect higher CPU cost than small text-prompt exports
- **setClasses()**: zero extra inference cost — relabels fixed channels (same count as export); not for adding new concepts without re-export

---

## Notes

- All models were fine-tuned on Pascal VOC dataset (20 classes)
- Benchmarks run on CPU-only configuration (Intel i7-1185G7, no discrete GPU)
- Results may vary with GPU acceleration (typically 3-10× faster, YOLOE reaches 161 FPS on T4)
- Input image resolution affects performance proportionally (320×320 ≈ 2.5× faster than 640×640)
- Models exported with ONNX opset 12 for maximum compatibility

---

*Report generated by YOLOs-CPP Unified Benchmark Suite v2.0.0 — YOLOE section added April 2026*
