# Model Guide

Supported models, ONNX export, and optimization for YOLOs-CPP.

## Supported Models

### YOLOE (Open-Vocabulary Detection & Segmentation)

YOLOE ([arXiv 2503.07465](https://arxiv.org/abs/2503.07465)) is a promptable YOLO variant for
detection and instance segmentation. In **Python**, you can choose classes with `set_classes()`
before inference, use **prompt-free** checkpoints with a large fixed vocabulary, or (in some setups)
**visual** prompts.

#### YOLOE and ONNX runtime (Ultralytics)

[Ultralytics](https://docs.ultralytics.com/) documents ONNX export for deployment in other
applications and devices; **ONNX Runtime** (including C and C++) is a normal path—this project is
not Python-only at inference time.

The important limitation for **exported ONNX**:

- **Text-prompt YOLOE:** Ultralytics shows `model.set_classes([...])` **before**
  `model.export(format="onnx")`. That **bakes** the chosen class list into the graph weights /
  head for that export. A plain `.onnx` file does **not** accept arbitrary new text prompts at
  runtime the way interactive Python `set_classes()` does. In C++, the strings you pass are
  **labels** for the fixed output channels (same count and order as export).
- **Prompt-free YOLOE-PF (`-pf` checkpoints):** These ship with a **large built-in vocabulary**
  and do not need prompts at inference. Ultralytics documents this as a **fixed** predefined list
  (e.g. **4,585** classes in current docs—confirm against your Ultralytics version). That is broad
  “open-set” behavior inside that list, not unlimited free-form phrases at ORT runtime.
- **Visual prompting:** Ultralytics states that **visual prompting at inference currently requires
  the Python API**. After you fix classes from a visual prompt, you can **export** to ONNX and
  deploy that exported model in C++ like any other text-prompt export.

**Summary:** ONNX deployment is not Python-only; **authoring** dynamic prompts (beyond relabeling
fixed channels) is mostly a Python/export-time concern unless your ONNX explicitly includes a
runtime text path matching Ultralytics’ export.

#### Prompt Modes

| Mode | How it works | When to use |
|------|-------------|-------------|
| **Text prompt** | Class list set in Python **before** `export(format="onnx")` — baked into that ONNX | Fixed custom list, fastest ORT inference |
| **Visual prompt** | Inference-time visual prompt: **Python API**; then export retains fixed classes | Prototype in Python, ship an exported ONNX |
| **Prompt-free** | Large **fixed** built-in vocabulary in `-pf` checkpoints (Ultralytics: e.g. thousands of classes) | Discovery within that vocabulary; no per-query text in ORT |

#### Available Models

| Model | Backbone | Params | LVIS mAP† | Speed (T4)† | CPU 640×640‡ | ONNX Size |
|-------|----------|-------:|----------:|------------:|-------------:|----------:|
| YOLOE-26N-seg | YOLO26 | 3.2M | — | ~8ms (125 FPS) | **97ms (10 FPS)** | 10.6 MB |
| YOLOE-26S-seg | YOLO26 | 11M | 29.9% | 6.2ms (161 FPS) | **256ms (3 FPS)** | 39.9 MB |
| YOLOE-26M-seg | YOLO26 | 21M | 33.1% | 6.2ms (161 FPS) | ~600ms | ~80 MB |
| YOLOE-26L-seg | YOLO26 | 32.3M | 36.8% | 6.2ms (161 FPS) | ~900ms | ~130 MB |
| YOLOE-26X-seg | YOLO26 | 53M | 38.0% | ~8ms | — | — |
| YOLOE-11S-seg | YOLO11 | 10M | — | 6.2ms | — | — |
| YOLOE-11M-seg | YOLO11 | 20M | — | 6.2ms | — | — |
| YOLOE-11L-seg | YOLO11 | 26.2M | 35.2% | 6.2ms (161 FPS) | — | — |

† Paper numbers (Ultralytics, T4 GPU, full LVIS evaluation).  
‡ **Measured on Intel i7-1185G7 CPU, ONNX Runtime 1.20.1, 6-class text-prompt export, 200 iterations.**  
  See [`benchmarks/results/BENCHMARK_REPORT.md`](https://github.com/Geekgineer/YOLOs-CPP/blob/main/benchmarks/results/BENCHMARK_REPORT.md) for full statistics.

#### CPU Benchmark — YOLOE vs Closed-Set Baseline

*Measured: i7-1185G7, ORT 1.20.1 CPU, 640×640, 200 iters, 30 warmup, `dog.jpg` 768×576*

| Model | Type | Classes | FPS | Avg(ms) | P50(ms) | P90(ms) | P99(ms) | σ(ms) | Mem |
|-------|------|--------:|----:|--------:|--------:|--------:|--------:|------:|----:|
| **yoloe-26n-seg** | Open-vocab | 6 | **10** | **97.0** | 92.3 | 118.9 | 181.0 | 23.5 | 218 MB |
| **yoloe-26s-seg** | Open-vocab | 6 | **3** | **255.6** | 246.1 | 301.0 | 367.4 | 44.0 | 346 MB |
| yolo26n-seg *(baseline)* | Closed-set | 80 | 9 | 110.6 | 104.9 | 147.8 | 291.4 | 37.5 | 219 MB |

> **Finding:** YOLOE-26n-seg with 6 exported classes is **faster and more consistent** than the 80-class
> YOLO26n-seg on CPU (+11% FPS, −12% avg latency, −37% P99). The YOLOE architecture overhead is zero
> at inference — RepRTA/SAVPE modules are fully re-parameterised into the standard YOLO head on export.
> Speed advantage grows as the target vocabulary shrinks: fewer classes = fewer postprocessing channels.

All `-pf` prompt-free variants are also available (e.g. `yoloe-26s-seg-pf.pt`).

#### Exporting to ONNX

```bash
pip install -U ultralytics
python scripts/export_yoloe_onnx.py
```

To export a **custom** class list (same count and order as in C++), use:

```bash
python scripts/export_yoloe_classes.py --out models/my-yoloe.onnx dog person car bus bicycle motorcycle truck
```

Or manually:

```python
from ultralytics import YOLOE

# Text-prompt export (classes baked in)
model = YOLOE("yoloe-26s-seg.pt")
model.set_classes(["person", "car", "bus"])
model.export(format="onnx", nms=False)   # nms=False — C++ handles NMS

# Prompt-free export (large fixed vocab — match labels file line count to this export)
model_pf = YOLOE("yoloe-26s-seg-pf.pt")
model_pf.export(format="onnx", nms=False)
```

> **Note:** Always export with `nms=False`. YOLOs-CPP uses class-agnostic NMS
> (enabled by default in `YOLOESegDetector`) which prevents duplicate detections
> across YOLOE's large vocabulary.

#### ONNX filenames (consistent with `scripts/export_yoloe_onnx.py`)

| Artifact | When |
|----------|------|
| `yoloe-26s-seg-text.onnx`, `yoloe-11s-seg-text.onnx`, `yoloe-26l-seg-text.onnx` | Text-prompt exports from the script (renamed after `model.export()`) |
| `yoloe-26s-seg-pf.onnx` | Prompt-free export |
| `yoloe-26n-seg.onnx` (or any name) | Manual export keeps Ultralytics’ default stem unless you rename |

Use the **same class list in C++** as in `model.set_classes([...])` for that ONNX.

#### C++ Usage

```cpp
#include "yolos/tasks/yoloe.hpp"
using namespace yolos::yoloe;

// ── Text-prompt mode (classes baked in at export time) ──────────────────────
YOLOESegDetector det(
    "models/yoloe-26s-seg-text.onnx",
    {"person", "car", "bus", "bicycle", "motorcycle", "truck"}
    /* useGPU = true by default */
);
auto segs = det.segment(frame, 0.35f, 0.45f);
det.drawSegmentations(frame, segs, 0.45f);

// Same API with a nano export (e.g. manual: yoloe-26n-seg.onnx)
// YOLOESegDetector det_n("models/yoloe-26n-seg.onnx", { ... }, useGPU);

// ── Prompt-free mode (large fixed vocabulary — labels file must match export) ─
YOLOESegDetector det_pf(
    "models/yoloe-26s-seg-pf.onnx",
    "models/yoloe_pf.names"   // one class name per line; line count = ONNX class count
);
auto segs_pf = det_pf.segment(frame);

// ── Relabel fixed channels (same count as export — not new concepts) ─────────
det.setClasses({"person", "car", "bus", "bicycle", "motorcycle", "truck"});

// ── Detection only (no masks): export a YOLOE *detection* checkpoint to ONNX, then:
// YOLOEDetector det_only("models/my_yoloe_det.onnx", {"person", "car"});
// auto dets = det_only.detect(frame);

// ── Factory helper ──────────────────────────────────────────────────────────
auto p = createYOLOESegDetector(
    "models/yoloe-26l-seg-text.onnx",
    {"person", "bus"},
    /*useGPU=*/true,
    /*agnosticNms=*/true
);
```

#### Running the Demo

Use **`image_yoloe_seg`** for still images (image → image) and **`video_yoloe_seg`** for videos (MP4 → MP4).

```bash
# Text-prompt mode — single image
./image_yoloe_seg data/dog.jpg out.jpg \
    models/yoloe-26n-seg.onnx \
    "person,car,bus,bicycle,motorcycle,truck" 0

# Text-prompt mode — video
./video_yoloe_seg data/Transmission.mp4 out.mp4 \
    models/yoloe-26n-seg.onnx \
    "person,car,bus,bicycle,motorcycle,truck" 1

# Prompt-free mode — video (labels file must match PF ONNX class count)
./video_yoloe_seg data/Transmission.mp4 out.mp4 \
    models/yoloe-26s-seg-pf.onnx \
    models/yoloe_pf.names 1
```

#### Running Benchmarks

```bash
# Quick single-model benchmark
./benchmarks/build/yolo_unified_benchmark \
    image yoloe-26n-seg yoloe-seg \
    models/yoloe-26n-seg.onnx \
    "person,car,bus,bicycle,motorcycle,truck" \
    data/dog.jpg --iterations=200 --warmup=30 --json

# Comprehensive suite — all models including YOLOE, sorted by FPS
./benchmarks/build/yolo_unified_benchmark \
    comprehensive data/dog.jpg models --iterations=150 --warmup=20

# GPU benchmark
./benchmarks/build/yolo_unified_benchmark \
    comprehensive data/dog.jpg models --gpu --iterations=500 --warmup=50
```

See [`benchmarks/results/BENCHMARK_REPORT.md`](https://github.com/Geekgineer/YOLOs-CPP/blob/main/benchmarks/results/BENCHMARK_REPORT.md) for the
full report including latency percentiles, memory usage, and comparative analysis.

#### Label Files

| File | Classes | Use Case |
|------|--------:|----------|
| `coco.names` | 80 | Closed-set detection |
| `yoloe_pf.names` (example name) | **Must match your PF ONNX** | Prompt-free YOLOE — one line per output class |

For prompt-free models, the **line count must equal** the number of classes in that ONNX (see
Ultralytics docs / ONNX metadata for your checkpoint). A **LVIS** category list is one way to
obtain names when your export is LVIS-aligned (~1203 categories); Ultralytics YOLOE-PF docs describe
a **larger** fixed vocabulary (e.g. 4,585 classes)—use the list that matches **your** exported
`yoloe-*-pf.onnx`, not a generic file from another model.

Example (LVIS categories — only if compatible with your export):
```python
from ultralytics.data.dataset import LVISDataset
names = LVISDataset.get_lvis_categories()
with open("models/lvis.names", "w") as f:
    f.write("\n".join(names))
```

---

### Detection

| Model | Params | mAP | Speed (GPU) |
|-------|-------:|----:|------------:|
| YOLOv5n | 1.9M | 28.0 | 6.3ms |
| YOLOv8n | 3.2M | 37.3 | 6.2ms |
| YOLOv11n | 2.6M | 39.5 | 6.5ms |
| YOLO26n | 2.5M | 40.2 | 7.1ms |

### Segmentation

| Model | Params | mAP | Speed (GPU) |
|-------|-------:|----:|------------:|
| YOLOv8n-seg | 3.4M | 36.7 | 8.4ms |
| YOLOv11n-seg | 2.9M | 38.9 | 8.1ms |
| YOLO26n-seg | 2.8M | 39.4 | 8.8ms |

### Pose Estimation

| Model | Params | mAP | Speed (GPU) |
|-------|-------:|----:|------------:|
| YOLOv8n-pose | 3.3M | 50.4 | 5.9ms |
| YOLOv11n-pose | 2.9M | 52.1 | 5.7ms |
| YOLO26n-pose | 2.8M | 53.0 | 6.2ms |

### OBB (Oriented Bounding Boxes)

| Model | Params | Dataset |
|-------|-------:|---------|
| YOLOv8n-obb | 3.1M | DOTA |
| YOLOv11n-obb | 2.7M | DOTA |
| YOLO26n-obb | 2.6M | DOTA |

### Classification

| Model | Params | Top-1 Acc |
|-------|-------:|----------:|
| YOLOv8n-cls | 2.7M | 66.6% |
| YOLOv11n-cls | 1.6M | 70.0% |
| YOLO26n-cls | 1.5M | 71.2% |

## Exporting to ONNX

### Using Ultralytics

```python
from ultralytics import YOLO

# Load model
model = YOLO("yolo11n.pt")

# Export to ONNX
model.export(
    format="onnx",
    imgsz=640,
    opset=12,        # ONNX opset version
    simplify=False,
    half=False,
    dynamic=False,
    nms=False        # NMS is done in C++
)
```

### Export Options

| Option | Value | Notes |
|--------|-------|-------|
| `opset` | 12-17 | Use 12 for max compatibility |
| `imgsz` | 640 | Match your inference resolution |
| `half` | False | FP32 for accuracy (FP16 optional) |
| `dynamic` | False | Static shapes for best performance |
| `nms` | False | C++ handles NMS |

### Batch Export Script

```bash
python models/export_onnx.py
```

## Label Files

| File | Classes | Use Case |
|------|--------:|----------|
| `coco.names` | 80 | General detection |
| `Dota.names` | 15 | Aerial/satellite OBB |
| `imagenet_classes.txt` | 1000 | Classification |

## Model Paths

```cpp
// Detection
"models/yolo11n.onnx"

// Segmentation
"models/yolo11n-seg.onnx"

// Pose
"models/yolo11n-pose.onnx"

// OBB
"models/yolo11n-obb.onnx"

// Classification
"models/yolo11n-cls.onnx"
```

## Quantization

Quantized models offer:
- **2-4x smaller** file size
- **1.5-2x faster** CPU inference
- **Slight accuracy loss** (~1-2% mAP)

### Quantize with ONNX

```python
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    "model.onnx",
    "model_int8.onnx"
)
```

See `quantized_models/yolos_quantization.py` for examples.

## Custom Models

To use custom-trained models:

1. Train with Ultralytics
2. Export to ONNX with compatible settings
3. Create matching label file
4. Load in YOLOs-CPP

```cpp
yolos::det::YOLODetector detector(
    "custom_model.onnx",
    "custom_labels.txt",
    true
);
```