"""
Export YOLOE models to ONNX for use with YOLOs-CPP.

YOLOE supports multiple prompting modes; for ONNX + C++ deployment, note:

  Mode 1 – Text Prompt:
    Call set_classes([...]) before export. The class list is baked into that ONNX;
    C++ passes the same names as labels for fixed output channels (not new prompts at ORT runtime).

  Mode 2 – Visual Prompt:
    Ultralytics: visual prompting at inference uses the Python API; after fixing
    classes you can export and deploy the resulting ONNX in C++ like text-prompt exports.

  Mode 3 – Prompt-Free (*-pf.pt):
    Export without set_classes(). The checkpoint has a large fixed built-in vocabulary
    (Ultralytics docs: e.g. thousands of classes — confirm for your version).
    In C++ pass a .names file with one line per class; line count must match that ONNX.

Usage:
    pip install -U ultralytics
    python scripts/export_yoloe_onnx.py
"""

from pathlib import Path
from ultralytics import YOLOE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

IMGSZ      = 640   # Input resolution (must match C++ inference)
OPSET      = 12    # ONNX opset (12 = widest compatibility; 17 = latest ops)
HALF       = False # FP32 for accuracy; set True for FP16 on supported GPUs
DYNAMIC    = False # Static shapes = best ORT performance

# ---------------------------------------------------------------------------
# Mode 1: Text Prompt – YOLOE-26s (YOLO26 backbone, end-to-end NMS-free)
# ---------------------------------------------------------------------------
# The exported ONNX will have exactly len(CLASSES) output channels.
# In C++ construct: YOLOESegDetector(model_path, CLASSES, use_gpu)

CLASSES_TEXT = ["person", "car", "bus", "bicycle", "motorcycle", "truck"]

print("=== Mode 1: Text Prompt (YOLOE-26s-seg) ===")
model = YOLOE("yoloe-26s-seg.pt")
model.set_classes(CLASSES_TEXT)

export_path = model.export(
    format="onnx",
    imgsz=IMGSZ,
    opset=OPSET,
    simplify=True,
    half=HALF,
    dynamic=DYNAMIC,
    nms=False,   # NMS is handled in C++ (YOLOs-CPP uses agnostic NMS for YOLOE)
)
dest = OUTPUT_DIR / "yoloe-26s-seg-text.onnx"
Path(export_path).rename(dest)
print(f"  Saved → {dest}")
print(f"  Classes ({len(CLASSES_TEXT)}): {CLASSES_TEXT}\n")

# ---------------------------------------------------------------------------
# Mode 1 (variant): YOLOE-11s (YOLO11 backbone, standard V11 output format)
# ---------------------------------------------------------------------------

print("=== Mode 1: Text Prompt (YOLOE-11s-seg) ===")
model11 = YOLOE("yoloe-11s-seg.pt")
model11.set_classes(CLASSES_TEXT)

export_path_11 = model11.export(
    format="onnx",
    imgsz=IMGSZ,
    opset=OPSET,
    simplify=True,
    half=HALF,
    dynamic=DYNAMIC,
    nms=False,
)
dest11 = OUTPUT_DIR / "yoloe-11s-seg-text.onnx"
Path(export_path_11).rename(dest11)
print(f"  Saved → {dest11}")
print(f"  Classes ({len(CLASSES_TEXT)}): {CLASSES_TEXT}\n")

# ---------------------------------------------------------------------------
# Mode 1 (large): YOLOE-26l for higher accuracy deployments
# ---------------------------------------------------------------------------

CLASSES_LARGE = ["person", "bus"]

print("=== Mode 1: Text Prompt (YOLOE-26l-seg) ===")
model_l = YOLOE("yoloe-26l-seg.pt")
model_l.set_classes(CLASSES_LARGE)

export_path_l = model_l.export(
    format="onnx",
    imgsz=IMGSZ,
    opset=OPSET,
    simplify=True,
    half=HALF,
    dynamic=DYNAMIC,
    nms=False,
)
dest_l = OUTPUT_DIR / "yoloe-26l-seg-text.onnx"
Path(export_path_l).rename(dest_l)
print(f"  Saved → {dest_l}")
print(f"  Classes ({len(CLASSES_LARGE)}): {CLASSES_LARGE}\n")

# ---------------------------------------------------------------------------
# Mode 3: Prompt-Free (large fixed vocabulary)
# ---------------------------------------------------------------------------
# Use the *-pf.pt checkpoint – no set_classes() needed.
# In C++ pass a labels file whose line count matches this ONNX (see Ultralytics docs).
# Agnostic NMS is critical to avoid duplicate detections across many classes.

print("=== Mode 3: Prompt-Free (YOLOE-26s-seg-pf) ===")
model_pf = YOLOE("yoloe-26s-seg-pf.pt")

export_path_pf = model_pf.export(
    format="onnx",
    imgsz=IMGSZ,
    opset=OPSET,
    simplify=True,
    half=HALF,
    dynamic=DYNAMIC,
    nms=False,
)
dest_pf = OUTPUT_DIR / "yoloe-26s-seg-pf.onnx"
Path(export_path_pf).rename(dest_pf)
print(f"  Saved → {dest_pf}")
print("  Note: use a .names file matching this PF export's class count (see docs/guides/models.md)\n")

# ---------------------------------------------------------------------------
# Notes for C++ usage (YOLOs-CPP)
# ---------------------------------------------------------------------------
print("=" * 60)
print("C++ usage examples:")
print()
print("// Text-prompt mode (YOLOE-26s, classes baked in at export):")
print('YOLOESegDetector det("models/yoloe-26s-seg-text.onnx",')
print('                     {"person","car","bus","bicycle","motorcycle","truck"});')
print('auto segs = det.segment(frame);')
print()
print("// Prompt-free mode (YOLOE-26s-pf — labels file matches export):")
print('YOLOESegDetector det("models/yoloe-26s-seg-pf.onnx",')
print('                     "models/yoloe_pf.names");')
print('auto segs = det.segment(frame);')
print()
print("// Relabel same channel count as text export (not new concepts):")
print('det.setClasses({"person","car","bus","bicycle","motorcycle","truck"});')
print("=" * 60)
