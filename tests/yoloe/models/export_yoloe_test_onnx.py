#!/usr/bin/env python3
"""
Export yoloe-26n-seg.onnx for tests/yoloe parity.

Class list must match tests/yoloe/inference_config.json (used by Python + C++).
"""

import json
import sys
from pathlib import Path

from ultralytics import YOLOE

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "inference_config.json"


def main() -> int:
    if not CONFIG.is_file():
        print(f"Missing {CONFIG}", file=sys.stderr)
        return 1

    with open(CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    classes = cfg.get("classes")
    if not classes:
        print("inference_config.json must contain a non-empty 'classes' array", file=sys.stderr)
        return 1

    out_dir = Path(__file__).resolve().parent
    pt_name = "yoloe-26n-seg.pt"
    onnx_name = "yoloe-26n-seg.onnx"
    dest = out_dir / onnx_name

    print(f"Loading {pt_name}, classes ({len(classes)}): {classes}")
    model = YOLOE(pt_name, task="segment", verbose=True)
    model.set_classes(classes)

    export_path = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        half=False,
        dynamic=False,
        nms=False,
    )
    Path(export_path).rename(dest)
    print(f"Saved {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
