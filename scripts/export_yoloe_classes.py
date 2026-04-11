#!/usr/bin/env python3
"""
Export YOLOE segmentation ONNX with a custom class list (must match C++ inference).

Example (dogs + default road classes):
  python scripts/export_yoloe_classes.py \\
    --out models/yoloe-26n-seg-dog-vocab.onnx \\
    dog person car bus bicycle motorcycle truck
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLOE


def main() -> int:
    p = argparse.ArgumentParser(description="Export YOLOE-26n-seg ONNX with set_classes([...]).")
    p.add_argument("--pt", default="yoloe-26n-seg.pt", help="Source checkpoint (.pt)")
    p.add_argument("--out", required=True, help="Output .onnx path")
    p.add_argument(
        "classes",
        nargs="+",
        help="Class names in order (same list you pass to image_yoloe_seg / video_yoloe_seg)",
    )
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.pt}, classes ({len(args.classes)}): {args.classes}")
    model = YOLOE(args.pt, task="segment", verbose=True)
    model.set_classes(list(args.classes))

    export_path = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        half=False,
        dynamic=False,
        nms=False,
    )
    Path(export_path).rename(out)
    print(f"Saved {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
