#!/usr/bin/env python3
"""Generate tiny synthetic ONNX models for the API test suite.

These models are *not* YOLO networks. They only reproduce the tensor shapes and
the ONNX-level contract that YOLOs-CPP depends on (input NCHW layout, output
layouts per task, dynamic vs fixed batch dimension), so the batch-inference and
in-memory-loading code paths can be tested without downloading real weights or
installing PyTorch/Ultralytics.

Each model computes one scalar per image (the mean of that image's pixels) and
multiplies a fixed constant tensor by it. That makes every output element depend
on its own image and nothing else, so any batch-slicing mistake shows up as
wrong numbers rather than as a plausible-looking result.

Requires: onnx, numpy
Usage:    python3 make_synthetic_models.py [output_dir]
"""

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 13


def _scaled_constant(name, const_array, reshape_dims, input_name="images"):
    """Nodes computing `mean(input) * const_array`, broadcast over the batch.

    reshape_dims is the rank the per-image scalar is reshaped to (excluding the
    batch dim), e.g. 2 -> [B, 1, 1] so it broadcasts against a [1, F, D] constant.
    """
    shape = [-1] + [1] * reshape_dims
    return (
        [
            helper.make_node("ReduceMean", [input_name], ["_mean"], axes=[1, 2, 3], keepdims=0),
            helper.make_node("Reshape", ["_mean", "_reshape_" + name], ["_scale_" + name]),
            helper.make_node("Mul", ["_scale_" + name, "_const_" + name], [name]),
        ],
        [
            numpy_helper.from_array(np.array(shape, dtype=np.int64), "_reshape_" + name),
            numpy_helper.from_array(const_array.astype(np.float32), "_const_" + name),
        ],
    )


def _save(path, nodes, initializers, inputs, outputs, metadata=None):
    graph = helper.make_graph(nodes, os.path.basename(path), inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8  # ONNX Runtime 1.20 supports IR <= 9; 8 is widely compatible
    if metadata:
        for key, value in metadata.items():
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = value
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"wrote {path}")


def _image_input(batch, size, channels=3):
    """Input tensor info; batch=None marks the batch dimension as dynamic."""
    dim0 = "batch" if batch is None else batch
    return helper.make_tensor_value_info("images", TensorProto.FLOAT, [dim0, channels, size, size])


def _output(name, batch, dims):
    dim0 = "batch" if batch is None else batch
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [dim0] + list(dims))


def detection_constant(num_features, num_detections):
    """[1, F, D] constant laid out as YOLOv8/v11 detection: cx, cy, w, h, scores."""
    c = np.zeros((1, num_features, num_detections), dtype=np.float32)
    num_classes = num_features - 4
    for d in range(num_detections):
        c[0, 0, d] = 300.0 + 5.0 * d          # cx before scaling by the image mean
        c[0, 1, d] = 500.0 + 3.0 * d          # cy
        c[0, 2, d] = 220.0                    # w
        c[0, 3, d] = 180.0                    # h
        for k in range(num_classes):
            # A handful of confident detections per class, the rest near zero,
            # so postprocessing and NMS both have real work to do.
            c[0, 4 + k, d] = 1.8 if d % 17 == k else 0.05
    return c


def pose_constant(num_detections):
    """[1, 56, D] constant: cx, cy, w, h, conf, then 17 keypoint triplets."""
    c = np.zeros((1, 56, num_detections), dtype=np.float32)
    for d in range(num_detections):
        c[0, 0, d] = 300.0 + 5.0 * d
        c[0, 1, d] = 500.0 + 3.0 * d
        c[0, 2, d] = 220.0
        c[0, 3, d] = 180.0
        c[0, 4, d] = 1.8 if d % 17 == 0 else 0.05
        for k in range(17):
            c[0, 5 + k * 3 + 0, d] = 200.0 + 9.0 * k
            c[0, 5 + k * 3 + 1, d] = 400.0 + 7.0 * k
            c[0, 5 + k * 3 + 2, d] = 1.0
    return c


def obb_constant(num_labels, num_detections):
    """[1, 5 + L, D] constant: cx, cy, w, h, scores..., angle."""
    num_features = 5 + num_labels
    c = np.zeros((1, num_features, num_detections), dtype=np.float32)
    for d in range(num_detections):
        c[0, 0, d] = 300.0 + 5.0 * d
        c[0, 1, d] = 500.0 + 3.0 * d
        c[0, 2, d] = 220.0
        c[0, 3, d] = 180.0
        for k in range(num_labels):
            c[0, 4 + k, d] = 1.8 if d % 17 == k else 0.05
        c[0, 4 + num_labels, d] = 1.2  # angle in radians after scaling
    return c


def write_models(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # ---- detection, dynamic batch (true batched path) --------------------
    nodes, inits = _scaled_constant("output0", detection_constant(6, 100), reshape_dims=2)
    _save(
        os.path.join(out_dir, "det_dynamic.onnx"),
        nodes, inits,
        [_image_input(None, 640)],
        [_output("output0", None, [6, 100])],
        metadata={"names": "{0: 'alpha', 1: 'beta'}"},
    )

    # ---- detection, batch dimension fixed to 1 (fallback path) -----------
    nodes, inits = _scaled_constant("output0", detection_constant(6, 100), reshape_dims=2)
    _save(
        os.path.join(out_dir, "det_batch1.onnx"),
        nodes, inits,
        [_image_input(1, 640)],
        [_output("output0", 1, [6, 100])],
    )

    # ---- detection that *claims* a dynamic batch but cannot run one -------
    # Mirrors real Ultralytics exports: the batch dim is dynamic, but a Reshape
    # target is hard-coded for batch 1, so ONNX Runtime throws at N>1. The
    # library must notice and fall back instead of surfacing the error.
    nodes, inits = _scaled_constant("_scaled", detection_constant(6, 100), reshape_dims=2)
    nodes.append(helper.make_node("Reshape", ["_scaled", "_fixed_shape"], ["output0"]))
    inits.append(numpy_helper.from_array(np.array([1, 6, 100], dtype=np.int64), "_fixed_shape"))
    _save(
        os.path.join(out_dir, "det_liar.onnx"),
        nodes, inits,
        [_image_input(None, 640)],
        [_output("output0", 1, [6, 100])],
    )

    # ---- segmentation, dynamic batch -------------------------------------
    # output0: [B, 4 + 2 classes + 32 coeffs, D]; output1: [B, 32, 160, 160]
    det_nodes, det_inits = _scaled_constant("output0", detection_constant(38, 100), reshape_dims=2)
    protos = np.linspace(-2.0, 2.0, 32 * 160 * 160, dtype=np.float32).reshape(1, 32, 160, 160)
    proto_nodes, proto_inits = _scaled_constant("output1", protos, reshape_dims=3)
    # Both branches recompute the mean; drop the duplicate ReduceMean/name clash
    proto_nodes = [n for n in proto_nodes if n.op_type != "ReduceMean"]
    _save(
        os.path.join(out_dir, "seg_dynamic.onnx"),
        det_nodes + proto_nodes, det_inits + proto_inits,
        [_image_input(None, 640)],
        [_output("output0", None, [38, 100]), _output("output1", None, [32, 160, 160])],
    )

    # ---- pose, dynamic batch ---------------------------------------------
    nodes, inits = _scaled_constant("output0", pose_constant(100), reshape_dims=2)
    _save(
        os.path.join(out_dir, "pose_dynamic.onnx"),
        nodes, inits,
        [_image_input(None, 640)],
        [_output("output0", None, [56, 100])],
    )

    # ---- OBB, dynamic batch ---------------------------------------------
    nodes, inits = _scaled_constant("output0", obb_constant(15, 100), reshape_dims=2)
    _save(
        os.path.join(out_dir, "obb_dynamic.onnx"),
        nodes, inits,
        [_image_input(None, 640)],
        [_output("output0", None, [20, 100])],
    )

    # ---- YOLO-NAS style, dynamic batch, two outputs ----------------------
    # Two outputs route detection through postprocessNAS, which reads both
    # tensors — so this covers slicing more than one output per batch element.
    boxes = np.zeros((1, 100, 4), dtype=np.float32)
    for d in range(100):
        boxes[0, d, 0] = 200.0 + 4.0 * d   # x1
        boxes[0, d, 1] = 400.0 + 3.0 * d   # y1
        boxes[0, d, 2] = 420.0 + 4.0 * d   # x2
        boxes[0, d, 3] = 580.0 + 3.0 * d   # y2
    scores = np.full((1, 100, 2), 0.05, dtype=np.float32)
    for d in range(0, 100, 17):
        scores[0, d, d % 2] = 1.8
    box_nodes, box_inits = _scaled_constant("output0", boxes, reshape_dims=2)
    score_nodes, score_inits = _scaled_constant("output1", scores, reshape_dims=2)
    score_nodes = [n for n in score_nodes if n.op_type != "ReduceMean"]
    _save(
        os.path.join(out_dir, "nas_dynamic.onnx"),
        box_nodes + score_nodes, box_inits + score_inits,
        [_image_input(None, 640)],
        [_output("output0", None, [100, 4]), _output("output1", None, [100, 2])],
    )

    # ---- detection, dynamic batch AND dynamic input shape ----------------
    # The single-image path stride-aligns the letterbox per image while the
    # batched path must use one shared target, so this model covers the
    # documented divergence between the two.
    nodes, inits = _scaled_constant("output0", detection_constant(6, 100), reshape_dims=2)
    dyn_input = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, ["batch", 3, "height", "width"])
    _save(
        os.path.join(out_dir, "det_dynshape.onnx"),
        nodes, inits,
        [dyn_input],
        [_output("output0", None, [6, 100])],
    )

    # ---- classification, dynamic batch ----------------------------------
    scores = np.linspace(0.05, 0.95, 10, dtype=np.float32).reshape(1, 10)
    nodes, inits = _scaled_constant("output0", scores, reshape_dims=1)
    _save(
        os.path.join(out_dir, "cls_dynamic.onnx"),
        nodes, inits,
        [_image_input(None, 224)],
        [_output("output0", None, [10])],
    )


if __name__ == "__main__":
    write_models(sys.argv[1] if len(sys.argv) > 1 else "models")
