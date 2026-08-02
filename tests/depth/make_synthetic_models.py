#!/usr/bin/env python3
"""Generate tiny synthetic ONNX models for the depth API tests.

These are not YOLO networks. They reproduce the tensor shapes and dtypes of a
YOLO26-depth export so the C++ estimator can be tested without downloading
weights or installing PyTorch.

depth_synthetic.onnx emits a fixed depth ramp from 1 m to 10 m across the
width, independent of the input, which makes the expected postprocessing
result exactly computable.

not_depth.onnx has a detection-shaped output so the constructor's validation
can be tested.

Requires: onnx, numpy
Usage:    python3 make_synthetic_models.py [output_dir]
"""

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 13
SIZE = 320


def _save(path, nodes, initializers, inputs, outputs):
    graph = helper.make_graph(nodes, os.path.basename(path), inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"wrote {path}")


def write_models(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])

    # ---- depth_synthetic.onnx: constant 1..10 m ramp across the width ----
    ramp = np.linspace(1.0, 10.0, SIZE, dtype=np.float32)
    depth = np.tile(ramp, (SIZE, 1)).reshape(1, 1, SIZE, SIZE)
    # Identity on a constant keeps the graph valid while ignoring the input.
    nodes = [helper.make_node("Identity", ["_depth_const"], ["output0"])]
    inits = [numpy_helper.from_array(depth, "_depth_const")]
    _save(
        os.path.join(out_dir, "depth_synthetic.onnx"),
        nodes, inits, [images],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 1, SIZE, SIZE])],
    )

    # ---- not_depth.onnx: detection-shaped output, must be rejected ----
    det = np.zeros((1, 84, 100), dtype=np.float32)
    nodes = [helper.make_node("Identity", ["_det_const"], ["output0"])]
    inits = [numpy_helper.from_array(det, "_det_const")]
    _save(
        os.path.join(out_dir, "not_depth.onnx"),
        nodes, inits, [images],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 84, 100])],
    )


if __name__ == "__main__":
    write_models(sys.argv[1] if len(sys.argv) > 1 else "models")
