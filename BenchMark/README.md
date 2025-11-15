

# YOLO Evaluation Tool

This tool evaluates an onnx YOLO model on COCOVal2017.

---

## Build Instructions

1. Make the preparation script executable and run it:

```bash
chmod +x prepare.sh
./prepare.sh
```

2. Build the project:

```bash
./build.sh
```

3. Run the evaluation on CPU:

```bash
cd build
./evaluate 0
```
On GPU:
```bash
./evaluate 1
```

You can also specify a custom model path:

```bash
./evaluate 0 ../../models/yolo12n.onnx
```

---

## Example Output

```
Model loaded successfully with 1 input nodes and 1 output nodes.
Found 5000 images.

=== Evaluation Results ===
IoU 0.5   AP=0.53056
IoU 0.55  AP=0.514973
IoU 0.6   AP=0.495787
IoU 0.65  AP=0.471318
IoU 0.7   AP=0.440934
IoU 0.75  AP=0.394625
IoU 0.8   AP=0.337166
IoU 0.85  AP=0.264971
IoU 0.9   AP=0.175256

AP50 = 0.53056
mAP50-95 = 0.402843

=== Speed ===
Images processed = 5000
Inference time (mean ± stddev): 73.7667 ± 11.402 ms
FPS (from mean): 13.5563
```

