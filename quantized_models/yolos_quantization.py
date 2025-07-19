from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

def quantize_onnx_model(
    onnx_model_path: str,
    quantized_model_path: str,
    per_channel: bool = False
):
    """
    Quantizes an ONNX model using dynamic quantization.

    Args:
        onnx_model_path (str): Path to the original ONNX model file.
        quantized_model_path (str): Path to save the quantized ONNX model.
        per_channel (bool): If True, uses per-channel quantization (better for conv nets).
    """
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=quantized_model_path,
        per_channel=per_channel,
        weight_type=QuantType.QUInt8,
    )
    print(f"✅ Quantization completed.\nSaved: {quantized_model_path}")

if __name__ == "__main__":
    # Path to the source ONNX model 
    onnx_model_path = "../models/yolov8s.onnx"
    # Output filename for quantized model 
    quantized_model_path = "yolov8s_uint8.onnx"

    # Run quantization
    quantize_onnx_model(onnx_model_path, quantized_model_path, per_channel=True)
