



from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
from typing import Union

def quantize_onnx_model(onnx_model_path: Union[str, Path], quantized_model_path: Union[str, Path], per_channel: bool = False):
    """
    Quantizes an ONNX model and saves the quantized version.

    Args:
        onnx_model_path: Path to the original ONNX model file.
        quantized_model_path: Path to save the quantized model.
        per_channel: If True, quantizes weights per channel instead of per layer.
            Per-channel quantization can improve model accuracy by allowing each output channel
            to have its own scale and zero-point, which better captures the distribution of weights.
            This is especially beneficial for complex models with many channels or varying value ranges.
            Use this option when:
            - The model is complex (e.g., deep convolutional networks).
            - You observe accuracy degradation with per-layer quantization.
    """
    # Quantize the model
    quantize_dynamic(
        model_input=onnx_model_path, 
        model_output=quantized_model_path, 
        per_channel=per_channel,  # Set to True if per-channel quantization is desired
        weight_type=QuantType.QUInt8  # Specify the weight type for quantization
    )
    
    print("Quantization completed. Quantized model saved to:", quantized_model_path)

if __name__ == "__main__":

    # Load the original ONNX model file path
    # onnx_model_path = 'yolo5-n6.onnx' 
    # onnx_model_path = 'yolo7-tiny.onnx' 
    # onnx_model_path = 'yolo8n.onnx'
    onnx_model_path = 'yolo10n.onnx'   # Change this to your desired model

    # Specify the output path for the quantized model
    quantized_model_path = 'yolo10n.onnx'  # Change this to your desired output file name

    # Call the quantization function
    quantize_onnx_model(onnx_model_path, quantized_model_path, per_channel=True)  # Change True to False as needed
