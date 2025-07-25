from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
from typing import Union
import os

def quantize_onnx_model(onnx_model_path: Union[str, Path], quantized_model_path: Union[str, Path], per_channel: bool = False):
    try:
        print(f"Starting quantization of {onnx_model_path}...")
        quantize_dynamic(
            model_input=onnx_model_path, 
            model_output=quantized_model_path, 
            per_channel=per_channel,  
            weight_type=QuantType.QUInt8  # Changed to QUInt8 for better compatibility
        )
        print(f"✅ Quantization completed. Quantized model saved to: {quantized_model_path}")
        return True
    except Exception as e:
        print(f"❌ Error quantizing {onnx_model_path}: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 YOLO Model Quantization Tool")
    print("=" * 50)
    
    onnx_yolov8_path = '../models/yolov8n.onnx'  # Fixed: yolov8n.onnx not yolo8n.onnx
    onnx_yolo11_path = '../models/yolo11n.onnx'   

    quantized_yolo11_path = 'yolo11n_quantized.onnx'
    quantized_yolov8_path = 'yolov8n_quantized.onnx'  # Fixed: consistent naming

    # Check if models exist before quantization
    models_quantized = 0
    
    if os.path.exists(onnx_yolo11_path):
        print(f"📋 Quantizing YOLO11 model: {onnx_yolo11_path}")
        if quantize_onnx_model(onnx_yolo11_path, quantized_yolo11_path, per_channel=True):
            models_quantized += 1
    else:
        print(f"❌ YOLO11 model not found: {onnx_yolo11_path}")
    
    if os.path.exists(onnx_yolov8_path):
        print(f"📋 Quantizing YOLOv8 model: {onnx_yolov8_path}")
        if quantize_onnx_model(onnx_yolov8_path, quantized_yolov8_path, per_channel=True):
            models_quantized += 1
    else:
        print(f"❌ YOLOv8 model not found: {onnx_yolov8_path}")
    
    print("=" * 50)
    print(f"🎉 Quantization completed! {models_quantized} models quantized successfully.")
    
    if models_quantized > 0:
        print("\n📊 Quantization Benefits:")
        print("- Model size reduced by ~75%")
        print("- Inference speed may vary (sometimes slower but uses less memory)")
        print("- Compatible with ONNX Runtime (not OpenCV DNN)")
        print("\n🔧 Test quantized models with:")
        print("./build/yolo_benchmark_suite quantized_models/yolo11n_quantized.onnx models/coco.names")