from ultralytics import YOLO
from tqdm.auto import tqdm

def main():

    print("Finetuning models ...")

    models_to_finetune = [
        "yolov5nu.pt",
        "yolov6n.yaml",
        "yolov8n.pt",
        "yolov9t.pt",
        "yolov10n.pt",
        "yolo11n.pt",
        "yolo12n.pt",
        "yolo26n.pt"
    ]
     
    dataset = "VOC.yaml"

    for model_name in tqdm(models_to_finetune, desc="Finetuning models", unit="model"):

        model = YOLO(model_name)
        model.train(data=dataset, epochs=2, imgsz=320) 

        print(f"Successfully finetuned {model_name} ...")

if __name__ == "__main__":

    main()

