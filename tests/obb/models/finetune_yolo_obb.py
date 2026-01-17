from ultralytics import YOLO
from tqdm.auto import tqdm

def main():

    print("Finetuning OBB models ...")

    models_to_finetune = [
        "yolov8n.pt",
        "yolo11n.pt",
        "yolo12n.pt"
    ]
     
    dataset = "DOTAv1.yaml"

    for model_name in tqdm(models_to_finetune, desc="Finetuning OBB models", unit="model"):

        model = YOLO(model_name)
        model.train(data=dataset, epochs=2, imgsz=640, task="obb") 

        print(f"Successfully finetuned {model_name} for OBB ...")

if __name__ == "__main__":

    main()
