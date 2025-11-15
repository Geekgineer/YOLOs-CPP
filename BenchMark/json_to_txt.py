import json
import os
from tqdm import tqdm

json_path = "annotations/instances_val2017.json"
out_dir = "labels_val2017"

os.makedirs(out_dir, exist_ok=True)

with open(json_path, "r") as f:
    data = json.load(f)

# --- Build COCO → YOLO class mapping ---
coco_ids = sorted([cat["id"] for cat in data["categories"]])
id_to_yolo = {cid: i for i, cid in enumerate(coco_ids)}

# Create maps
id_to_filename = {img["id"]: img for img in data["images"]}

annotations_by_image = {}
for ann in data["annotations"]:
    img_id = ann["image_id"]
    annotations_by_image.setdefault(img_id, []).append(ann)

for img_id, img_info in tqdm(id_to_filename.items()):

    name = img_info["file_name"].replace(".jpg", "")
    w, h = img_info["width"], img_info["height"]

    fout = open(os.path.join(out_dir, name + ".txt"), "w")

    anns = annotations_by_image.get(img_id, [])

    for ann in anns:
        cid = ann["category_id"]
        cls = id_to_yolo[cid]

        x, y, bw, bh = ann["bbox"]

        # --- Clip to image borders ---
        x = max(0, x)
        y = max(0, y)
        bw = min(bw, w - x)
        bh = min(bh, h - y)

        # skip degenerate boxes
        if bw <= 1 or bh <= 1:
            continue

        # convert to YOLO normalized
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h

        fout.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    fout.close()

print("Done!")

