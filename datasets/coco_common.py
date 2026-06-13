"""COCO 轉檔共用定義。

本專案 4 類 canonical schema (見 CONTEXT.md):
  1 uav / 2 person / 3 vehicle / 4 bicycle
所有轉檔腳本輸出同一份 categories,merge 時不需再對齊。
file_name 一律為「相對 data/raw/ 的路徑」,訓練 config 的 img_prefix 指到 data/raw/。
"""

CATEGORIES = [
    {"id": 1, "name": "uav"},
    {"id": 2, "name": "person"},
    {"id": 3, "name": "vehicle"},
    {"id": 4, "name": "bicycle"},
]

SPLITS = ("train", "val", "test")


def coco_skeleton():
    return {"images": [], "annotations": [], "categories": CATEGORIES}


def save_json(obj, path):
    import json
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)
    n_img, n_ann = len(obj["images"]), len(obj["annotations"])
    print(f"  → {path}: {n_img} images, {n_ann} annotations")
