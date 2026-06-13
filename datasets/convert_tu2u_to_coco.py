#!/usr/bin/env python3
"""ThermalUAV2UAV (YOLO format) → COCO。

來源格式: {split}/images/*.png + {split}/labels/*.txt
  每行: `0 cx cy w h` (normalized 中心點格式),類別固定 0 = UAV。
轉換: cx,cy,w,h → [xmin, ymin, w, h] 絕對座標;類別 0 → canonical id 1 (uav)。
影像尺寸逐張從 PNG 讀,不寫死 640×512。
"""
import argparse
from pathlib import Path

from PIL import Image

from coco_common import SPLITS, coco_skeleton, save_json


def convert_split(root: Path, prefix: str, split: str):
    coco = coco_skeleton()
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    img_id = ann_id = 1
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        w_img, h_img = Image.open(img_path).size
        coco["images"].append({
            "id": img_id,
            "file_name": f"{prefix}/{split}/images/{img_path.name}",
            "width": w_img,
            "height": h_img,
        })
        txt = lbl_dir / (img_path.stem + ".txt")
        if txt.exists():
            for line in txt.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _cls, cx, cy, w, h = (float(v) for v in parts)
                bw, bh = w * w_img, h * h_img
                x, y = cx * w_img - bw / 2, cy * h_img - bh / 2
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,  # uav
                    "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                })
                ann_id += 1
        img_id += 1
    return coco


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/raw/ThermalUAV2UAV_Dataset")
    ap.add_argument("--out-dir", default="data/coco/annotations")
    args = ap.parse_args()

    root = Path(args.root)
    prefix = root.name  # file_name 相對 data/raw/ 的第一層
    for split in SPLITS:
        print(f"[TU2U] {split}")
        coco = convert_split(root, prefix, split)
        save_json(coco, Path(args.out_dir) / f"tu2u_{split}.json")


if __name__ == "__main__":
    main()
