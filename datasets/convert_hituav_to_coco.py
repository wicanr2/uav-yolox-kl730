#!/usr/bin/env python3
"""HIT-UAV normal_json → COCO (canonical 4 類)。

來源是「近似 COCO」,有三個已實測確認的陷阱 (2026-06-11 查證):
  1. top-level key 是 `annotation` (單數),不是標準的 `annotations`。
  2. image 條目用 `filename`,不是標準的 `file_name`。
  3. README 宣稱 bbox 為 [xc,yc,w,h],但實檔 normal_json 的 bbox 是
     [xmin,ymin,w,h] (由 segmentation 多邊形驗證) — 直接沿用,不做中心點轉換。

類別映射 (HIT-UAV id → canonical id):
  0 Person → 2 person / 1 Car → 3 vehicle / 2 Bicycle → 4 bicycle /
  3 OtherVehicle → 3 vehicle / 4 DontCare → 丟棄
"""
import argparse
import json
from pathlib import Path

from coco_common import SPLITS, coco_skeleton, save_json

CAT_MAP = {0: 2, 1: 3, 2: 4, 3: 3, 4: None}


def convert_split(root: Path, prefix: str, split: str):
    src = json.loads((root / "normal_json" / "annotations" / f"{split}.json").read_text())
    coco = coco_skeleton()

    for im in src["images"]:
        fname = im.get("file_name") or im["filename"]  # 陷阱 2
        coco["images"].append({
            "id": im["id"],
            "file_name": f"{prefix}/normal_json/{split}/{fname}",
            "width": im["width"],
            "height": im["height"],
        })

    src_anns = src.get("annotations") or src.get("annotation")  # 陷阱 1
    ann_id = 1
    for a in src_anns:
        cat = CAT_MAP.get(a["category_id"])
        if cat is None:  # DontCare
            continue
        x, y, w, h = a["bbox"]  # 已是 [xmin,ymin,w,h],見陷阱 3
        coco["annotations"].append({
            "id": ann_id,
            "image_id": a["image_id"],
            "category_id": cat,
            "bbox": [x, y, w, h],
            "area": a.get("area", w * h),
            "iscrowd": 0,
        })
        ann_id += 1
    return coco


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/raw/HIT-UAV")
    ap.add_argument("--out-dir", default="data/coco/annotations")
    args = ap.parse_args()

    root = Path(args.root)
    for split in SPLITS:
        print(f"[HIT-UAV] {split}")
        coco = convert_split(root, root.name, split)
        save_json(coco, Path(args.out_dir) / f"hituav_{split}.json")


if __name__ == "__main__":
    main()
