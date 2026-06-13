#!/usr/bin/env python3
"""把各來源的 COCO json 合併成最終 train/val/test.json。

各來源已輸出相同的 canonical categories 與「相對 data/raw/ 的 file_name」,
這裡只做 image/annotation id 重新編號後串接。
缺少的來源檔 (例如 Anti-UAV410 尚未下載) 印警告後略過。
"""
import argparse
import json
from pathlib import Path

from coco_common import CATEGORIES, SPLITS, save_json

SOURCES = ("tu2u", "hituav", "antiuav410")


def merge(ann_dir: Path, split: str):
    merged = {"images": [], "annotations": [], "categories": CATEGORIES}
    img_off = ann_off = 0
    for src in SOURCES:
        path = ann_dir / f"{src}_{split}.json"
        if not path.exists():
            print(f"  警告: {path} 不存在,略過此來源")
            continue
        part = json.loads(path.read_text())
        assert part["categories"] == CATEGORIES, f"{path} categories 不符 canonical schema"
        id_map = {}
        for im in part["images"]:
            img_off += 1
            id_map[im["id"]] = img_off
            merged["images"].append({**im, "id": img_off})
        for a in part["annotations"]:
            ann_off += 1
            merged["annotations"].append(
                {**a, "id": ann_off, "image_id": id_map[a["image_id"]]})
        print(f"  + {src}: {len(part['images'])} images / {len(part['annotations'])} anns")
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ann-dir", default="data/coco/annotations")
    args = ap.parse_args()

    ann_dir = Path(args.ann_dir)
    for split in SPLITS:
        print(f"[merge] {split}")
        save_json(merge(ann_dir, split), ann_dir / f"{split}.json")


if __name__ == "__main__":
    main()
