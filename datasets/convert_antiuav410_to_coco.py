#!/usr/bin/env python3
"""Anti-UAV410 (tracking 格式) 抽幀 → COCO (全部映射為 uav)。

來源結構: AntiUAV410/{split}/{序列}/N.jpg + IR_label.json
  IR_label.json = {"exist": [0/1,...], "gt_rect": [[x,y,w,h],...]}
  兩陣列與幀數一一對應;exist=0 表遮蔽/出視野 (gt_rect 可能為空)。

策略:
  - 每序列以 --stride 抽幀 (預設 10),避免相鄰幀近重複灌水。
  - 只取 exist=1 且 rect 合法的幀。
  - 解析度逐序列從第一張實際幀讀取 (官方未載明,不可寫死 640×512)。
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from coco_common import SPLITS, coco_skeleton, save_json


def convert_split(root: Path, prefix: str, split: str, stride: int):
    coco = coco_skeleton()
    img_id = ann_id = 1
    split_dir = root / split
    if not split_dir.is_dir():
        print(f"  (找不到 {split_dir},略過)")
        return None
    for seq_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label = json.loads((seq_dir / "IR_label.json").read_text())
        frames = sorted(seq_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        n = min(len(frames), len(label["gt_rect"]))
        if len(frames) != len(label["gt_rect"]):
            print(f"  警告: {seq_dir.name} 幀數 {len(frames)} != 標註數 {len(label['gt_rect'])},取交集")
        if n == 0:
            continue
        w_img, h_img = Image.open(frames[0]).size  # 同序列解析度固定
        for i in range(0, n, stride):
            if not label["exist"][i]:
                continue
            rect = label["gt_rect"][i]
            if not rect or len(rect) != 4:
                continue
            x, y, w, h = rect
            if w <= 0 or h <= 0:
                continue
            coco["images"].append({
                "id": img_id,
                "file_name": f"{prefix}/{split}/{seq_dir.name}/{frames[i].name}",
                "width": w_img,
                "height": h_img,
            })
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,  # uav
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            img_id += 1
            ann_id += 1
    return coco


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/raw/AntiUAV410")
    ap.add_argument("--out-dir", default="data/coco/annotations")
    ap.add_argument("--stride", type=int, default=10, help="抽幀間隔 (預設 10)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"[Anti-UAV410] {root} 不存在 (尚未手動下載?),整包略過 — 不影響 TU2U/HIT-UAV 流程")
        return
    for split in SPLITS:
        print(f"[Anti-UAV410] {split} (stride={args.stride})")
        coco = convert_split(root, root.name, split, args.stride)
        if coco is not None:
            save_json(coco, Path(args.out_dir) / f"antiuav410_{split}.json")


if __name__ == "__main__":
    main()
