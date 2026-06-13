#!/usr/bin/env python3
"""從 merged COCO json 抽小子集,供 CPU smoke 訓練用。

均勻抽樣 (非隨機) 以保持可重現;依 image id 排序後等距取 N 張。
"""
import argparse
import json
from pathlib import Path


def subset(src_path: Path, dst_path: Path, n: int):
    src = json.loads(src_path.read_text())
    images = sorted(src["images"], key=lambda im: im["id"])
    step = max(1, len(images) // n)
    picked = images[::step][:n]
    ids = {im["id"] for im in picked}
    anns = [a for a in src["annotations"] if a["image_id"] in ids]
    dst = {"images": picked, "annotations": anns, "categories": src["categories"]}
    dst_path.write_text(json.dumps(dst))
    print(f"{src_path.name} → {dst_path.name}: {len(picked)} images, {len(anns)} anns")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ann-dir", default="data/coco/annotations")
    ap.add_argument("--train-n", type=int, default=200)
    ap.add_argument("--val-n", type=int, default=50)
    args = ap.parse_args()

    d = Path(args.ann_dir)
    subset(d / "train.json", d / "train_subset.json", args.train_n)
    subset(d / "val.json", d / "val_subset.json", args.val_n)


if __name__ == "__main__":
    main()
