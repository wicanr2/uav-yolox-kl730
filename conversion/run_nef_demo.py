#!/usr/bin/env python3
"""端到端 demo:NEF → 模擬器推論 → decode+NMS → 畫框存 PNG。

在 kneron/toolchain container 內跑 (有 ktc 模擬器 + PIL)。
用 COCO 80 類 zoo NEF 跑 HIT-UAV 熱影像 (內含 person/car,COCO 認得的類別),
驗證「整條推論鏈」可運作 — 比 per-channel corr 更直觀的成果。

注意:zoo 模型是 COCO 可見光訓練,餵熱影像有 domain gap,偵測可能偏弱;
這個 demo 的目的是驗證 pipeline 正確,不是驗證精度 (精度要等熱影像 fine-tune)。
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/data1/firmware/detect")
from postprocess import postprocess  # noqa: E402

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee"] + \
       [f"cls{i}" for i in range(30, 80)]


def preprocess(path, size):
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size
    arr = np.array(img.resize((size, size), Image.BILINEAR)).astype(np.float32) / 256.0 - 0.5
    return np.transpose(arr, (2, 0, 1))[None], img, (w0, h0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nef", default="/data1/kneron_flow/models_730.nef")
    ap.add_argument("--platform", type=int, default=730)
    ap.add_argument("--img-dir", default="/data1/data/raw/HIT-UAV/normal_json/test")
    ap.add_argument("--out-dir", default="/data1/artifacts/demo")
    ap.add_argument("--num", type=int, default=6, help="抽幾張圖跑")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    import ktc  # toolchain container 內才有

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(args.img_dir).iterdir()
                   if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
    step = max(1, len(paths) // args.num)
    picked = paths[::step][:args.num]
    print(f"demo: {len(picked)} 張 (自 {len(paths)} 張)")

    summary = []
    for p in picked:
        x, img, (w0, h0) = preprocess(p, args.size)
        outs = ktc.kneron_inference([x], nef_file=args.nef,
                                    input_names=["input"], platform=args.platform)
        dets = postprocess(outs, input_size=args.size, conf_thres=args.conf)
        sx, sy = w0 / args.size, h0 / args.size
        draw = ImageDraw.Draw(img)
        for x0, y0, x1, y1, score, cls in dets:
            box = [x0 * sx, y0 * sy, x1 * sx, y1 * sy]
            draw.rectangle(box, outline=(255, 0, 0), width=2)
            name = COCO[cls] if cls < len(COCO) else str(cls)
            draw.text((box[0], max(0, box[1] - 10)), f"{name} {score:.2f}", fill=(255, 255, 0))
        out_path = out_dir / f"det_{p.stem}.png"
        img.save(out_path)
        cls_cnt = {}
        for *_, c in dets:
            cls_cnt[COCO[c] if c < len(COCO) else c] = cls_cnt.get(COCO[c] if c < len(COCO) else c, 0) + 1
        print(f"  {p.name}: {len(dets)} 框 {cls_cnt} → {out_path.name}")
        summary.append((p.name, len(dets), cls_cnt))

    total = sum(n for _, n, _ in summary)
    print(f"\n完成。{len(picked)} 張共 {total} 個偵測框,圖存於 {out_dir}")
    print("(zoo=COCO 可見光模型跑熱影像;有框=pipeline 通,框準不準要等熱影像 fine-tune)")


if __name__ == "__main__":
    main()
