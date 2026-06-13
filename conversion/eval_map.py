#!/usr/bin/env python3
"""量化驗收:在標註 val set 上比對 float ONNX 與 NEF (INT8) 的 COCO mAP。

在 kneron/toolchain container 內跑 (有 ktc 模擬器 + pycocotools)。
合格線: NEF 相對 float 的 mAP 掉幅 < 3% (PLAN §3.3)。

⚠️ 需要「4 類訓練後模型」才有意義 — COCO 80 類 zoo 模型對不上 4 類 val.json。
   正式流程: Kaggle 訓完 → make export-onnx → make nef → 本腳本驗收。

用法 (container 內):
  python eval_map.py --onnx <opt.onnx> --nef <models_730.nef> \
      --ann data/coco/annotations/val.json --img-root data/raw --num 500
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/data1/firmware/detect")
from postprocess import postprocess  # noqa: E402

# canonical 4 類 → COCO category_id (見 datasets/coco_common.py)
CAT_IDS = [1, 2, 3, 4]


def preprocess(path, size):
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size
    arr = np.array(img.resize((size, size), Image.BILINEAR)).astype(np.float32) / 256.0 - 0.5
    return np.transpose(arr, (2, 0, 1))[None], (w0, h0)


def run_set(infer_fn, images, img_root, size, conf):
    """對每張圖跑推論+後處理,輸出 COCO detection 格式 list。"""
    results = []
    for im in images:
        x, (w0, h0) = preprocess(Path(img_root) / im["file_name"], size)
        outs = infer_fn(x)
        dets = postprocess(outs, input_size=size, conf_thres=conf)
        sx, sy = w0 / size, h0 / size
        for x0, y0, x1, y1, score, cls in dets:
            results.append({
                "image_id": im["id"],
                "category_id": CAT_IDS[cls] if cls < len(CAT_IDS) else cls + 1,
                "bbox": [x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy],  # xywh
                "score": float(score),
            })
    return results


def coco_map(ann_file, results, used_img_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(ann_file)
    if not results:
        return 0.0
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.imgIds = list(used_img_ids)
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0])  # mAP@[.5:.95]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True, help="kneronnxopt 優化後的 float ONNX")
    ap.add_argument("--nef", required=True)
    ap.add_argument("--ann", required=True, help="COCO val.json")
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--platform", type=int, default=730)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05, help="mAP 評估用低閾值")
    ap.add_argument("--num", type=int, default=500, help="抽幾張 val 圖 (全跑很慢)")
    args = ap.parse_args()

    import ktc

    ann = json.loads(Path(args.ann).read_text())
    images = ann["images"]
    step = max(1, len(images) // args.num)
    images = images[::step][:args.num]
    ids = {im["id"] for im in images}
    # 只留抽中影像的標註,另存暫存 ann 供 COCO 評估
    sub = {"images": images,
           "annotations": [a for a in ann["annotations"] if a["image_id"] in ids],
           "categories": ann["categories"]}
    sub_path = "/tmp/val_sub.json"
    Path(sub_path).write_text(json.dumps(sub))
    print(f"評估 {len(images)} 張 val 影像")

    in_name = "input"
    float_fn = lambda x: ktc.kneron_inference(  # noqa: E731
        [x], onnx_file=args.onnx, input_names=[in_name], platform=args.platform)
    nef_fn = lambda x: ktc.kneron_inference(    # noqa: E731
        [x], nef_file=args.nef, input_names=[in_name], platform=args.platform)

    print("\n--- float ONNX ---")
    m_float = coco_map(sub_path, run_set(float_fn, images, args.img_root, args.size, args.conf), ids)
    print("\n--- NEF (INT8) ---")
    m_nef = coco_map(sub_path, run_set(nef_fn, images, args.img_root, args.size, args.conf), ids)

    drop = (m_float - m_nef) / m_float * 100 if m_float > 0 else float("nan")
    print("\n=== 量化驗收 ===")
    print(f"float mAP = {m_float:.4f}")
    print(f"NEF   mAP = {m_nef:.4f}")
    print(f"掉幅      = {drop:.2f}%  → {'通過 (<3%)' if drop < 3 else '未通過 (≥3%,需調量化)'}")


if __name__ == "__main__":
    main()
