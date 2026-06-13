#!/usr/bin/env python3
"""ONNX → KL730 NEF (在 kneron/toolchain docker 內執行)。

流程 (官方 toolchain v0.32.x):
  1. kneronnxopt.optimize          — graph 優化
  2. ktc.ModelConfig(..., "730")   — KL730 platform id 為字串 "730"
  3. km.evaluate()                 — NPU-only 效能模擬 (估 fps、抓 CPU fallback node)
  4. km.analysis(calib)            — PTQ 量化 → BIE
  5. ktc.compile([km])             — 編譯 → NEF
  6. (--check) bie/nef 模擬推論 sanity check — 防「量化後輸出全 0」事故

注意:
  - 必須在 toolchain 預設 conda env `onnx1.13` 執行 (base env 不支援 730)。
  - 校正前處理必須與訓練 img_norm 一致: x/256 - 0.5 (mean=128, std=256)。
  - 通道順序: 訓練 config to_rgb=True → 預設 RGB。官方 yolox 教學的校正範例
    有做 BGR 翻轉 ([..., ::-1]);熱影像三通道為灰階複製時兩者等價,
    彩色 colormap 影像則有差 — 用 --bgr 可切換,實機驗證時以 mAP 高者為準。
  - layout: toolchain >= v0.21.0 要求輸入 shape 與 ONNX 相同 (NCHW)。
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnx
from PIL import Image


def load_calib_images(calib_dir, size, num, bgr, layout):
    paths = sorted(
        p for p in Path(calib_dir).rglob("*")
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not paths:
        sys.exit(f"校正目錄 {calib_dir} 內沒有影像")
    step = max(1, len(paths) // num)
    picked = paths[::step][:num]
    print(f"校正集: {len(picked)} 張 (自 {len(paths)} 張均勻抽樣)")

    imgs = []
    for p in picked:
        img = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.array(img).astype(np.float32)
        if bgr:
            arr = arr[..., ::-1]
        arr = arr / 256.0 - 0.5  # 與訓練 img_norm (mean=128, std=256) 一致
        if layout == "nchw":
            arr = np.transpose(arr, (2, 0, 1))[None]  # 1x3xHxW
        imgs.append(arr)
    return imgs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib-dir", required=True, help="量化校正影像目錄 (建議用 val split)")
    ap.add_argument("--out-dir", default="/data1/artifacts")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--num", type=int, default=200, help="校正張數")
    ap.add_argument("--model-id", type=int, default=20008)
    ap.add_argument("--platform", default="730", choices=["520", "720", "530", "630", "730"],
                    help="目標平台 (720 可作為 730 異常時的鑑別對照組)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bgr", action="store_true", help="校正影像轉 BGR (對齊官方教學作法)")
    ap.add_argument("--layout", choices=["nchw", "hwc"], default="nchw")
    # 量化 bitwidth — int8 最省記憶體 (mix 模式做 int16 模擬會吃爆 RAM, 已致一次當機)
    ap.add_argument("--datapath-mode", default="int8",
                    help='datapath_bitwidth_mode: int8/int16/"mix balance"/"mix light"/mixbw')
    ap.add_argument("--weight-mode", default="int8",
                    help='weight_bitwidth_mode: int8/int16/int4/"mix balance"/"mix light"/mixbw')
    ap.add_argument("--out-mode", default="int8", choices=["int8", "int16"],
                    help="model_out_bitwidth_mode")
    # [關鍵 2026-06-12] range 估計法。percentage=1.0 對 raw-logit 輸出 (--skip-postprocess)
    # 會把範圍張到含 -15 離群值 → 崩潰成常數。mmse 為 SNR-based, 對離群值穩健。
    ap.add_argument("--range-method", default="percentage", choices=["percentage", "mmse"],
                    help="datapath_range_method (raw-logit 模型建議 mmse)")
    ap.add_argument("--percentage", type=float, default=1.0,
                    help="percentage 法的範圍百分位 (detection 官方建議 1.0; raw-logit 可試 0.999)")
    ap.add_argument("--check", action="store_true", help="量化/編譯後跑 bie+nef 模擬推論")
    args = ap.parse_args()

    import ktc           # noqa: E402  (toolchain docker 內才有)
    import kneronnxopt   # noqa: E402

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. optimize
    m = onnx.load(args.onnx)
    m = kneronnxopt.optimize(m)
    opt_path = out_dir / (Path(args.onnx).stem + ".opt.onnx")
    onnx.save(m, str(opt_path))
    input_name = m.graph.input[0].name
    print(f"優化完成 → {opt_path} (input: {input_name})")

    # 2-3. KL730 config + NPU 效能評估
    km = ktc.ModelConfig(args.model_id, "0001", args.platform, onnx_model=m)
    print("\n=== NPU 效能評估 (留意 CPU fallback node 與估算 fps) ===")
    print(km.evaluate())

    # 4. PTQ 量化 → BIE
    # detection 模型官方建議 percentage=1.0 (manual_4_bie)
    img_list = load_calib_images(args.calib_dir, args.size, args.num, args.bgr, args.layout)
    analysis_kw = dict(
        threads=args.threads,
        datapath_range_method=args.range_method,
        datapath_bitwidth_mode=args.datapath_mode,
        weight_bitwidth_mode=args.weight_mode,
        model_out_bitwidth_mode=args.out_mode,
    )
    if args.range_method == "percentage":
        analysis_kw["percentage"] = args.percentage
    print(f"量化參數: {analysis_kw}")
    bie_path = km.analysis({input_name: img_list}, **analysis_kw)
    print(f"量化完成 → {bie_path}")

    # 5. compile → NEF
    nef_path = ktc.compile([km])
    print(f"編譯完成 → {nef_path}")

    # 6. sanity check: ONNX float 為基準,比對 NEF 量化輸出的相關性
    #    (比「非全 0」更有鑑別力 — 可抓出論壇回報過的「量化後輸出退化」問題)
    if args.check:
        sample = [img_list[0]]
        plat = int(args.platform)
        ref = ktc.kneron_inference(
            sample, onnx_file=str(opt_path), input_names=[input_name], platform=plat)
        nef_outs = ktc.kneron_inference(
            sample, nef_file=str(nef_path), input_names=[input_name], platform=plat)
        # 逐 channel 比對 (730 為 per-channel 量化, 攤平整個 tensor 算 corr 會誤判)
        print("\n=== sanity check: ONNX float vs NEF 量化 (逐 channel) ===")
        worst = 1.0
        for i, (r, q) in enumerate(zip(ref, nef_outs)):
            r, q = np.asarray(r)[0], np.asarray(q)[0]  # C,H,W
            cs, dead = [], 0
            for c in range(r.shape[0]):
                rc, qc = r[c].ravel(), q[c].ravel()
                if rc.std() < 1e-6:
                    continue  # float 端本身無變化的 channel 不計
                if qc.std() < 1e-6:
                    dead += 1  # float 有變化但 NEF 是常數 → 該 channel 崩潰
                    continue
                cs.append(float(np.corrcoef(rc, qc)[0, 1]))
            mean_c = float(np.mean(cs)) if cs else 0.0
            worst = min(worst, mean_c if not dead else 0.0)
            print(f"output[{i}]: 有效 ch {len(cs)}, 崩潰 ch {dead}, mean per-ch corr={mean_c:.3f}")
        verdict = "通過" if worst > 0.9 else ("勉強 (建議檢查量化設定)" if worst > 0.5 else "未通過 — 量化退化")
        print(f"判定: {verdict} (最低 {worst:.3f}; 合格線 mean per-ch corr > 0.9 且無崩潰 ch)")

    print("\n完成。NEF 可進 Kneron PLUS 部署 (KL730 板)。")


if __name__ == "__main__":
    main()
