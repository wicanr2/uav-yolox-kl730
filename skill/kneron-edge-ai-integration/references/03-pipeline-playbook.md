# 03 · 七站 Pipeline Playbook (逐站指令)

```
1.蒐集 → 2.轉COCO → 3.訓練 → 4.匯出ONNX → 5.量化編譯NEF → 6.後處理 → 7.部署
```

## 站 1-2 · 資料 → COCO

多來源資料集格式各異,全轉成統一 COCO (固定 categories) 再 merge。
**轉檔器各自獨立輸出同一份 categories,merge 只做 image/annotation id 重編** → 轉檔期就對齊,不留到訓練期。
資料集格式陷阱 (HIT-UAV 偽 COCO key 是單數 annotation、欄位 filename;tracking 格式抽幀) 見專案註解。

## 站 3 · 訓練 (YOLOX)

```bash
python tools/train.py configs/yolox/yolox_s_..._img_norm.py --work-dir ...
```
- config 用 Kneron 的 `img_norm` 版 (activation 已換 LeakyReLU、mean128/std256 — NPU 友善,Kneron 出廠改好的)。
- 只改 num_classes、lr (線性縮放: base 0.01 對應總 batch 64)、evaluation `classwise=True` (類別不平衡時看逐類 AP)。
- **CPU-only 策略**:本機只跑 1-epoch smoke 驗管線;正式訓練丟 Kaggle 免費 GPU (30h/週)。
  Notebook 用相同版本鎖 (uv venv + cu113),抓回 latest.pth 即可。

## 站 4 · 匯出 ONNX

```bash
python tools/deployment/pytorch2onnx_kneron.py CONFIG CKPT \
    --output-file out.onnx --skip-postprocess --shape 640 640
```
- opset 鎖 11。`--skip-postprocess` = 只到 raw 輸出 (9 個張量: cls/reg/obj × 3 尺度),decode 不進 ONNX。
- 此步**與平台無關**;平台選擇在站 5 的 ModelConfig。
- 腳本尾段舊版優化器可能 crash 但檔案已存 → 以 `onnx.checker.check_model` 驗有效性,別看 exit code。

## 站 5 · 量化編譯 → NEF (toolchain docker, conda env onnx1.13)

```python
import onnx, ktc, kneronnxopt
m = kneronnxopt.optimize(onnx.load("out.onnx"))          # 新版優化器接手
km = ktc.ModelConfig(20008, "0001", "730", onnx_model=m) # platform 字串 "730"
print(km.evaluate())                                      # 估 fps + 抓 cpu_node (要 N/A)
bie = km.analysis({input_name: calib_imgs},               # PTQ 量化, range_method="mmse" (見 ref 02)
                  datapath_range_method="mmse", threads=4)
nef = ktc.compile([km])                                   # → models_730.nef
```
官方 YOLO 部署 trick:多 scale 輸出不要 concat、cls 與 bbox 分開輸出、sigmoid/exp 留模型內。

## 站 6 · 後處理 (decode + NMS) — 跑在晶片 CPU 核,非 NPU

NEF 吐 raw 張量,要自己 decode (純 numpy,板上跑 A55):
```
cx=(reg_x+gx)*stride; cy=(reg_y+gy)*stride; w=exp(reg_w)*stride; h=exp(reg_h)*stride
score = sigmoid(obj) * sigmoid(cls)        # skip-postprocess 無 sigmoid,這裡補
→ 過 conf 閾值 → 逐類 NMS
```
依 channel 數 (cls=num_cls / reg=4 / obj=1) 分組、依空間大小配對尺度 → 對輸出順序穩健。

## 站 7 · 部署 (Kneron PLUS, 實體板)

```
kp.core.connect_devices() → load_model_from_file(nef)
→ generic_image_inference_send/receive → retrieve_float_node (反量化) → 站6 後處理
```
- KL730 自帶 A55 跑 Linux,host 就是板子本身 (免外接 PC)。
- decode/NMS/前處理在「模擬器 backend」與「板上 backend」**共用同一份**,只換 NPU 推論那塊 → 降低 sim→board 行為漂移。
- 結果經 UART/MAVLink (低頻寬框) + Ethernet/RTSP (帶 overlay 視訊) 下傳。

## 每站的 pass/fail 訊號 (feedback loop 優先)

| 站 | 驗證 | 抓什麼 |
|---|---|---|
| 2→3 | 載入 dataset | json/路徑/類別對 |
| 3 | 1-epoch smoke | loss 降、checkpoint 生 |
| 5 | `evaluate()` | fps + `cpu_node: N/A` |
| 5 | 量化 `--check` | 逐 channel corr >0.9 |
| 5→6 | mAP harness | 量化 vs float 掉幅 <3% |
| 6 | 端到端 demo | 真實圖出框 |
