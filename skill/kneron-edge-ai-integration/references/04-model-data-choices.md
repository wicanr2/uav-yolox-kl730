# 04 · 模型 / 資料 / 輸入尺寸的選型原則

## 為什麼 YOLOX,不選最新版 YOLO

「版本能動越新越好」是迷思 —— 正確標準是「**能在目標 NPU 正確量化執行的最新版**」。

| 版本 | 授權 | NPU 相容性 |
|---|---|---|
| **YOLOX (2021)** | **Apache-2.0** ✅ | Kneron 官方流程驗證,本案首選 |
| YOLOv5/v8/v11 | AGPL-3.0 (商用要付費) | v8/v11 官方說可改造,但有量化後輸出全 0 公開案例 |
| YOLOv9 | GPL-3.0 (無商業授權) | — |
| YOLOv10/YOLO26 | AGPL | **NMS-free 的 TopK 子圖在各家 NPU (OpenVINO/EdgeTPU/TensorRT) 量化都踩雷** |

- 商用閉源 → 避開 Ultralytics 全系列 (AGPL),選 YOLOX (Apache) 或買企業授權。
- 新版的 NMS-free「省一步後處理」優勢,在 NPU 上反而因 TopK 不被支援而成負擔。

## 晶片選型 (KL730 vs 前代)

| | KL730 | KL720 | KL630 |
|---|---|---|---|
| 算力 | 3.6 eTOPS INT8 | 1.4 TOPS | 0.5 eTOPS |
| CPU | 4×Cortex-A55 **可獨立跑 Linux** | Cortex-M4 (需 host) | Cortex-A5 |
| YOLOv5s 模擬 fps | ~52 | ~25 | ~17 |
| 影像 | MIPI 直入 + ISP, 4K60 | 4K | 5M |

選 KL730:獨立跑 Linux 免外接 host (省重量/功耗);吞吐約前代 2-3 倍;MIPI 直入感測器。
注意:KL730 只有 SBC/96board 形態 (無 dongle/M.2);板級工作溫度常為 0–50°C 消費級,工業版需確認。

## 資料集整合原則

- 多來源混用要注意 **視角 domain gap** (空對空/空對地/地對空) 與 **感測器差異** → 不同任務的偵測頭分開評估。
- **不可跨視角混測**;各用自己官方 test split 報數。
- 授權先查:商用要乾淨授權 (MIT/CC-BY/Apache);需註冊或授權頁失效的 (如某些 ADAS 集) 先 bypass。
- 影片資料集要**抽幀** (相鄰幀近重複,全收是灌水)。
- 類別不平衡先用 `classwise=True` 看逐類 AP,別急著改 sampler (YOLOX 的 MultiImageMixDataset 不易套 sampler,風險高)。

## 小目標 + 輸入尺寸 fps 權衡 (實測數據,可當決策模板)

物件像素尺寸 √(w×h) 分桶 (tiny<16 / small16-32 / med32-96 / large>96)。
偵測頭 stride 決定可分辨下限:P3 (stride8) ~16-24px,P2 (stride4) ~8px。

KL730 YOLOX-s 輸入尺寸 fps (evaluate 實測,全零 CPU fallback):

| 輸入 | fps | 適用 |
|---|---|---|
| 640 | 43.2 | 最快,漏 <16px 小目標 |
| 896 | 21.3 | **平衡點**,小目標進 P3 可靠區 |
| 1280 | 10.1 | 小目標最佳但跌破即時 |
| 640+P2 head | ~25-30 (估) | 最佳 CP:不放大全圖、只補小目標頭 |

fps 與像素數約成反比。取捨原則:先定**偵測距離需求** (幾公尺外要認出目標) → 反推目標像素 → 選尺寸/P2。
若小目標是硬需求,896 或 640+P2 是首選;1280 留給「慢但要看清」。

## 成本 / 重量參考 (機載熱影像偵測)

- 熱像儀:FLIR Boson 640 (640×512, 機身 7.5g, ~0.5W, ~US$3,558);低成本驗證線 Lepton 3.5 (~US$164)。
- 酬載估 170–290g / 5–8W (2–5kg 級多旋翼餘裕足)。原型 Boson 線 ~US$4,100/套。
- **出口管制**:非製冷熱像儀 ≥9Hz 落 EAR 6A003.b.4.b,出口需許可 — 規劃時要查目的地。
