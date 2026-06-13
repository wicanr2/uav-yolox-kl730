---
name: kneron-edge-ai-integration
description: 把物件偵測模型 (YOLO/YOLOX) 整合部署到耐能 Kneron edge AI NPU (KL730/KL720/KL630) 的完整 SOP 與必踩雷清單。涵蓋:kneron-mmdetection docker 環境建置 (版本鎖定 + 6 個 build 坑)、資料集轉 COCO、ONNX 匯出 (opset 11)、Kneron toolchain 量化編譯成 NEF、decode+NMS 後處理、逐 channel 量化驗收。**核心價值 = 一個會讓人 debug 一整天的量化崩潰根因**:`--skip-postprocess` 的 raw-logit 輸出必須用 `mmse` range method,不能用 detection 預設的 `percentage=1.0`(會把量化範圍撐爆→輸出崩潰成 per-channel 常數)。觸發條件:使用者提到「把 YOLO/模型跑在耐能/Kneron NPU 上」「KL730/KL720 部署」「ONNX 轉 NEF」「Kneron toolchain 量化」「NPU 量化後輸出全 0/崩潰/退化」「kneron-mmdetection 環境」「edge AI 晶片整合物件偵測」「熱影像/影像辨識上邊緣 AI 晶片」「量化掉精度怎麼救」。也適用於規劃「算法+硬體系統整合」型專案 (非算法研發)。
---

# Kneron Edge AI 整合部署 SOP

把**現成**物件偵測算法 (YOLOX) 整合到**現成** Kneron NPU 晶片,讓它在 edge (如無人機) 即時跑。
這是**系統整合 (systems integration)**,不是算法研發 — 不發明新網路,而是讓「算法 + AI 晶片 + 資料 + 部署」協同運作。

> 本 skill 階層式:先讀本檔 (總覽 + 致命雷),需要某階段細節再讀 `references/` 對應檔。
> 來源:UAV2UAV 熱影像辨識專案實戰 (2026-06, KL730 + YOLOX-s),repo: github.com/wicanr2/uav-yolox-kl730

## 何時用此 Skill

- 要把 YOLO / 物件偵測模型部署到 Kneron KL730 / KL720 / KL630
- ONNX → NEF 轉換、Kneron toolchain 量化
- **NPU 量化後輸出崩潰 / 全 0 / 精度暴跌** ← 本 skill 最高價值,直接跳 §3
- kneron-mmdetection 訓練/匯出環境建不起來
- 規劃「邊緣 AI 影像辨識」系統整合專案 (晶片選型、軟體鏈、成本重量)

## 0. 心智模型:兩個世界 + 一座橋

```
【訓練世界 PC/GPU, PyTorch】          橋        【部署世界 晶片, C/SDK】
資料 → 訓練 → checkpoint → ONNX ──[Kneron toolchain]──→ NEF → NPU 即時跑
                          (opset11)  優化→量化→編譯
```
- **NEF** = Kneron NPU 的可執行檔 (等於「編譯好的程式」)。
- NPU 只算 raw 張量;**decode + NMS 跑在晶片的 CPU 核 (如 KL730 的 A55)**,不是 NPU。
- 七站完整流程見 `references/03-pipeline-playbook.md`。

## 1. 致命雷 #1 — 量化崩潰 (最該先知道的)

**症狀**:NEF 模擬推論輸出退化成「每個 channel 一個常數值」,float 範圍 −15~+7 被壓成 ±0.01。

**根因**:`--skip-postprocess` 匯出的是 **raw logits** (無 sigmoid,範圍 −15~+1)。
量化用 detection 官方建議的 `percentage=1.0` 會把範圍張到含 −15 離群值 →
INT8 的 256 格攤在 ~16 寬範圍 → 多數值擠進同一格 → 崩潰。

**解法 (一個參數)**:`datapath_range_method="mmse"` (SNR-based,對離群值穩健)。
改這一個就好,**不必動位寬、不必重 export、不必 mix 模式 (mix 模式吃爆記憶體)**。

詳細 7 輪排查、per-channel 驗收法、其他量化參數 → `references/02-quantization-gotchas.md`

## 2. 致命雷 #2 — 把開發機記憶體用爆

Kneron toolchain 的 **mix-mode 量化做 int16 模擬,記憶體爆量**;曾把 30GB RAM 機器拖進
swap thrash 假死 (需重開機,kernel OOM killer 不會觸發)。

**對策 (一律照做)**:
```bash
docker run --rm --cpus=8 --memory=20g --memory-swap=20g ...   # 硬上限,爆了只 kill 容器不拖垮主機
```
量化降載:`threads=4`、校正集 100 張、優先 int8 (非 mix)。

## 3. 致命雷 #3 — kneron-mmdetection 環境 (舊生態 + 新工具)

kneron-mmdetection 是 **mmdetection 2.25.0 fork (2022)**,配新版套件會炸。版本鎖死:
| 套件 | 鎖定 | 不鎖會怎樣 |
|---|---|---|
| PyTorch | 1.12.1 | — |
| mmcv-full | 1.6.0 | mmdet assert 失敗 |
| ONNX opset | 11 | export 腳本內 assert |
| yapf | 0.40.1 | 新版移除 FormatCode(verify=) → TypeError |
| onnx-simplifier | 0.4.36 | 0.5.x 的 onnxsim 在舊 py 無 wheel |

用 uv venv 還有額外 3 個坑 (no-build-isolation / setuptools / UV_PYTHON_INSTALL_DIR)。
完整 Dockerfile + 6 個 build 坑 → `references/01-docker-toolchain-env.md`

## 4. 關鍵決策原則 (選型,非研發)

- **YOLOX 優於新版 YOLO**:授權乾淨 (Apache-2.0 vs Ultralytics AGPL);Kneron 官方驗證;
  **新版 NMS-free (YOLOv10/26) 的 TopK 子圖在各家 NPU 量化都踩雷**。「越新越好」是迷思 —
  正確標準是「能在目標 NPU 正確量化的最新版」。
- **platform id 是字串 `"730"`**;toolchain conda env 用 `onnx1.13` (base 不支援 730)。
- **驗收要有鑑別力**:① 量化後逐 channel corr (非攤平,>0.9 合格) ② `cpu_node: N/A` (零 fallback)
  ③ mAP 掉幅 <3%。「非全 0」這種檢查太弱,抓不到退化。
- 詳細模型/資料/輸入尺寸 fps 權衡 → `references/04-model-data-choices.md`

## 5. 標準工作流 (七站)

```
1.資料蒐集 → 2.轉COCO → 3.訓練(YOLOX) → 4.匯出ONNX(opset11,skip-postprocess)
→ 5.量化編譯(toolchain: 優化→evaluate→mmse量化→compile→NEF) → 6.後處理(decode+NMS,跑CPU核)
→ 7.部署(Kneron PLUS 載 NEF 上板)
```
每站都要有可驗證的 pass/fail 訊號 (smoke / corr / fps / mAP / 端到端出框)。
逐站指令與 make target → `references/03-pipeline-playbook.md`

## 6. 無 GPU 也能做到的事 (CPU-only 開發策略)

本機只有 CPU 時:資料管線、smoke 訓練、**整條 ONNX→NEF 量化鏈 (toolchain 本就 CPU)、端到端模擬推論**
全部可在本機驗證;**只有「正式訓練」需要 GPU** → 丟雲端 (Kaggle 免費 30h/週)。
用官方 Model Zoo 預訓練權重,甚至不訓練就能先把量化鏈與最大風險驗完。

## 方法論備註 (digital twin)

- 棘手問題先建「快速、決定性的 pass/fail 訊號」,再用對照組逐一排除假設 (一次只變一個變因)。
- silent failure 最危險 (knerex 忽略參數不報錯、報告與實際設定脫鉤) → 主動驗證「設定真的生效了嗎」。
- 知識交付分層 (問題→概念→技術鏈→深入→實戰),讓非專家也能漸進理解。
- 大記憶體 docker 工作一律設 `--memory`/`--cpus` 上限,出事前就要設。
