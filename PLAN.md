# PLAN — 無人機機載熱影像辨識系統 (UAV2UAV)

> 目標:在無人機上掛載紅外線熱像儀 + Kneron KL730 edge AI 晶片,機上即時辨識「無人機 / 車輛 / 人」。
> 本文件為第一版規劃,所有事實均經 web 查證並附來源;標記「未查證」「估算」者需後續確認。
> 產出日期:2026-06-11

---

## 0. 結論摘要 (先讀這段)

| 決策點 | 建議 | 理由 |
|---|---|---|
| 晶片 | **KL730 (KNEO Pi V2 SBC 形態)** | Quad A55 可獨立跑 Linux 免 host;YOLOv5s 640 模擬 51.8 fps;MIPI 直入 + ISP |
| 模型 | **首選 YOLOX-s/tiny (Apache-2.0)**,備選 YOLOv5s (官方全流程但 AGPL) | Kneron 官方 mmdetection 流程已驗證 YOLOX;Ultralytics 系列 (v8/v11/26) 為 AGPL 且 KL730 量化有未解 bug |
| 熱像儀 | **FLIR Boson 640 (60Hz)**,低成本驗證線用 Lepton 3.5 | 640×512 與三個資料集天然對齊;7.5g 機身;USB/CMOS 介面風險最低 |
| 任務拆分 | **兩個偵測任務分開評估**:空對空偵測 UAV、空對地偵測車/人 | 視角 domain gap 過大,混訓需謹慎;部署時再評估單模型 4 類 vs 雙模型切換 |
| 酬載 | 估算 **170–290 g / 5–8 W**,2–5kg 級多旋翼餘裕充足 | 詳見 §6 |
| 原型成本 | Boson 線 ~**USD 4,100–4,400**/套;Lepton 驗證線 ~**USD 700–1,000**/套 | 詳見 §7 |
| 最大風險 | ① 新版 YOLO 在 KL730 量化後輸出異常 (論壇已有 v11 全 0 案例);② FLIR ADAS 資料集商用授權未確認;③ ≥9Hz 熱像儀出口管制 (EAR 6A003.b.4.b) | 詳見 §10 |

---

## 0.1 決策記錄 (2026-06-11, 使用者定案)

1. **模型只用 YOLOX** (Apache-2.0, Kneron 官方流程) — 不再保留 YOLOv5 備選,新版 YOLO PoC 支線取消。
2. **FLIR ADAS 不採用** (bypass) — 授權風險 R2 關閉;車/人偵測改以 HIT-UAV 為主力。
3. **資料集定案**:ThermalUAV2UAV (空對空 UAV) + HIT-UAV (空對地車/人) + Anti-UAV410 (抽幀擴充 UAV)。
4. 本案性質為**技術評估/軟體環境建置**,非採購案;成本以網路公開資料估算即可。
5. 類別定案 4 類:`uav` / `person` / `vehicle` (car+other vehicle) / `bicycle`。
6. **開發機為 CPU-only**:本機 (docker, 14 核/30GB) 負責資料管線、smoke 訓練、ONNX→NEF 轉換驗證;正式訓練走雲端 GPU (Kaggle/Colab)。官方 Model Zoo 預訓練 YOLOX-s 可在無 GPU 情況下先驗證整條 KL730 轉換鏈 (`make zoo-export && make nef-zoo`)。

> 以下章節中與 FLIR ADAS / YOLOv5 / YOLO26 PoC 相關內容視為已被本節取代。

---

## 1. YOLO 演算法解釋

### 1.1 核心概念

YOLO (You Only Look Once) 是 single-stage 物件偵測:整張影像一次前向傳播,直接回歸出所有 bounding box + 類別機率,不像 two-stage (Faster R-CNN) 先產生候選區域再分類。因此速度快、適合 edge 即時推論。

組成三段:
- **Backbone**:卷積特徵抽取 (CSP/ELAN 系)。
- **Neck**:多尺度特徵融合 (FPN/PAN),輸出 P3/P4/P5 (stride 8/16/32) 三個尺度的 feature map,分別負責小/中/大目標。
- **Head**:每個尺度上預測 box 座標 + objectness + 類別分數;推論後以 NMS (非極大值抑制) 去除重複框。

### 1.2 版本演進與授權 (商用部署關鍵)

| 版本 | 關鍵差異 | 授權 | 對本案意義 |
|---|---|---|---|
| YOLOv5 (2020) | anchor-based, CSP | **AGPL-3.0** | Kneron 官方訓練+部署全流程支援,但商用需買 Ultralytics 企業授權 |
| YOLOX (2021, Megvii) | anchor-free, decoupled head | **Apache-2.0** ✅ | Kneron kneron-mmdetection 官方 step-by-step;授權乾淨,**本案首選** |
| YOLOv8 (2023) | anchor-free, C2f | AGPL-3.0 | 無官方 KL730 範例,需自行改 operator |
| YOLOv9 (2024) | PGI + GELAN | GPL-3.0 (無商業授權管道) | 不建議 |
| YOLOv10 (2024) | **NMS-free** (dual assignments) | AGPL-3.0 | export graph 含 TopK,各家 NPU (OpenVINO/EdgeTPU/TensorRT) 量化均踩雷;Kneron 成功案例未查證 |
| YOLO11 (2024) | C3k2 | AGPL-3.0 | 論壇實測:模擬正確、**量化編 NEF 後輸出全 0,未解** |
| YOLO26 (2026/01) | edge-first、移除 DFL、原生 NMS-free、ProgLoss+STAL 強化小目標 | AGPL-3.0 | 架構方向與 edge 部署契合,但 Kneron 轉換案例未查證,只能列 PoC 候選 |

「版本能動越新越好」的務實解讀:**新版本的價值要以「能在 KL730 上正確量化執行」為前提**。目前證據顯示:
- 低風險:YOLOv5s、YOLOX (官方驗證)。
- 中風險:YOLOv8/v11 (官方說 operator 改造後可跑,但有量化 bug 公開未解)。
- 高風險:YOLOv10/YOLO26 (NMS-free head 的 TopK 子圖是跨平台地雷)。

→ 策略:主線用 YOLOX,平行排一條 **YOLO26-PoC 支線**(砍掉 end-to-end 後處理子圖、CPU 端 decode),PoC 通過才升級。

### 1.3 熱影像特殊處理

- **單通道輸入**:訓練框架預設吃 3-channel;主流做法是 gray8 複製成 3ch,沿用 COCO 預訓練權重。KL730 toolchain 對 input_fmt 有硬體限制,3ch 複製也是 NPU 端最不易踩雷的路徑。16-bit raw 需先 AGC 壓到 8-bit。
- **小目標 (無人機常 <32px)**:
  1. **P2 head** (stride-4):可覆蓋 ~4px 目標,代價是 feature map 變大、NPU latency 上升。
  2. **提高輸入解析度** (640→896/1280):latency 約與像素數成正比 (推估)。
  3. **Tiling/SAHI**:VisDrone 上 AP +6.8~14.5%,但一幀變多次推論,fps 預算要先算。
  - 取捨順序:先試 640 baseline → 不夠再「896 輸入」或「P2 head」二擇一 → tiling 為最後手段。

---

## 2. 晶片選擇考量 (為什麼 KL730)

| | **KL730** | KL720 | KL630 |
|---|---|---|---|
| 算力 | 3.6 eTOPS INT8 / 7.2 INT4 | 1.4 TOPS | 0.5 eTOPS INT8 |
| CPU | **Quad Cortex-A55 (獨立跑 Linux)** | Cortex-M4 (companion,需 host) | Cortex-A5 |
| YOLOv5s 640 模擬 fps | **51.8** | 25.4 | 16.8 |
| 影像 | 8MP@90fps 輸入、4K60、進階 ISP、MIPI 直入 | 4K | 5M@30fps |
| 功耗 | 板級跑 YOLOv5 ~2W (官方宣稱);晶片級瓦數**未查證** | 平均 1.2W | — |

選 KL730 的理由:
1. **獨立運作**:A55 跑 Linux,不需再帶 RPi/CM4 host → 省重量、功耗、整合複雜度。
2. **吞吐量**:YOLO 類約為 KL720 的 2 倍,留有 P2 head / 高解析輸入的餘裕。
3. **MIPI CSI 直入 + 內建 ISP**:熱像儀可直接接 (官方 BSP 已有 IMX678 MIPI 驅動先例)。
4. 唯一支援 Transformer、toolchain 重點維護的世代;auto-grade 定位。

代價與風險:
- KL730 **無 dongle / M.2 形態**,只有 KNEO Pi SBC 與 96Boards HDK 兩種板。
- KNEO Pi 板級工作溫度 **0–50°C (消費級)**;高空/冬季需向 Kneron 確認工業溫度版本 (未查證,列入風險)。
- 板卡售價與重量官方未公開 (未查證)。

可取得形態:
| 形態 | 型號 | 重點 |
|---|---|---|
| KNEO Pi V2 SBC | KP73B703A-M1 | RPi 尺寸、2GB LPDDR4、40-pin GPIO、RJ45、USB3、USB-C 5V/2A、0–50°C |
| 96Boards HDK | KP73B753A-M2 | 85.6×56.5mm、8 層板、Ethernet、MIPI TX |

---

## 3. 資料集:如何 train & test

### 3.1 三個資料集的事實

| | FLIR ADAS v2 | ThermalUAV2UAV | HIT-UAV (建議補充) |
|---|---|---|---|
| 規模 | 26,442 張標註 (~520k box) | **3,856 張** (train 2,514/val 866/test 476) | 2,898 張 (24,899 box) |
| 解析度 | 640×512 (Tau 2) | 640×512 (Zenmuse XT S, 8-bit PNG) | 640×512 |
| 視角 | **地面車載前視** | **空對空** (M210 RTK 拍 UAV) | **空對地** (飛高 60–130m) |
| 類別 | 15 類 (person/car/bike…) | 單類 UAV (4 四旋翼+2 六旋翼) | Person/Car/Bicycle/OtherVehicle/DontCare |
| 標註格式 | COCO JSON (需轉 YOLO) | YOLO 原生 | XML+JSON (含 oriented bbox) |
| 授權 | 需註冊;**商用權未查證 (授權頁 404)** | **MIT** ✅ | **CC BY 4.0** ✅ |
| 已知缺陷 | 類別極不平衡 (car 73k vs dog 個位數) | 規模小、單感測器、6 機型 | 規模中等 |

補充資料集 (無人機偵測擴充):**Anti-UAV410** — 410 段 TIR 影片、>438K 標註框、1080p、地對空,repo 標 MIT;抽幀可擴充 tiny target 與雜訊背景。

### 3.2 資料集 → 任務對應

| 任務 | 主力 | 補充 | 注意 |
|---|---|---|---|
| 空對空偵測 UAV (核心) | ThermalUAV2UAV | Anti-UAV410 抽幀 | TU2U <4k 張單獨訓練易 overfit;Anti-UAV 1080p 縮到 640 後 UAV 可能 <8px,需 tile/crop 不可整幀 resize |
| 空對地偵測車/人 | HIT-UAV (視角正確、授權乾淨) | FLIR ADAS (量大,當 pretrain/augment) | FLIR 是地面前視,**不可直接當空對地 test 依據** |

### 3.3 訓練策略

1. **類別收斂**:最終 4 類 — `uav` / `person` / `vehicle` (car+truck+bus+other) / `bicycle`(可選)。FLIR 15 類映射過去,長尾類別丟棄。
2. **前處理統一**:全部轉單通道 8-bit → 複製 3ch → 相同 normalize;FLIR 16-bit raw 若自做 AGC,pipeline 要套用到所有資料集。
3. **訓練流程**:
   - Stage 1:COCO 預訓練權重 → FLIR ADAS fine-tune (學熱影像 domain 的 person/vehicle 特徵)。
   - Stage 2:混合 HIT-UAV + TU2U + Anti-UAV 抽幀,訓最終 4 類模型;以 sampling 權重平衡各來源。
   - 平行對照組:UAV 單類模型 + 車/人模型分開訓,比較 4 類單模型 vs 雙模型的 mAP 與 NPU 成本。
4. **Test / 驗收**:各用自己的官方 test split (TU2U 476 張、HIT-UAV 579 張、Anti-UAV410 120 段);**不可跨視角混測**,數字無工程意義。
5. **量化驗收**:FP32 → INT8 (Kneron PTQ) 後 mAP 掉幅 <3% 為合格線;校正集從各資料集 train split 均勻抽 200–500 張。
6. **環境**:依全域規則,訓練與轉換全部在 docker 內 (Kneron toolchain 本身就是 docker image `kneron/toolchain`;訓練側自建 PyTorch + mmdetection docker, uv venv)。

---

## 4. 軟體架構規劃

### 4.1 機上 pipeline (KL730 board, Linux)

```
[熱像儀 Boson 640]
   │ USB-UVC (低風險) 或 CMOS/BT656→MIPI bridge (低延遲)
   ▼
[Capture service]  V4L2 取流, AGC/正規化, resize 640×640
   ▼
[Inference service]  Kneron PLUS C API → NPU (NEF 模型)
   │                 YOLO decode + NMS 在 A55 (CPU) 端
   ▼
[Tracker]  輕量 SORT/ByteTrack (CPU), 抑制單幀誤報, 給出 track id
   ▼
[Output mux]
   ├─ UART → 飛控 (MAVLink custom message: 類別/box/置信度/track id, 低頻寬)
   └─ RTSP (H.264, 帶 overlay) → 圖傳 → 地面站人工確認
```

模組邊界 (deep module 原則,介面窄、內部藏複雜度):
- `capture/`:對外只給 `get_frame() → 8-bit 640×512`,內部藏 AGC、FFC 事件處理、糊幀過濾。
- `detect/`:對外只給 `infer(frame) → [Detection]`,內部藏 NEF 載入、前處理 layout、decode/NMS。
- `track/`:`update([Detection]) → [Track]`。
- `report/`:`publish([Track])`,內部藏 MAVLink 封裝與 RTSP overlay。

### 4.2 訓練/轉換 pipeline (地面, 全 docker)

```
資料準備 docker ──> 訓練 docker (mmdetection/YOLOX) ──> torch.onnx.export (opset ≤18)
                                                              │
                kneron/toolchain docker:  kneronnxopt 優化 → IP evaluator (估 fps/抓 CPU node)
                  → PTQ 量化 (校正集) → BIE → batch compile → NEF
                                                              │
                E2E simulator 比對 FP32 vs INT8 輸出 ──> KL730 實機驗證
```

Kneron 官方 YOLO 部署 trick (文件明載):多 scale 輸出不要 concat、class score 與 bbox 分開輸出、sigmoid/exp 保留在模型內。

### 4.3 Repo 結構 (vertical slice)

```
uav2uav/
├── CONTEXT.md            # domain glossary
├── PLAN.md               # 本文件
├── datasets/             # 下載/轉換腳本 + 統一前處理 (不進 git 的原始資料)
├── training/             # YOLOX/YOLOv5 訓練 config + docker
├── conversion/           # ONNX export + Kneron toolchain 腳本 + 量化校正集
├── firmware/             # 機上服務 (capture/detect/track/report)
├── bench/                # fps/mAP/功耗 量測 harness
└── docs/adr/             # 重大決策記錄
```

---

## 5. 硬體整合建議 (基於 KL730)

### 5.1 熱像儀選型

| | Boson 640 (建議) | Lepton 3.5 (驗證線) | Hadron 640R | GuideIR COIN612 |
|---|---|---|---|---|
| 解析度/幀率 | 640×512 @60Hz (或 8.6Hz 出口版) | 160×120 @8.7Hz | 640×512 @60Hz + 64MP EO | 640×512 @25/30Hz |
| 介面 | CMOS/BT656、USB3;MIPI 標 optional/future | VoSPI (SPI, 非標準視訊) | **原生 MIPI** + USB3 | CMOS/LVDS/USB2 |
| 重量/功耗 | 機身 7.5g / ~0.5W | 0.9g / 0.14W | 56g / <1.8W | 11.5g / 0.8W |
| 價格 | $3,558 (GroupGets) | $164 | $3,992 | 未查證 (估 $1–2k) |
| 出口管制 | ≥9Hz 落 EAR 6A003.b.4.b 需許可 | 8.7Hz 多數國家免許可 | 同 Boson | 中國產品,需評估供應鏈信任 |

建議:
- **開發期**:Lepton 3.5 + KNEO Pi 先打通全 pipeline (~$700 全套),同步採購 Boson。
- **原型**:Boson 640 60Hz,接法以 **USB-UVC 進 KNEO Pi** 為第一版 (風險最低);MIPI bridge (BT656→MIPI) 為延遲優化選項。
- Boson 操作溫度 -40~80°C、耐衝擊 1500g@0.4ms,環境規格足夠;瓶頸反而是 KNEO Pi 板級 0–50°C。
- 160×120 的 Lepton 對遠距小目標解析度不足,只當 pipeline 驗證,不做偵測距離評估。

### 5.2 機構與環境

- **減震**:熱像儀 + 運算板掛減震球/減震板;軟體端以銳利度過濾糊幀。
- **熱管理**:NPU 廢熱不可傳導至熱像儀 (影響輻射量測,Boson datasheet §9.4/9.5 明載);艙內隔熱 + 獨立氣流路徑;非製冷感測器需週期 FFC 快門校正,FFC 瞬間會掉幀,tracker 要能跨 FFC 維持 track。
- **EMI**:DDR/MIPI 高速訊號屏蔽 (導電襯墊);GPS/羅盤天線遠離酬載與電源線;視訊線用屏蔽線。

### 5.3 飛控整合

- KL730 UART → 飛控 TELEM2 (PX4/ArduPilot 標準 companion 接法),MAVLink custom message 上報 `類別/box/置信度/track id`。
- Ethernet → RTSP H.264 (帶 overlay) 走圖傳,地面人工確認通道。

---

## 6. 重量預算 (估算)

| 分項 | 重量 (g) | 功耗 (W) |
|---|---|---|
| Boson 640 + 中焦鏡頭 | 30–60 | 0.5–1.0 |
| KL730 板 (KNEO Pi) | 45–60 (估算,RPi 尺寸級) | 4–6 (估算;NPU 推論 2W 已查證) |
| 結構/外殼/減震架 | 60–120 (估算) | — |
| 線材/DC-DC/連接器 | 30–50 (估算) | ~0.5 轉換損耗 |
| **合計** | **~170–290 g** | **~5–8 W** |

- 2–5kg 級多旋翼酬載餘裕通常 0.5–1.5kg → 本酬載佔比低,**續航影響估算 <3%**。
- Lepton 驗證線可壓到 ~120–180g;Hadron 線約 200–300g。

---

## 7. 成本估算 (估算, USD)

| 分項 | 原型 1 套 | 小批量 10 套/單套 |
|---|---|---|
| Boson 640 (60Hz) | $3,558 | ~$3,000–3,200 (估 -10~15%) |
| KL730 板 | $100–250 (未查證) | 同左 / SoM 量產價未查證 |
| 結構/減震/外殼 | $300–500 | $150–250 |
| 線材/電源 | $100 | $60 |
| **Boson 線合計** | **~$4,100–4,400** | **~$3,400–3,700** |
| **Lepton 驗證線合計** | **~$700–1,000** | **~$500–600** |

成本主導者是熱像儀 (~85%);若可接受中國模組 (COIN612/InfiRay) 估可砍半,但需評估客戶端供應鏈要求。

---

## 8. 如何佈署到無人機上 (步驟)

1. **模型凍結**:訓練收斂 → `torch.onnx.export` (opset ≤18)。
2. **轉換**:`kneron/toolchain` docker → kneronnxopt → IP evaluator (確認無 operator 掉到 CPU node、估 fps) → PTQ 量化 → NEF。
3. **模擬驗證**:E2E simulator 比對 FP32/INT8 逐層輸出 (防 v11 式「量化後全 0」事故) → mAP 回歸測試。
4. **實機 bench**:KNEO Pi 上 Kneron PLUS 跑 NEF,量實際 fps / 延遲 / 板溫 / 功耗。
5. **HIL 測試**:板上接 Boson 實流,桌面播放錄好的飛行熱影像對鏡頭/或直接灌檔案,驗證端到端延遲與 tracker 行為。
6. **掛機地面測試**:整酬載上機不起飛,驗證供電、EMI (GPS/羅盤健康度)、MAVLink 鏈路。
7. **飛行測試**:先空對地 (車/人) 場景,再雙機空對空場景;記錄 rosbag/log 供回歸。
8. **OTA/維護**:NEF 模型檔與服務以版本化部署 (SD 卡映像 + 簽章),飛前自檢 (模型 hash、相機 FFC、NPU self-test)。

---

## 9. 里程碑

| 階段 | 內容 | 出場條件 |
|---|---|---|
| M0 採購/環境 (2 週) | KNEO Pi + Lepton + Boson 採購;docker 環境;資料集下載註冊 | toolchain docker 跑通官方 YOLOv5s 範例 NEF |
| M1 Pipeline 打通 (3 週) | Lepton→KNEO Pi→官方模型→UART 輸出全鏈路 | 端到端 demo (任意模型) |
| M2 模型 v1 (4 週) | YOLOX 4 類訓練 + 量化 + 實機 bench | INT8 mAP 掉幅 <3%;640 輸入 ≥25 fps 實機 |
| M3 小目標優化 (4 週) | P2 head / 896 輸入對照實驗;tracker 整合 | TU2U test set 達標 (目標值 M2 後定) |
| M4 硬體整合 (4 週) | Boson 上板、機構、減震、EMI、掛機測試 | 掛機地面測試通過 |
| M5 飛測 (持續) | 空對地 → 空對空飛行驗證 | 飛行 log 回歸體系建立 |
| 支線 PoC | YOLO26/YOLOv8 → KL730 轉換可行性 | 量化後輸出正確即升級評估 |

---

## 10. 風險登記

| # | 風險 | 等級 | 對策 |
|---|---|---|---|
| R1 | 新版 YOLO 量化後輸出異常 (v11 全 0 案例公開未解) | ~~高~~ **已緩解** | 主線鎖 YOLOX;**2026-06-12 實證:YOLOX-s zoo 權重在 KL730 量化通過 (mmse range method, 9 輸出 corr 0.90–0.97)**。關鍵:raw-logit 輸出須用 `mmse` 而非 `percentage=1.0`,否則崩潰成常數。每次轉換跑逐 channel sanity check |
| R2 | FLIR ADAS 商用授權未確認 (授權頁 404) | 高 | 短期僅內部實驗;產品化前取得 Teledyne 書面確認;備案 HIT-UAV (CC BY 4.0) |
| R3 | Boson ≥9Hz 出口管制 (EAR 6A003.b.4.b) | 高 | 確認採購/部署地域;必要時 8.6Hz 版 + 軟體插幀評估;禁運地域不可部署 |
| R4 | KNEO Pi 0–50°C 消費級溫度範圍 | 中 | 向 Kneron 確認工業級版本;艙內熱設計;冬季/高空測試前實測板溫 |
| R5 | Ultralytics AGPL 傳染 (若用 v5/v8/11/26) | 中 | 主線 YOLOX (Apache-2.0);否則編列 Ultralytics 企業授權預算 |
| R6 | TU2U 資料量小 (<4k),空對空泛化不足 | 中 | Anti-UAV410 抽幀擴充;自錄飛行資料回灌;主動蒐集負樣本 (鳥/雲/地面熱源) |
| R7 | KL730 晶片級功耗/溫度/板卡價格未查證 | 中 | M0 採購階段直接向 Kneron/代理商索取 datasheet 與報價 |
| R8 | KL730 fps 為官方模擬值,實機含前後處理會更低 | 中 | M2 以實機 bench 為準;預留 2 倍餘裕 (模擬 51.8 → 目標 25 fps) |
| R9 | NPU 廢熱影響熱像儀輻射量測 | 低 | 機構隔熱;Boson datasheet thermal considerations 對策 |

---

## 11. 待確認清單 (進 M0 前)

- [ ] KL730 晶片級功耗 / 封裝 / 工業溫度版本 → 問 Kneron/代理商
- [ ] KNEO Pi V2 售價、重量、MIPI CSI 輸入 lane 數 → 問 Kneron/代理商
- [ ] FLIR ADAS dataset 商用授權條款 → 問 Teledyne FLIR
- [ ] Boson 640 採購地域的出口許可流程 → 問代理商
- [ ] ThermalUAV2UAV 影像 EXIF/拍攝距離分布 → 下載後實測統計
- [ ] YOLOv8/v11/26 → KL730 轉換 PoC (operator 改造 + 量化驗證)

---

## 附錄:研究來源

四個研究主題的完整來源清單 (官方文件、datasheet、論文、論壇) 已在研究階段逐項查證,關鍵來源:

- Kneron 官方文件中心:https://doc.kneron.com/ (toolchain / PLUS / model zoo / performance)
- KL730 產品頁:https://www.kneron.com/page/soc/
- KNEO Pi 文件:https://kneron.github.io/kneopi-documentation/
- YOLOX (Apache-2.0):https://github.com/Megvii-BaseDetection/YOLOX
- kneron-mmdetection YOLOX 流程:https://github.com/kneron/kneron-mmdetection/blob/main/docs_kneron/yolox_step_by_step.md
- KL730 上 YOLOv11 量化問題討論:https://www.kneron.com/forum/discussion/comment/2393#Comment_2393
- FLIR ADAS v2 README:https://adas-dataset-v2.flirconservator.com/dataset/README.txt
- ThermalUAV2UAV:https://github.com/GabryV00/ThermalUAV2UAV_Dataset (MIT;3,856 張已實際 clone 驗證)
- HIT-UAV:https://www.nature.com/articles/s41597-023-02066-6 (CC BY 4.0)
- Anti-UAV410:https://github.com/HwangBo94/Anti-UAV410
- Boson datasheet Rev 340:https://groupgets-files.s3.amazonaws.com/boson/documents/Boson%20datasheet,%20102-2013-40,%20Rev%20340.pdf
- 出口管制:https://www.federalregister.gov/documents/2024/02/23/2024-03661/
- PX4 companion computer:https://docs.px4.io/main/en/companion_computer/
- SAHI 小目標切片:https://arxiv.org/abs/2202.06934
- YOLO26:https://arxiv.org/abs/2606.03748
