# YOLOX 原理解說 — 從物件偵測基礎到 KL730 部署

> 目的:理解本專案用的模型「為什麼長這樣、訓練時發生什麼事、推論時每個數字哪來的」,
> 而不是只會下指令。從基礎一路講到 Kneron 版的修改。
> 論文:Ge et al., *YOLOX: Exceeding YOLO Series in 2021* (arXiv:2107.08430)

---

## 1. 物件偵測在解什麼問題

輸入一張影像,輸出一組 `(bounding box, 類別, 信心分數)`。兩個基本工具:

- **IoU (Intersection over Union)**:兩個框的交集面積 ÷ 聯集面積,衡量「框得準不準」。預測框與真值框 IoU ≥ 0.5 通常算命中。
- **mAP (mean Average Precision)**:對每個類別,掃過所有信心閾值畫 precision-recall 曲線取面積 (AP),再對類別取平均。COCO 慣例的 mAP 是 IoU 0.5~0.95 十個門檻的平均,比單一 IoU 0.5 嚴格得多。

偵測器分兩派:
- **Two-stage** (Faster R-CNN):先猜「哪裡可能有東西」(region proposal),再對每個候選區域精細分類。準,但慢。
- **One-stage** (YOLO 系):把影像切成密集網格,**每個網格位置直接同時預測「這裡有沒有東西、是什麼、框多大」**,一次前向傳播全部出來。這就是 "You Only Look Once" 的意思。edge 部署幾乎都選這派。

## 2. 密集預測的幾何:grid、stride、特徵金字塔

CNN backbone 逐層下採樣。YOLOX 從中取三個尺度的 feature map(neck 為 PAN 結構,把深層語意與淺層細節雙向融合後輸出):

| 層級 | stride | 640×640 輸入時的 grid | 每格負責的感受區域 | 擅長 |
|---|---|---|---|---|
| P3 | 8 | 80×80 = 6,400 格 | 細 | **小目標** |
| P4 | 16 | 40×40 = 1,600 格 | 中 | 中目標 |
| P5 | 32 | 20×20 = 400 格 | 粗 | 大目標 |

三個尺度合計 **8,400 個預測位置**。每個位置吐出一組向量,推論時 8,400 組全部解碼再篩選。

> 對本案的直接意義:P3 的 stride 8 決定了「最小可分辨目標」的數量級。熱影像裡遠距無人機常 < 32 px,
> 落在 P3 的負責範圍邊緣 — 這就是 PLAN 中「P2 head (stride 4) 或拉高輸入解析度」兩個小目標選項的由來。

## 3. Anchor:YOLOv3/v5 的做法,以及 YOLOX 為什麼丟掉它

**Anchor-based**(v2~v5):每個 grid 位置預先擺 k 個固定長寬比的「錨框」(例如 3 個),網路學的是「相對錨框的修正量」。錨框尺寸是對訓練集 bbox 跑 k-means 聚類得來的。

問題:
1. 錨框尺寸是 **dataset-specific 超參數** — 換到熱影像小目標域,COCO 聚出來的錨框不適用,要重聚、重調。
2. 每格 k 個錨框 → 預測數量 ×k(8,400 → 25,200),head 計算量與後處理負擔都變大。
3. 訓練時「哪個錨框配哪個真值」的 assignment 規則一堆 if-else,工程複雜。

**YOLOX 的 anchor-free**:每個 grid 位置只出 **1 組預測**,直接回歸:

```
預測向量 (每個位置):
  reg: (dx, dy, dw, dh)   4 維
  obj: objectness          1 維 — 「這裡有沒有物體」
  cls: 類別分數            C 維 — 本案 C=4 (uav/person/vehicle/bicycle)

解碼 (grid 座標 (gx, gy)、該層 stride s):
  中心 x = (gx + dx) * s        ← dx, dy 是相對該格左上角的偏移
  中心 y = (gy + dy) * s
  寬   w = exp(dw) * s          ← 寬高走 log 空間,保證恆正、且大小目標的
  高   h = exp(dh) * s             回歸誤差尺度相當
  分數  = sigmoid(obj) * sigmoid(cls)
```

沒有錨框 → 沒有錨框超參數、預測數砍到 1/3、assignment 邏輯也能換成更聰明的 (見 §5)。

## 4. Decoupled head:分類與回歸分家

YOLOv3~v5 的 head 是「耦合」的:一條 1×1 conv 同時吐出 cls+reg+obj。但分類要的特徵(紋理、語意)和回歸要的特徵(邊緣、幾何)本質衝突,共用一條路會互相干擾。

YOLOX 把 head 拆開:

```
          feature (P3/P4/P5 各一份, 先 1×1 conv 降到 256ch)
                      │
        ┌─────────────┴─────────────┐
   cls 分支 (2× 3×3 conv)      reg 分支 (2× 3×3 conv)
        │                      ┌─────┴─────┐
     cls (C 維)             reg (4 維)   obj (1 維)
```

論文消融:單是換 decoupled head,收斂速度大幅加快,AP +1.1。代價是 head 參數略增 — 這也是後面 Kneron 轉換時「**輸出不要 concat**」trick 的結構背景:三個尺度 × 兩三條分支,本來就是多個獨立輸出張量,保持分開最適合 NPU。

## 5. SimOTA:訓練時「誰該負責這個真值框」

訓練密集偵測器的核心問題:8,400 個預測位置,**哪些算正樣本(該對某個真值框負責)、哪些算負樣本?** 這叫 label assignment,對最終精度的影響常大於網路結構本身。

YOLOX 的答案是 **SimOTA**(把 assignment 視為最佳傳輸問題的簡化版):

1. **Center prior 預篩**:只有「落在真值框內、或落在真值框中心 5×5×stride 區域內」的位置才是候選 — 先把 8,400 砍到幾十個。
2. **算 cost**:每個 (候選位置 i, 真值框 j) 配對算成本
   `cost(i,j) = L_cls(i,j) + λ · L_IoU(i,j)`(λ=3),意思是「如果讓 i 負責 j,分類 + 回歸會多痛」。
3. **Dynamic-k**:每個真值框 j 該分幾個正樣本?取「與 j IoU 最高的前 10 個預測的 IoU 總和」四捨五入為 k_j。直觀:**框越大越清楚 → 高 IoU 候選越多 → 分越多正樣本;小目標天然只分到少數幾個**。
4. 對每個 j 取 cost 最低的 k_j 個位置當正樣本;一個位置若被多個真值搶,給 cost 最低者。

關鍵性質:
- **動態**:assignment 隨訓練進行自動變化,網路學得越好,分配越準(對比 anchor IoU 閾值的死規則)。
- **Multi-positives**:一個真值框有多個正樣本,梯度訊號比「一格一真值」的 YOLOv3 時代豐富得多。
- 論文消融:SimOTA 單項 AP +2.3,是 YOLOX 所有改動中最大的一筆。

> 對本案的意義:dynamic-k 對小目標只分配很少正樣本 → 小目標訊號天然弱。
> 這是熱影像 UAV 偵測要靠「資料量(Anti-UAV410 抽幀擴充)+ 解析度/P2」補強的理論原因。

## 6. Loss 函數

```
total = L_cls + L_obj + λ_iou · L_iou  (+ L_1, 只在最後 15 epochs)
```

- **L_cls**:BCE (binary cross-entropy),只算正樣本。注意 YOLOX 不用 softmax — 每類獨立 sigmoid,類別間不互斥。
- **L_obj**:BCE,全部 8,400 個位置都算(這是負樣本唯一的訊號來源)。
- **L_iou**:`1 - IoU²`,只算正樣本,直接最佳化框的重合度。
- **L_1**:最後 15 epochs 加上 L1 回歸 loss — 因為此時 Mosaic/MixUp 已關(見 §7),分布變乾淨,L1 做精修。

## 7. 訓練增強:Mosaic + MixUp

- **Mosaic**:4 張圖拼成 1 張(隨機縮放、裁切)。一個 batch 等效看到 4 倍場景,且目標尺度被人為打散 — 小目標樣本量大增。
- **MixUp**:兩張圖加權疊加。
- **最後 15 epochs 全關**:增強圖分布失真(拼接邊界、半透明鬼影),收尾要讓網路看真實分布。`YOLOXModeSwitchHook` 就是做這個切換,同時 L1 loss 開啟。

YOLOX 因為增強夠強,**不需要 ImageNet 預訓練**,從零訓練即可(我們仍用 COCO 預訓練權重起步,收斂更快)。

> 熱影像注意:Mosaic 的拼接對熱影像合法(灰階分布不變),但 MixUp 的半透明疊加在熱輻射語意上
> 不太自然 — 若 val 表現異常,關 MixUp 是第一個可以試的 ablation。

## 8. 推論流程:YOLOX 仍需要 NMS

```
8,400 × (4+1+C) 原始輸出
  → 解碼 (式見 §3)
  → score = sigmoid(obj)·sigmoid(cls) 過閾值 (如 0.25)
  → NMS: 同類別中,IoU > 閾值 (如 0.45) 的框只留分數最高者
  → 最終偵測結果
```

NMS (非極大值抑制) 必要的原因:multi-positives 訓練 → 一個物體周圍多個位置都會開火,要去重。
這正是部署的分工點:**8,400 維張量計算在 NPU,解碼 + NMS 在 KL730 的 Cortex-A55 (CPU) 上做**。
(YOLOv10/YOLO26 的 "NMS-free" 就是想把這步也訓練掉,但其 TopK 子圖在各家 NPU 量化都踩雷 — PLAN 已記錄,本案不採用。)

## 9. 模型家族:怎麼縮放出 nano~x

整個家族同一套結構,只調兩個係數:

| 型號 | deepen (層數倍率) | widen (通道倍率) | 參數量 | 備註 |
|---|---|---|---|---|
| nano | 0.33 | 0.25 | 0.9M | depthwise conv |
| tiny | 0.33 | 0.375 | 5.1M | 預設 416 輸入 |
| **s (本案)** | **0.33** | **0.5** | **9.0M** | Kneron 唯一 verified |
| m | 0.67 | 0.75 | 25.3M | |
| l / x | 1.0 / 1.33 | 1.0 / 1.25 | 54M / 99M | edge 不考慮 |

本案選 **s**:Kneron Model Zoo 唯一官方驗證過的 YOLOX (COCO box AP 40.5);若實機 fps 不足,降 tiny 是一行 config 的事 (`widen_factor=0.375`)。

## 10. Kneron 版 YOLOX 與上游的差異 (kneron-mmdetection fork)

| 項目 | 上游 YOLOX | Kneron 版 (`yolox_s_8x8_300e_coco_img_norm.py`) | 原因 |
|---|---|---|---|
| Activation | SiLU (x·sigmoid(x)) | **LeakyReLU(0.1)** | SiLU 在 NPU 上要拆成乘法+sigmoid,量化誤差大;LeakyReLU 是硬體原生支援 |
| 正規化 | x/255 縮放 | **mean=128, std=256 → x/256−0.5** | 對齊 NPU 定點運算的 2 的冪次;**量化校正集必須用同一條式子** |
| Export | torch.onnx 任意 opset | `pytorch2onnx_kneron.py`,**鎖 opset 11**、`--skip-postprocess` | toolchain 相容性;解碼/NMS 不進 ONNX,留給 CPU |
| 輸出結構 | 可能 concat 成單張量 | **各尺度、各分支分開輸出,不 concat** | 官方 trick:concat 會讓不同數值範圍的張量共用量化參數,精度崩 |
| sigmoid/exp | 後處理做 | **保留在模型內** | 讓 NPU 用查表做非線性,CPU 只剩純幾何解碼 |

### 端到端部署鏈 (本 repo 的 make 目標對應)

```
訓練 (kneron-mmdetection docker, GPU)          make train
  → checkpoint (.pth)
ONNX export (opset 11, skip-postprocess)       make export-onnx
  → yolox_s_uav2uav.onnx
Kneron toolchain docker (conda env onnx1.13):  make nef
  kneronnxopt 優化 → ktc.ModelConfig(..., "730") → evaluate (模擬 fps / CPU node 檢查)
  → analysis (PTQ 量化, 校正集 = val 影像 × (x/256−0.5)) → BIE
  → ktc.compile → NEF
  → kneron_inference sanity check (防量化後全 0)
KL730 板 (Kneron PLUS C/Python API)
  NPU 跑 NEF → A55 解碼 + NMS → tracker → MAVLink/RTSP
```

## 11. 一個 batch 的完整故事 (把以上串起來)

以本案 config (`samples_per_gpu=8`) 的某一步為例:

1. Dataloader 取 8 張熱影像;每張先經 **Mosaic**(再可能 MixUp),resize 到 640×640,套 `x/256−0.5`。
2. 前向:backbone (CSPDarknet, LeakyReLU) → PAN neck → decoupled head,吐出 P3/P4/P5 共 8,400 × (4+1+4) 的預測。
3. **SimOTA**:對每張圖的每個真值框做 center 預篩 → cost 矩陣 → dynamic-k 分配正樣本。一個 640 畫面裡 10 px 的無人機,大概只會分到 1~2 個正樣本;一台 100 px 的車可能分到 7~8 個。
4. 算 loss:正樣本算 cls + IoU loss,全位置算 obj loss;反向傳播,SGD (lr=0.00125,batch 64→8 線性縮放) 更新。
5. 第 86 epoch 起 (max_epochs=100 − 15):Mosaic/MixUp 關閉、L1 loss 開啟,進入精修期。
6. 每 5 epochs 在 val split 上算 COCO mAP — 這個數字之後會和 INT8 量化版比,**掉幅 < 3% 才算量化合格** (PLAN §3.3)。

---

## 延伸閱讀

- YOLOX 論文:https://arxiv.org/abs/2107.08430 (重點讀 §2.1 的逐項消融表)
- OTA (SimOTA 的完整版前身):https://arxiv.org/abs/2103.14259
- kneron-mmdetection YOLOX 教學:https://github.com/kneron/kneron-mmdetection/blob/main/docs_kneron/yolox_step_by_step.md
- Kneron toolchain YOLO 部署 trick:https://doc.kneron.com/docs/toolchain/appendix/yolo_example/
- FPN (特徵金字塔):https://arxiv.org/abs/1612.03144
