# Kneron NPU 部署筆記:NEF 載入、Command Stream、Operator 解惑

> 整理自 2026-06-12 的三個問答:Kneron PLUS 如何讀取 NEF、NPU command 有哪些、operator 該怎麼理解。
> 搭配閱讀:`glossary.md`(名詞)、`yolox-explained.md`(模型原理)、`session-walkthrough.md`(全程記錄)。
> 內容依官方文件 + 本專案實際編譯產物;個別 API 名稱以實際安裝的 PLUS 版本為準。

---

## 1. Kneron PLUS 如何讀取 NEF 放到 NPU

### 1.1 角色分工:編譯期 vs 執行期

```
編譯期 (PC, toolchain docker)          執行期 (板上, Kneron PLUS)
ONNX → 量化 → ktc.compile() → NEF ──→  PLUS 載入 NEF → NPU 執行
```

- toolchain 的職責到「產出 NEF」為止。
- **Kneron PLUS** 是執行期 SDK(C/Python),跑在 host 上。host 可以是 PC(USB 接 dongle/板),而 **KL730 因為自帶 A55 跑 Linux,KNEO Pi 上 host 就是板子本身** — PLUS 直接在板上執行,不需外部電腦。

### 1.2 NEF 裡面裝了什麼

NEF 不只是權重,是自帶說明書的完整包:

| 內容 | 用途 |
|---|---|
| NPU command stream | 編譯器排好的指令序列(哪層先算、DMA 怎麼搬) |
| 量化後權重 (INT8/16) | 模型參數本體 |
| **ioinfo**(輸入/輸出描述) | 每個輸入輸出的 shape、**radix/scale**、硬體資料排列格式(如 `1W16C8B`) |
| model ID / version | 多模型管理用(toolchain `ModelConfig(20008, "0001", ...)` 給的就是這個) |

執行期不需要 ONNX、不需要知道網路結構 — NEF 已是「編譯好的程式」。

### 1.3 載入流程(Python `kp` 模組,C API 同名異形)

```python
import kp

# 1. 掃描並連線裝置 (USB 接 dongle/板;板上自跑時連自己)
device_group = kp.core.connect_devices(usb_port_ids=[port_id])

# 2. (KL520/720 需要) 載入 SCPU/NCPU firmware
#    KL730 跑自己的 Linux OS,此步驟不同/可省 — 以板上 BSP 為準
kp.core.load_firmware_from_file(device_group, scpu_fw, ncpu_fw)

# 3. 載入 NEF — 關鍵一步
model_desc = kp.core.load_model_from_file(device_group, "models_730.nef")
#    背後發生的事:
#    a. NEF 整包傳進裝置 DDR (USB 傳輸或板上直接讀檔)
#    b. 裝置 firmware 解析 NEF header,把 ioinfo/model ID 註冊進 model manager
#    c. 權重與 command stream 放到 NPU 可定址的記憶體區
#    d. 回傳 model descriptor: 各輸入輸出的 shape/格式/radix
```

### 1.4 每次推論的資料流

```python
# 4. 送影像 (host 端先 resize;正規化可交給裝置或自己做)
kp.inference.generic_image_inference_send(device_group, input_descriptor)

# 5. 收結果 — 拿回來的是「定點原始值 + radix/scale」
raw = kp.inference.generic_image_inference_receive(device_group)

# 6. 反量化成浮點 (PLUS 提供現成函式)
out = kp.inference.generic_inference_retrieve_float_node(
          node_idx=i, generic_raw_result=raw,
          channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW)

# 7. CPU 端後處理: YOLO decode + NMS (見 yolox-explained.md §8)
```

裝置內部:firmware 收影像 → DMA 搬進 NPU 工作記憶體 → NPU 照 command stream 逐層執行(權重走 GETW 通道,中間結果走 RDMA/WDMA)→ 輸出寫回 DDR → 回傳 host。

### 1.5 與本專案的三個呼應點

1. **radix/scale 在第 6 步收尾**:量化階段決定的 radix 就是反量化的依據 — 量化崩潰(radix 選錯)上板會原樣重現,這是為什麼值得在軟體端先驗收。
2. **send/receive 非同步成對**:可以 pipeline(送第 N+1 張時第 N 張還在算),是達到 43 fps 標稱值的必要寫法;單張同步輪詢會慢很多。
3. **KL730 hico mode**(PLUS v3.1.0+,Enterprise):MIPI 相機直入 NPU,不經 host 記憶體 — 未來接 Boson 的低延遲選項(PLAN §4.1 capture 模組)。

---

## 2. Kneron NPU command 有哪些

### 2.0 邊界

Kneron NPU 的**真正 ISA(二進位指令編碼)是專有的、官方不公開**。但編譯器的中間產物讓我們能看到「command node 層級」— 已足夠做效能分析與除錯。以下數字全部來自本專案 YOLOX-s 的實際編譯 log(`kneron_flow/batch_compile.log`)。

### 2.1 Command stream 怎麼生出來的

```
ONNX 197 節點
  → 前端圖優化 (FE)              197 → 294 節點 (展開硬體相關資訊)
  → IR lowering                   230 節點
  → FM cut 分析 (feature map 切割) → 361 個 cmd node
  → tiling 分析 (大層切小塊塞進片上記憶體) → +3,290 個 cmd node
  → MMU 記憶體配置: RDMA 94 / WDMA 85 / PFUNC 79 / Inplace 92
```

關鍵概念:**一個 ONNX operator ≠ 一個 command**。一層大 Conv 會被 tiling 切成幾十個 command(每塊 feature map 一個)— 9M 參數的 YOLOX-s 最後是 ~3,600 個 command。

### 2.2 Command node 的類型(evaluate 報告實際出現過的)

| 類型 | 例子(本專案模型) | 做什麼 |
|---|---|---|
| **融合計算節點** | `npu_fusion_node_Conv_41_LeakyRelu_42` | 主力:Conv+activation 融成一個 command,在 MAC 陣列上算(activation 查表 LUT,幾乎免費 — LeakyReLU 對 NPU 友善的原因) |
| **資料重排** | `input_KNERON_REFORMAT_next_0` | 格式轉換,如轉成硬體排列 `1W16C8B`(16-channel 對齊) |
| **資料搬移類運算** | `Concat_40_KNX_ks2d` | Concat 不是「算」,是記憶體排列;編譯成搬移 command |
| **優化器輔助節點** | `LeakyRelu_44_split_KNOPT_dummy_bn_1` | 分流/佔位用的假 BN(乘 1 加 0),處理一對多分支 |
| **其他直接支援 op** | MaxPool、Resize、Add | 各有對應 command |
| **CPU node** | (本專案 0 個 ✅) | NPU 不支援的 op 退回 CPU — 出現即效能警訊 |

### 2.3 每個 command 的微階段(evaluate 表格欄位)

| 階段 | 意義 |
|---|---|
| **MAC** | MAC 陣列實際運算(`MAC_cycle` × 時脈 = `MAC_runtime`) |
| **RDMA** | 從 DRAM 讀 feature map 進片上記憶體(KL730:8 GB/s) |
| **WDMA** | 結果寫回 DRAM(8 GB/s) |
| **GETW** | 權重載入專用通道(4.5 GB/s,獨立於 RDMA) |
| **PFUNC / CFUNC** | 韌體層前置/輔助函式 |
| **SYNC** | 等其他階段對齊的同步時間 |

效能分析讀法:某 command 的 `RDMA_runtime` >> `MAC_runtime` = 該層是**頻寬瓶頸**而非算力瓶頸。實例:本專案 `Conv_43` 的 MAC 0.55ms 但 RDMA 1.1ms — 640×640 大 feature map 進出 DRAM 的代價。

### 2.4 給使用者的「實際指令集」= operator 支援清單

日常開發真正面對的抽象層是公開的 **operator 支援表**(doc.kneron.com → toolchain → appendix → operators):Conv/DW-Conv、BN(融合)、ReLU/LeakyReLU/PReLU、sigmoid/tanh/exp(LUT)、Pool、Resize、Concat/Slice/Pad、Add/Mul、Gemm;KL730 另加 Transformer 類。

**模型設計守在這張表內 = 零 CPU fallback。**

### 2.5 自己看的方法(不用跑任何東西)

- `kneron_flow/opt_stage2_730.svg` — backend node graph 視覺化,瀏覽器直接開
- `kneron_flow/batch_compile.log` — §2.1 數字的出處
- evaluate 的逐 command 表格 — 在 nef-zoo 的輸出 log 裡

---

## 3. Operator 解惑:何時「用」?還是理解位置就好?

### 3.1 直接回答

**你不需要(也不會)逐一「選用」operator — 你選的是架構,operator 是架構的原子。**

類比組合語言:寫 C 的人不會問「何時用 MOV 還是 ADD」— 編譯器決定;但看得懂組語的人能讀懂 crash dump。

| 層級 | 誰決定 | 你的角色 |
|---|---|---|
| 架構(YOLOX、ResNet…) | 論文作者設計 operator 的組合 | **選擇**(本專案選了 YOLOX) |
| operator 組合 | 跟著架構走,位置固定 | **看懂**(讀圖、查支援表) |
| 單一 operator | 框架/編譯器實作 | 幾乎不碰 |

### 3.2 每個 operator 的白話職責 + 在 YOLOX 的固定位置

**主力計算(~95% 算力)**

| Operator | 白話 | 在 YOLOX 哪裡 |
|---|---|---|
| **Conv**(卷積) | 小窗口(3×3)滑過整張圖,每位置算加權和 — 「找局部圖案」 | 到處:backbone/neck/head 主體,197 節點過半是它 |
| **DW-Conv**(depthwise) | 省算力版 Conv:每 channel 自己卷自己,不跨 channel | YOLOX-**nano** 才用;-s 不用 |
| **Gemm/MatMul**(矩陣乘) | 全連接 — 每個輸出連每個輸入 | YOLOX **幾乎沒有**(全卷積);分類網路末層、Transformer 才大量用 |

**配角(讓主力正常工作)**

| Operator | 白話 | 在 YOLOX 哪裡 |
|---|---|---|
| **BN**(BatchNorm) | 把每層輸出拉回標準分布,訓練才不爆 | 每個 Conv 後;**部署時融進 Conv 權重消失** — 所以 NEF 裡找不到獨立 BN |
| **ReLU/LeakyReLU/PReLU** | activation 非線性(沒有它,百層 Conv 等於一層)。ReLU=負數歸零;Leaky=負數×0.1;PReLU=0.1 可學習 | 固定三件套 `Conv→BN→Act`,每組一個。編譯後即 `npu_fusion_node_Conv_X_LeakyRelu_Y` |
| **sigmoid/tanh/exp** | 壓縮函數:sigmoid 壓到 0~1(當機率)、exp 恆正(當寬高) | **只在 head 輸出端**:obj/cls 過 sigmoid、decode 的 `exp(dw)×stride`。NPU 查表 LUT,幾乎免費 |

**結構操作(不算數學,只搬資料)**

| Operator | 白話 | 在 YOLOX 哪裡 |
|---|---|---|
| **Pool**(Max/Avg) | 縮小 feature map:每 2×2 取最大/平均 | backbone 末端 SPP(三個 MaxPool 並排 = log 裡的 `MaxPool_109/110/111`) |
| **Resize** | 放大 feature map(插值) | neck top-down:深層放大後與淺層融合(`Resize_132/151`) |
| **Concat** | 沿 channel 拼接 | neck 融合處、SPP 出口 — 編譯成搬移 command `KNX_ks2d` |
| **Slice** | 切出 tensor 一部分 | **Focus 層**(YOLOX 第一層):640×640 切 4 份交錯子圖再拼 — 量化崩潰假設 #7 的 `_v_480~495` 殘留節點就是它 |
| **Pad** | 補邊(通常補 0)讓卷積尺寸對齊 | 各 Conv 隱含使用 |
| **Add / Mul** | 逐元素加/乘 | Add = residual 捷徑(`Add_53/67/72...`);Mul = SiLU 內部(x×sigmoid(x))— Kneron 版換掉 SiLU 後 Mul 就少了 |

### 3.3 什麼時候真的會「動」到 operator?三種實戰情境(本專案全發生過)

1. **NPU 相容性手術**:Kneron 把 YOLOX 的 SiLU 全換 LeakyReLU — 把支援表外/量化不友善的 operator 換成表內的。edge 部署最常見的 operator 級操作;動手的是 Kneron,我們只需理解為什麼。
2. **讀懂編譯器在抱怨什麼**:`torch_exported_onnx_flow` 報 unreachable nodes 時,認得 `Slice/Concat = Focus 層`,才能把錯誤翻譯成「export 前處理沒清乾淨第一層」。
3. **架構微調**:未來加 P2 head(小目標對策)= 在 neck 多接一組 Resize+Concat+Conv — 改 config 一行,但要能預期新增哪些 operator、是否仍零 CPU fallback。

### 3.4 結論

- **設計 operator 組合**:論文作者的事,永遠不用做。
- **理解 operator**:投資報酬在「讀圖、讀錯誤訊息、查支援表、預估改動後果」。
- 本專案的唯一 operator 級準則:**模型裡每個 operator 都落在 KL730 支援表內**,verify 方法 = `evaluate()` 報告的 `cpu_node: N/A`。

> 延伸:`yolox-explained.md` §2–§3 講的 backbone/neck/head,現在可以對應到「每段各用哪幾種 operator」再讀一次。
