# 名詞解釋 (Glossary)

> 按管線流程排序,不照字母 — 從上往下讀就是整條技術鏈。
> 每個詞附本專案的實例。速查用 Ctrl+F。

---

## 一、資料與標註

**COCO 格式**
物件偵測最通用的標註格式:一個大 JSON,裡面三張表 — `images`(每張圖的檔名/尺寸/id)、`annotations`(每個框的 `[x,y,寬,高]` + 類別 + 所屬圖 id)、`categories`(類別清單)。mmdetection 直接吃這格式。本專案三個資料集格式各異,全部轉成 COCO 再合併(`data/coco/annotations/train.json`)。

**YOLO 格式 (標註)**
另一種標註格式:每張圖配一個 `.txt`,每行 `類別 cx cy w h`,座標是 0~1 的比例值且 `cx,cy` 是**框中心**。TU2U 用這格式。轉 COCO 時要做 `中心點→左上角` 與 `比例→像素` 的換算。

**bounding box (bbox)**
框住目標的矩形。注意各格式的座標慣例不同:COCO 是 `[左上x, 左上y, 寬, 高]`,YOLO txt 是中心點比例 — 搞混會讓所有框偏移半個框(HIT-UAV 的 README 就把自己的格式寫錯了,我們用實檔反推驗證)。

**train / val / test split**
資料三切分:train 拿來學、val 在訓練中監控(挑最好的 epoch)、test 只在最後驗收用一次。本專案沿用各資料集官方切分,不同視角的 test 不可混測。

**抽幀 (frame sampling / stride)**
影片相鄰幀幾乎一樣,全部拿來訓練是灌水。Anti-UAV410 是影片資料集,我們每 10 幀取 1 張(`--stride 10`)。

---

## 二、模型與訓練

**YOLOX**
本專案選用的物件偵測模型(2021,曠視)。anchor-free、decoupled head、SimOTA 三大特點 — 詳細原理見 `docs/yolox-explained.md`。選它的原因:Kneron 官方流程驗證過 + Apache-2.0 授權可商用。`-s` 是尺寸代號(9M 參數)。

**mmdetection / mmcv**
OpenMMLab 的物件偵測框架(mmdetection)與其底層工具庫(mmcv)。用 Python config 檔描述整個訓練(模型結構、資料、優化器),`tools/train.py` 一鍵訓練。

**kneron-mmdetection**
Kneron fork 的 mmdetection(基於 2.25.0),加了 NPU 友善修改:activation 換 LeakyReLU、專用的 ONNX export 腳本。我們的訓練環境就是裝這個。

**checkpoint (.pth)**
訓練過程存的模型權重檔。`latest.pth` 是最新一份;Kaggle 訓完要抓回來的就是它(~100MB)。

**epoch / iteration / batch**
batch = 一次餵給模型的圖片數(我們本機 2、雲端 8);iteration = 處理一個 batch;epoch = 整個訓練集全部過一遍。smoke 訓練 = 200 張 ÷ batch 2 = 100 iterations = 1 epoch。

**loss**
模型「錯多少」的數字,訓練就是讓它下降。YOLOX 的 loss 有三項:`loss_cls`(類別錯)、`loss_bbox`(框不準)、`loss_obj`(有沒有東西判錯)。smoke 訓練看到 65.7→49.1 就是「有在學」的證據。

**mAP (mean Average Precision)**
偵測模型的標準評分(0~1,越高越好)。同時考慮「找得全不全」與「框得準不準」。我們的量化驗收線:INT8 模型的 mAP 比 FP32 掉 <3% 才合格。

**預訓練權重 (pretrained weights)**
別人先在大資料集(COCO,12 萬張日常照片)上訓好的模型。我們拿它當起點再用熱影像微調(fine-tune),比從零訓快得多。Kneron Model Zoo 的 YOLOX-s 就是這種——本專案還沒訓練就先拿它驗證了整條轉換鏈。

**smoke test / smoke 訓練**
「冒煙測試」:不求結果好,只求整條管線會動、不冒煙。`make train-smoke` 用 200 張跑 1 epoch,證明資料→模型→checkpoint 全通。

---

## 三、模型交換與轉換

**ONNX (Open Neural Network Exchange)**
模型的「通用中間格式」。PyTorch 訓好的模型 export 成 `.onnx`,任何支援 ONNX 的硬體工具鏈都能接手 — 它是「訓練世界」與「晶片世界」的交界。內容是一張運算圖(節點 = Conv、LeakyRelu 等運算)。

**opset**
ONNX 的「版本號」,規定圖裡能用哪些運算與其行為。Kneron 的 export 腳本鎖 opset 11(較舊但相容性穩)。

**operator (op / 節點)**
運算圖裡的單一運算,如 Conv(卷積)、Concat(拼接)、Slice(切片)。NPU 只支援特定 operator 清單;不支援的會「掉到 CPU 執行」(見 CPU fallback node)。

**onnx-simplifier / kneronnxopt / torch_exported_onnx_flow**
三種「圖優化器」:把 export 出來的 ONNX 圖化簡(合併常數、刪冗餘節點)。`onnx-simplifier` 是通用工具;`torch_exported_onnx_flow` 是 kneron-mmdetection 內建的舊版 Kneron 專用優化(在我們環境會 crash);`kneronnxopt` 是 toolchain 裡的新版官方優化器。目前量化崩潰的最強假設就是「舊版優化沒走完,新版沒清乾淨 mmdet 特有的圖結構」。

---

## 四、Kneron 工具鏈與量化

**Kneron toolchain**
Kneron 官方的 docker image(`kneron/toolchain`),包含把 ONNX 變成晶片可執行檔的全部工具。我們用 v0.33.0。

**ktc**
toolchain 裡的 Python API(**K**neron **T**ool**C**hain)。核心物件 `ktc.ModelConfig(id, version, "730", onnx_model=...)`,流程:`evaluate()` 評估 → `analysis()` 量化 → `ktc.compile()` 編譯。

**量化 (Quantization)**
把模型從 32-bit 浮點數(FP32)壓成 8-bit 整數(INT8)。模型變 1/4 大、NPU 才跑得快 — 但精度會掉,掉多少取決於量化做得好不好。**這是整條部署鏈風險最高的一步**(本專案目前卡的就是這裡)。

**PTQ (Post-Training Quantization)**
「訓練後量化」:模型訓完才量化,不用重訓。做法:餵一批代表性圖片(校正集),觀察每層數值的實際範圍,據此決定怎麼把浮點數映射到 256 個整數格。Kneron 用的就是 PTQ。

**校正集 (calibration set)**
PTQ 觀察用的那批圖(我們用 100 張 TU2U val 熱影像)。關鍵原則:**前處理必須與訓練完全一致**(`x/256−0.5`),且格式是 NCHW + batch 維(官方範例已證實)。

**INT8 / INT16 / bitwidth (位寬)**
數值用幾個 bit 存。INT8 = 256 個格子,INT16 = 65,536 個格子。位寬越大越精確但越慢。`analysis()` 可分別指定 weight/datapath/輸出的位寬,還有 `mix balance`/`mix light` 混合模式。

**radix / scale**
浮點↔整數的換算係數。`radix=8` 意思是「浮點值 ×2⁸=256 後取整存成 INT8」,等於假設數值範圍在 ±0.5。radix 選錯(範圍估太寬)→ 所有值擠進同一格 → 輸出變常數,就是我們觀察到的崩潰型態。

**per-channel 量化**
每個輸出 channel 用自己的 radix(而非整層共用)。更精確,但也讓驗收變麻煩 — 攤平整個 tensor 算相關性會被不同 channel 的 scale 干擾,必須逐 channel 比。

**knerex**
toolchain 內實際執行量化分析的引擎(`analysis()` 背後的程式)。我們發現它對某些參數**默默忽略不報錯**(報告仍顯示 int8)— 評估報告要記的 silent failure。

**BIE**
量化完成後的中間檔(**B**inary **I**ntermediate ...,Kneron 自家格式):「已定點化但還沒編譯」的模型。流程上在 ONNX 與 NEF 之間。可在模擬器跑,用來單獨驗證「量化」這一步的精度。

**NEF (NPU Executable Format)** ★你問的這個
**Kneron NPU 的最終可執行檔** — 等同於「編譯好的程式」,燒進 KL730 板就直接跑。由 `ktc.compile()` 從量化後的模型編出來。我們產出的 `models_730.nef` 是 9.3MB(原 FP32 ONNX 35MB 的 ~1/4,符合 INT8 壓縮比)。上板部署 = Kneron PLUS 載入這個檔。

**batch compile**
`ktc.compile([km])` 的正式名稱 — 可以把多個模型編進同一個 NEF(我們只編一個)。

**ip_evaluator / evaluate()**
量化前的「紙上評估」:不用真硬體,模擬計算這顆模型在目標 NPU 上的 fps、記憶體頻寬、有沒有 operator 掉到 CPU。我們的 43.2 fps 就是它算的。

**CPU fallback node (cpu_node)**
NPU 不支援的 operator 會退回 CPU 執行,嚴重拖慢速度。評估報告裡 `cpu_node: N/A` = 零個 = 整張圖全在 NPU 上跑,是我們最重要的綠燈之一。

**E2E Simulator / csim**
toolchain 內的模擬器:在 PC 上模擬 NPU 執行 BIE/NEF,不需要實體板子。`csim` 是其中模擬硬體行為的元件。我們所有「NEF 推論驗證」都是在這上面跑的 — 官方 mobilenetv2 範例 corr 0.971 證明它本身可信。

**Kneron PLUS**
**上板後**用的 SDK(C/Python):負責把 NEF 載入實體 KL730、餵影像、收推論結果。和 toolchain 的分工:toolchain 管「編譯出 NEF」,PLUS 管「在板子上跑 NEF」。

---

## 五、硬體

**NPU (Neural Processing Unit)**
專為神經網路設計的處理器。比 CPU 快、比 GPU 省電,代價是只支援固定的 operator 集合、通常要 INT8 量化。

**KL730**
Kneron 的 edge AI SoC:NPU(3.6 eTOPS INT8)+ 四核 Cortex-A55(可獨立跑 Linux)+ ISP。本專案的目標晶片。KL720/KL630 是前代(算力約 1/2、1/7)。

**eTOPS / TOPS**
算力單位,每秒兆次運算(Tera Operations Per Second)。e 前綴是 Kneron 行銷用的「effective」。只能同代比較,跨家比意義有限。

**KNEO Pi**
KL730 的開發板形態(類 Raspberry Pi)。可獨立跑 Linux,官方宣稱跑 YOLOv5 >30fps @2W。未來上板部署的載體。

**MIPI CSI**
相機模組接 SoC 的標準高速介面(手機相機都用這個)。KL730 支援直入,熱像儀可不經 USB 直接接。

**FLIR Boson / Lepton**
Teledyne FLIR 的熱像儀模組。Boson 640(640×512,7.5g,~$3,500)是正式選型;Lepton 3.5(160×120,0.9g,$164)是便宜的管線驗證用。

**FFC / AGC**
熱像儀特有的兩個處理:FFC(Flat Field Correction)= 週期性快門校正,校正瞬間會掉幀;AGC = 把 16-bit 原始熱數據壓到 8-bit 可顯示/可推論範圍的自動增益。

---

## 六、環境與工具

**Docker image / container**
image = 打包好的環境快照(含 OS、Python、所有套件);container = image 跑起來的實例。本專案兩個 image:自建的 `uav2uav-train`(訓練)與官方 `kneron/toolchain`(轉換)。主機上不裝任何 Python 套件。

**Dockerfile**
描述「怎麼建 image」的腳本。我們的在 `training/Dockerfile`,六個建置坑的修法都以註解形式記在裡面。

**uv / venv**
uv = 新一代 Python 套件管理器(快、嚴格);venv = 隔離的 Python 環境。規則:容器內也用 uv venv,雙層隔離。

**wheel**
Python 套件的預編譯安裝包(`.whl`)。「無 wheel」= 要從源碼編譯 = 在舊 Python 版本上常失敗(onnxsim 0.6 之坑)。

**pin (版本鎖定)**
明確指定套件版本(`mmcv-full==1.6.0`)而不是裝最新。舊生態(mmdet 2.x)+ 新套件 = 各種爆炸,所以全部 pin 死,清單在 README「關鍵版本鎖定」。

**Makefile / make target**
把常用指令序列寫成短命令:`make train-smoke` 背後是一長串 docker run。本專案所有操作的入口,`make help` 看全部。

**`--memory` (docker 記憶體上限)**
限制容器最多用多少 RAM。當機事故後所有 toolchain 容器都加 `--memory=20g`:再爆只會 kill 容器,不會把主機拖進 swap thrash。

---

## 七、診斷與驗證

**OOM (Out Of Memory)**
記憶體耗盡。kernel 的 OOM killer 會殺掉最肥的程序自保 — 但它沒觸發時,系統會先進 swap thrash。

**swap / swap thrash**
swap = RAM 滿了拿硬碟當記憶體用。thrash = RAM 與硬碟之間瘋狂搬頁,系統看起來「凍住但沒死」— 6/11 當機就是這個,證據是 journald 連續的 "Under memory pressure"。

**journalctl**
Linux 系統日誌查詢工具。`--list-boots` 列開機記錄、`-b -1` 看上一次開機的日誌 — 重開機後回溯死因的標準工具。

**corr (相關性, correlation)**
兩組數字的線性一致程度(−1~1)。我們用它驗收量化:float 模型輸出 vs NEF 輸出逐 channel 算 corr,>0.9 合格。官方範例 0.971 = 好;我們的 YOLOX 全 channel 常數 = 崩潰。

**sanity check**
「理智檢查」:快速驗證結果不是明顯荒謬(輸出全 0?NaN?常數?)。`convert_to_nef.py --check` 內建這個,正是它抓到量化崩潰。

**silent failure**
失敗但不報錯,默默給你壞結果 — 最危險的一類問題。本專案已遇兩次:knerex 忽略量化參數不吭聲、`wget -nc` 檔案已存在時回非零碼中斷流程。對策:每步都驗證「真的生效了嗎」。

**learning loop**
本專案的除錯方法論(觀察→假設→最小實驗→排除→更新):量化問題六輪實驗,每輪只變一個變因,假設空間從「哪都可能」收斂到單一嫌疑。完整過程見 `session-walkthrough.md` §6。

**對照組 (control)**
固定其他變因、只變一個的比較實驗。例:同 ONNX 編 720 vs 730(排除平台)、官方範例 vs 我們的模型(排除模擬器)。

**最小重現包 (minimal repro)**
讓別人(如 Kneron 原廠)能重現問題的最小檔案組合:一個 ONNX + 校正集 + 步驟 + 預期/實際結果。要上論壇求助前的標準準備。
