# Session 教學:從零到 KL730 NEF — 我們做了什麼、為什麼這樣做

> 對象:想完整理解這個專案每一步的人(也就是你)。
> 性質:教學 + 決策記錄。每節都有「做了什麼 / 為什麼 / 怎麼重現 / 學到什麼」。
> 時間範圍:2026-06-11 ~ 06-12。
> 搭配閱讀:`PLAN.md`(規劃)、`docs/yolox-explained.md`(YOLOX 原理)、`README.md`(操作手冊)。

---

## 0. 全貌:我們在做什麼

目標:評估「無人機掛熱像儀 + Kneron KL730,機上即時辨識 uav/person/vehicle/bicycle」的可行性。

整條技術鏈:

```
資料集 (熱影像) → YOLOX-s 訓練 → ONNX (opset 11) → Kneron toolchain
                                                      ├ 優化 (kneronnxopt)
                                                      ├ 評估 (模擬 fps / CPU node)
                                                      ├ INT8 量化 (PTQ + 校正集)
                                                      └ 編譯 → NEF → KL730 板
```

本機只有 CPU,所以策略是:**本機驗證每一段管線會動,正式訓練丟雲端 GPU**。
截至目前,除了「正式訓練」,整條鏈都在本機跑通或暴露出真實問題了 — 後者正是評估案的價值。

---

## 1. 規劃階段:先查證,再寫 PLAN

### 做了什麼
派了 4 個研究 agent 平行查證:KL730 晶片/工具鏈、YOLO 版本與 NPU 相容性、三個資料集、硬體整合與成本。所有事實都附來源,查不到的標「未查證」。彙整成 `PLAN.md` + `CONTEXT.md`(術語表)。

### 學到什麼(顛覆直覺的發現)
1. **「YOLO 越新越好」是錯的**:Kneron 論壇有 YOLOv11 量化後輸出全 0 的未解案例;YOLOv10/26 的 NMS-free 結構(TopK 子圖)在各家 NPU 都是地雷。最後選 **YOLOX**:Kneron 官方流程驗證過、Apache-2.0 授權乾淨(Ultralytics 全系列是 AGPL,商用要買授權)。
2. **資料集授權要先查**:FLIR ADAS 授權頁 404 → 你決定 bypass。最終組合:TU2U(空對空 uav,MIT)+ HIT-UAV(空對地人/車,CC BY 4.0)+ Anti-UAV410(地對空 uav 擴充,**無 LICENSE,只能內部研究**)。
3. KL730 規格中「晶片功耗 / 板價 / 工業溫度」官方皆未公開 — 列入採購前待確認清單。

---

## 2. 軟體環境:docker + uv venv + 版本鎖定

### 為什麼版本要鎖這麼死
kneron-mmdetection 是 **mmdetection 2.25.0 時代的 fork**(2022 年),官方文件明示不再維護訓練環境。新版套件會在各種地方炸掉,所以:

| 元件 | 鎖定值 | 原因 |
|---|---|---|
| PyTorch | 1.12.1 (+cpu / +cu113) | mmcv-full 1.6.0 有預編 wheel 的最新組合 |
| mmcv-full | 1.6.0 | mmdet 2.25.0 的 assert 上限 |
| ONNX opset | 11 | export 腳本內寫死 `assert opset==11` |
| yapf | 0.40.1 | 新版移除 `FormatCode(verify=)`,mmcv 1.x 會 TypeError |
| onnx-simplifier | 0.4.36 | 0.5.x 依賴的 onnxsim 0.6 在 py3.8 無 wheel |

### 六個建置坑(依序踩到,全修進 `training/Dockerfile` 註解)

1. mmdet 舊式 `setup.py` 在 build 期 `import torch` → uv 隔離 build 環境沒有 torch → **`--no-build-isolation`**
2. `--no-build-isolation` 需要 venv 內有 setuptools,但 uv venv 預設不帶 → **先裝 `setuptools wheel`**
3. onnxsim 0.6 源碼編譯失敗 → **pin onnx-simplifier 0.4.36**
4. yapf 新版 API 變動 → **pin 0.40.1**
5. 容器以 root 跑,產物變 root 所有,host 操作被拒 → **`docker run -u $(id -u):$(id -g) -e HOME=/tmp`**
6. uv 把 CPython 裝在 `/root/.local` 下,非 root 使用者進不去(`python: command not found`)→ **`UV_PYTHON_INSTALL_DIR=/opt/uv-python`**

### 學到什麼
- 跑舊生態(mm 系列)+ 新工具(uv)= 每一層假設都要驗。
- **背景指令不要接 `| tail`**:pipe 會吃掉錯誤碼,我因此兩次把失敗誤判為成功。

---

## 3. 資料管線:三個來源 → 統一 COCO

### 格式陷阱(全部實測確認,寫進轉檔腳本註解)

| 資料集 | 宣稱格式 | 實際狀況 |
|---|---|---|
| HIT-UAV | 「JSON (COCO)」 | top-level key 是 `annotation`(單數)、影像欄位叫 `filename`、README 說 bbox 是中心點格式但**實檔是 `[xmin,ymin,w,h]`**(用 segmentation 多邊形反推驗證) |
| Anti-UAV410 | tracking 格式 | 每序列 `IR_label.json` = `{"exist":[0/1...], "gt_rect":[[x,y,w,h]...]}`;`exist=0` 的幀要跳過;解析度官方沒寫,要從實檔讀 |
| TU2U | YOLO txt | 正常,normalized cxcywh → 轉絕對座標 |

### 設計決策
- 4 類 canonical schema 寫在 `datasets/coco_common.py`,三個轉檔器輸出同一份 categories,merge 只做 id 重編 — **轉檔期就對齊,不留到訓練期**。
- Anti-UAV410 以 **stride=10 抽幀**(相鄰幀近重複,全收是灌水);本機轉出 2 萬張 vs TU2U 2,514 張,**地對空:空對空 = 8:1 的比例失衡**已記錄,Kaggle 版改 stride=30 平衡。
- 最終 merge:**train 25,459 張 / 40,970 標註**。

### 怎麼重現
```bash
make data-download && make data-convert && make smoke
```

---

## 4. 訓練管線驗證(CPU smoke)

設計:`make train-smoke` = 1 epoch × 200 張子集 × batch 2,目的不是練出模型,是證明「資料 → dataloader → loss → checkpoint」整條會動。

結果:loss 65.7 → 49.1(loss_obj 60→43),`epoch_1.pth` 正常產出。**結論:雲端 GPU 上跑 `make train` 不會有管線問題,只剩速度差異。**

config 重點(`training/configs/yolox_s_uav2uav.py`):
- 繼承 Kneron 版 `yolox_s_..._img_norm.py`(LeakyReLU 取代 SiLU、mean=128/std=256 — 都是 NPU 友善修改,原理見 yolox-explained.md §10)
- 路徑可用環境變數覆寫(`KMM_ROOT`/`UAV2UAV_ROOT`),同一份 config 本機 docker 與 Kaggle 通用

---

## 5. 轉換鏈:不訓練就能驗證的聰明做法

關鍵洞察:**Kneron Model Zoo 有官方預訓練 YOLOX-s(COCO 80 類)**,可以在「沒有 GPU、沒有板子」的情況下把風險最高的轉換鏈先走完。

```bash
make zoo-export   # 官方權重 → ONNX (opset 11, --skip-postprocess)
make nef-zoo      # ONNX → 優化 → 評估 → 量化 → NEF → 驗收
```

成果(這是專案目前最有價值的數字):

| 指標 | 結果 |
|---|---|
| KL730 模擬 fps | **43.2**(YOLOX-s 640×640)— 高於 25 fps 目標線 |
| CPU fallback node | **0 個**(整張圖全在 NPU 上) |
| NEF 大小 | 9.3MB(FP32 35MB 的 ~1/4,INT8 正常) |

ONNX 結構驗證:input 1×3×640×640、**9 個分開的輸出**(3 尺度 × cls/reg/obj)— 正是官方「不要 concat」trick 的正確形狀。

途中發現:kneron-mmdetection 內建的舊 beta 優化器會 crash,但檔案在 crash 前已存好,新版 kneronnxopt 會在 toolchain 內接手 — Makefile 已改成以「產物有效性」驗收而非腳本 exit code。

---

## 6. 量化退化:一場教科書級的 learning loop(進行中)

這是本 session 技術含量最高的部分。完整記錄假設與證據,因為**過程本身就是評估報告的素材**。

### 症狀
NEF 模擬推論輸出退化:float 端 logits 範圍 −15~+7,NEF 端被壓成 ±0.012,每個 channel 只剩 **1 個唯一值**(整面常數)。

### 假設演進(每輪一個實驗,逐一排除)

| # | 假設 | 實驗 | 結果 |
|---|---|---|---|
| 1 | 量化 bitwidth 設定不足(官方建議 mix 模式) | `mix balance`/`mix light` 重量化 | ❌ 輸出一模一樣;且報告顯示**參數被默默忽略**(報告仍全 int8) |
| 2 | per-channel scale 未除回(檢查方法錯) | 逐 channel 算相關性 | ❌ 每 channel 唯一值 = 1,是真崩潰不是 scale 問題 |
| 3 | 輸出位寬不夠(radix 過寬) | `model_out_bitwidth_mode=int16` | ❌ 仍全崩(int16 都救不了 → 問題更根本) |
| 4 | 730 平台路徑不成熟 | 同 ONNX 編 **KL720** 對照 | ❌ 720 也崩(型態略異)→ 排除 730 限定 |
| 5 | 校正資料格式錯(HWC vs NCHW) | `--layout hwc` 重量化 | ❌ 官方 mobilenetv2 範例證實 **NCHW+batch 才是對的**,我原本就對;HWC 反而讓 csim assert(順便洩漏 input radix=[8,8,8] 正確) |
| 6 | NEF 模擬器本身有問題 | **官方 mobilenetv2 範例完整跑**(官方模型+官方流程+同一個模擬器) | ❌ **官方範例 float vs NEF corr = 0.971**、輸出範圍幾乎重合、bie 與 nef 結果一致 → 模擬器沒問題 |
| **7** | **`percentage=1.0` 對 raw-logit 輸出把量化範圍撐爆** | 改 `datapath_range_method="mmse"`,其餘維持 int8 | ✅ **通過!9 輸出全部 corr 0.90–0.97、零崩潰 channel** |

### 收斂後的結論(2026-06-12,已解決 ✅)

**根因確認:`percentage=1.0` 範圍估計法 × `--skip-postprocess` 的 raw-logit 輸出 = 崩潰。**

`--skip-postprocess` 匯出的是無 sigmoid 的 raw logits(cls 範圍 −15~+1)。`percentage=1.0`(官方對 detection 的建議,但**前提是輸出有界**)會把量化範圍張到含 −15 離群值 → INT8 的 256 格攤在 ~16 寬的範圍 → 多數實際值擠進同一格 → 整面常數。改用 **`mmse`(SNR-based,對離群值穩健)** 一個變因即完全修復,**無需動位寬、無需重 export、無需 mix 模式(也就無記憶體風險)**。

之前的假設 #7(graph 殘留)被否證:同一顆 ONNX、同樣的圖,只換 range method 就過了 → 問題不在 graph,在範圍估計。

**對部署的意義**:Makefile 的 `nef` / `nef-zoo` 已預設 `--range-method mmse`。正式 4 類模型走同一條路即可;校正集屆時換成混合資料集的 val(記得前處理一致 `x/256−0.5`)。量化精度驗收線仍是 mAP 掉幅 <3%(corr 只是 sanity,真正驗收要等有標註的 mAP 比對)。

### 若繼續,下一步選項(按成本排序)
1. **修 export 端**:查 `torch_exported_onnx_flow` 的 unreachable nodes 成因(可能是新版 onnx 產生的 graph 差異),讓官方前處理走完,重新量化 — 一次實驗即可驗證假設 #7。
2. **對照官方 KL720 yolox zoo NEF**(Kneron 自己編好的 yolox NEF)在同一模擬器的輸出 — 分辨「YOLOX 模型類型」vs「我們的 export」。
3. **上 Kneron 論壇**:最小重現包已齊(input.onnx + 校正集 + 步驟 + 六輪對照數據)。
4. **擱置到正式模型階段**:Kaggle 訓完 4 類模型重走一遍(屆時校正集也換),問題不消失再追。

### 重要備註
這個問題**不影響已確認的成果**(編譯成功、43 fps、零 CPU node)。它影響的是「上板前能不能在軟體端驗證量化精度」— 最壞情況是要等實機才能驗,這本身就是評估報告該寫的風險。

---

## 7. 插曲:把你的電腦搞當機(已確診、已修)

### 發生什麼
6/11 17:44 我啟動 mix 模式量化(threads=8 × 200 張校正圖),17:46 起 journald 連續記錄 "Under memory pressure",17:48 你被迫重開機。

### 根因
mix 模式做 int16 模擬,記憶體遠高於 int8;30GB RAM 用罄 + 8GB swap → **swap thrash 假死**(kernel OOM killer 沒觸發,所以是凍結不是 crash)。診斷方法:`journalctl --list-boots` 對時間軸 + `journalctl -b -1 -k` 看前次 boot 的 kernel 訊息。

### 防護(已上)
- toolchain 容器一律 `--memory=20g --memory-swap=20g`:再爆只會 kill 容器,不會拖垮主機
- 量化降載:threads 4、校正 100 張
- 建議(你自己跑實驗也適用):大記憶體 docker 工作都設 `--memory`;swap 8GB 偏小可加大

---

## 7.5 端到端 demo:用 zoo 模型親手證明「為什麼非訓練不可」(2026-06-12)

量化解決後,把推論的後半段補上:寫 `firmware/detect/postprocess.py`(YOLOX decode + NMS,純 numpy,板上跑 A55),用 zoo NEF(COCO 80 類)跑 6 張真實 HIT-UAV 空對地熱影像,decode、畫框、存 PNG。

結果(圖見 `docs/images/demo/det_*.png`):

| 影像 | 偵測框 |
|---|---|
| 6 張 HIT-UAV(實際含多個人/車) | 共 **3 框,全錯類**:2 個 "bird"、1 個 cls55,**零個 person/car** |

**這正是預期、且是最好的教學結果**:
- **pipeline 完全可運作** — 6 張全跑完、decode 出框、畫圖存檔,證明站 5(量化)→站 6(後處理)鏈通。
- **偵測是垃圾** — COCO 是「地面、可見光」訓練,餵它「空中、熱影像」domain gap 巨大,只能把熱斑亂猜成 bird。
- → **這就是「為什麼非得用熱影像資料 fine-tune」的鐵證**(也就是 Kaggle 那一步的全部意義)。等 4 類熱影像模型訓出,同一條 demo 會畫出正確的框。

方法論意義:**用一個「預期會失敗」的對照,具體展示問題的形狀** — 比口頭說「domain gap 會影響精度」有說服力得多。

---

## 8. Kaggle 訓練包(已備好,等你決定)

你要做的只有:① Kaggle 帳號+手機驗證;② 上傳 `kaggle/uav2uav-code.zip`(62KB)與 `datasets/Anti-UAV410.zip`(8.8GB)成兩個 private dataset;③ import `kaggle/train_uav2uav_kaggle.ipynb`,開 GPU + Internet,Run All;④ 抓回 `latest.pth`。細節在 `kaggle/README.md`。

Notebook 與本機 docker 用**完全相同的版本鎖定**(uv venv py3.10 + cu113),40 epochs 在 T4 估 5–8 小時(12h session 內)。

---

## 9. 目前狀態與下一步選項

### 已完成 ✅
- PLAN / CONTEXT / YOLOX 原理文件
- 全 docker 化環境(6 坑修平)、資料管線(25k 張)、訓練管線(smoke 過)
- KL730 編譯鏈:**43.2 fps、零 CPU fallback、NEF 產出**
- Kaggle 訓練包、當機根因修復

### 已停止(依指示,2026-06-12)
所有 docker 實驗已停止;兩個對照實驗在停止前自然完成,結果已寫入 §6。

### 你要決定的事
1. **量化問題要追到底嗎?** 選項見 §6「若繼續,下一步選項」(建議:選項 1 成本最低、鑑別力最高;選項 4 最省事)。
2. **Kaggle 訓練何時啟動?**(素材已備好,純看你時間)
3. `datasets/Anti-UAV410.zip`(8.8GB)解壓完成,確認後可刪。

---

## 附:本 session 的方法論備忘

1. **先建快速驗證迴路再下結論**(rules/60):量化問題每輪實驗 15-20 分,六輪下來假設空間收斂得很乾淨。
2. **對照組思維**:720 vs 730、官方範例 vs 我們的模型 — 一次只變一個變因。
3. **驗收器要有鑑別力**:「非全 0」抓不到退化,「攤平 corr」會被 per-channel scale 騙,最後落在「逐 channel corr + 崩潰 ch 計數」。
4. **silent failure 最危險**:knerex 對錯誤參數不報錯、報告與實際設定脫鉤 — 工具鏈評估時要主動驗證「設定真的生效了嗎」。
5. **資源護欄先於大任務**:`--memory` 上限這種事,出事前沒人想到,出事後人人想得到。
