# CONTEXT.md — UAV2UAV Domain Glossary

> 本專案 canonical 術語表。寫程式 / 命名 / 文件優先使用此處詞彙。

## 術語

- **UAV2UAV** — 本專案代號:無人機機載熱影像即時辨識 (目標:UAV / 車輛 / 人)。
- **air-to-air (空對空)** — 無人機拍攝無人機的取像幾何。_Avoid_: 機對機。
- **air-to-ground (空對地)** — 無人機拍攝地面車/人的取像幾何。
- **KL730** — Kneron edge AI SoC (3.6 eTOPS INT8, Quad Cortex-A55)。本案推論晶片。
- **KNEO Pi** — KL730 的 SBC 開發板形態 (KP73B703A-M1),可獨立跑 Linux。
- **NEF** — NPU Executable Format,Kneron toolchain 編譯出的部署模型檔。
- **toolchain** — Kneron 官方 docker (`kneron/toolchain`):ONNX → 優化 → 量化 → NEF。
- **Kneron PLUS** — host 端推論 SDK (C/Python),機上服務透過它呼叫 NPU。
- **PTQ** — Post-Training Quantization,Kneron 的 INT8 量化方式,需校正集。
- **CPU node** — 轉換時 NPU 不支援、落到 CPU 執行的 operator;部署優化要消除。
- **P2 head** — stride-4 偵測頭,小目標對策之一。
- **FFC** — Flat Field Correction,非製冷熱像儀的週期快門校正;校正瞬間掉幀。
- **AGC** — 16-bit 熱影像壓到 8-bit 顯示/推論範圍的自動增益控制。
- **TU2U** — ThermalUAV2UAV dataset 簡稱 (3,856 張、空對空、MIT)。
- **FLIR ADAS** — Teledyne FLIR ADAS Thermal Dataset v2。**2026-06-11 決策:不採用 (bypass)**,授權風險迴避。
- **HIT-UAV** — 空對地熱影像資料集 (CC BY 4.0)。
- **Anti-UAV410** — 地對空熱紅外追蹤 benchmark,抽幀作 UAV 偵測擴充。
- **Boson 線 / Lepton 驗證線** — 兩條硬體 BOM:正式原型 (Boson 640) vs 低成本 pipeline 驗證 (Lepton 3.5)。
- **掛機地面測試** — 酬載裝機但不起飛的整合驗證階段。

## Flagged ambiguities

- 「TOPS」:Kneron 官網 (3.6/7.2 eTOPS) 與新聞稿 (最高 8 TOPS) 數字不一致 → 本專案一律引用官網 3.6 eTOPS INT8。
- 「YOLO 最新版」:指「能在 KL730 正確量化的最新版」,非單純版本號最大。
- 「偵測距離」:尚未定義需求值 (m),待使用者確認後補進本表。
