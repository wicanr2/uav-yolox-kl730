# 02 · 量化崩潰根因與驗收 (本 skill 最高價值)

## 症狀

NEF 模擬推論輸出退化:float ONNX 端 logits 範圍 −15~+7,NEF 端被壓成 ±0.012,
**每個 channel 只剩 1 個唯一值 (整面常數)**。模型「跑得動」但輸出全是垃圾。

## 7 輪排查 (示範:如何用對照組收斂,不要照抄結論,要學方法)

| # | 假設 | 實驗 | 結果 |
|---|---|---|---|
| 1 | bitwidth 不足 | 改 mix balance/light | ❌ 一樣;且報告顯示參數被默默忽略 |
| 2 | per-channel scale 沒除回 (檢查法錯) | 逐 channel 算 corr | ❌ 每 ch 唯一值=1,真崩潰 |
| 3 | 輸出位寬不夠 | model_out_bitwidth=int16 | ❌ int16 也崩 → 不是位寬問題 |
| 4 | 730 平台不成熟 | 同 ONNX 編 KL720 對照 | ❌ 720 也崩 → 非平台限定 |
| 5 | 校正格式錯 (HWC/NCHW) | 改 layout | ❌ 官方 mobilenetv2 範例證實 NCHW+batch 才對 |
| 6 | 模擬器本身壞 | 跑官方 mobilenetv2 範例 | ❌ 官方 corr=0.971 → 模擬器正常 |
| **7** | **percentage=1.0 對 raw-logit 把範圍撐爆** | 改 `mmse` range method | ✅ **通過,9 輸出 corr 0.90–0.97** |

## 根因

`pytorch2onnx_kneron.py --skip-postprocess` 匯出的是**無 sigmoid 的 raw logits**
(cls 範圍 −15~+1)。detection 官方建議 `percentage=1.0` 的前提是**輸出有界 (0~1)**;
對 raw logit,`percentage=1.0` 會把量化範圍張到含 −15 的離群值 →
INT8 256 格攤在 ~16 寬範圍 → 多數實際值擠進同一格 → 整面常數。

## 解法

`km.analysis(..., datapath_range_method="mmse")` — SNR-based,對離群值穩健。
**只改這一個參數**;不必動位寬、不必重 export、不必 mix 模式 (mix = int16 模擬 = 吃爆記憶體)。

```python
bie_path = km.analysis(
    {input_name: img_list},
    threads=4,
    datapath_range_method="mmse",     # ← 關鍵 (raw-logit 模型)
    datapath_bitwidth_mode="int8",    # int8 最省記憶體
    weight_bitwidth_mode="int8",
    model_out_bitwidth_mode="int8",
)
```
(替代解:不要 `--skip-postprocess`,讓 sigmoid/exp 留在模型內 → 輸出有界 → percentage 可用。
但那會牽動 export 與後處理分工,mmse 是成本最低的解。)

## 校正集前處理 — 必須與訓練一致

YOLOX Kneron config 的 img_norm 是 mean=128/std=256,即 `x/256 − 0.5`;layout NCHW + batch 維:
```python
arr = np.array(img.resize((640,640))).astype(np.float32)/256.0 - 0.5
arr = np.transpose(arr,(2,0,1))[None]   # 1x3xHxW
```
校正集從 val split 均勻抽 100–200 張即可。

## 驗收方法 (要有鑑別力)

```python
ref = ktc.kneron_inference([x], onnx_file=opt_onnx, input_names=[n], platform=730)
nef = ktc.kneron_inference([x], nef_file=nef_path, input_names=[n], platform=730)
# 逐 channel corr (730 為 per-channel 量化,攤平整 tensor 算 corr 會被 scale 騙)
for r,q in zip(ref,nef):
    r,q = np.asarray(r)[0], np.asarray(q)[0]   # C,H,W
    for c in range(r.shape[0]):
        if r[c].std()>1e-6:
            corr = np.corrcoef(r[c].ravel(), q[c].ravel())[0,1]   # 合格 >0.9
            dead = q[c].std()<1e-6   # 崩潰 channel
```
三道驗收:① 逐 channel corr >0.9 且零崩潰 ② evaluate 報告 `cpu_node: N/A` ③ 有標註後 mAP 掉幅 <3%。
**「輸出非全 0」太弱**,抓不到本案的退化。

## 名詞

- **BIE** = 量化後、編譯前的中間檔 (可單獨在模擬器驗量化精度)。
- **NEF** = 最終晶片可執行檔。
- **knerex** = 實際做量化分析的引擎;會 silent 忽略不支援的參數 → 要驗證報告反映你的設定。
