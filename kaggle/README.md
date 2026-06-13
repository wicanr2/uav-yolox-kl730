# Kaggle 訓練指引

本機 CPU 只負責驗證管線;正式訓練在 Kaggle 免費 GPU 上跑(每週 30h 額度)。

## 你要做的事 (一次性準備,約 20 分鐘 + 上傳時間)

1. **Kaggle 帳號**:註冊 + **手機號碼驗證**(沒驗證不能開 GPU 與 Internet)。
2. **上傳兩個 Dataset**(Create → New Dataset,設 Private 即可):
   | Dataset 名稱 | 內容 | 來源 |
   |---|---|---|
   | `uav2uav-code` | `kaggle/uav2uav-code.zip`(本 repo 程式碼,已產好) | 本機 |
   | `antiuav410` | `datasets/Anti-UAV410.zip`(8.8GB,瀏覽器上傳慢可改用 kaggle CLI) | 本機 |
   - kaggle CLI 上傳法:`pip install kaggle` → 帳號頁下載 API token → `kaggle datasets create -p <資料夾>`(資料夾內放 zip + 自動產生的 metadata)。
3. **建 Notebook**:Create → New Notebook → File → Import Notebook → 上傳 `kaggle/train_uav2uav_kaggle.ipynb`。
4. **Notebook 設定**(右側面板):Accelerator = GPU **T4/P100**;Internet = **ON**;Add Input 掛上面兩個 dataset。
5. **Run All**。TU2U 與 HIT-UAV 會在 notebook 內自動從 GitHub 下載,不用上傳。

## 跑完之後

1. Notebook Output 頁下載 `work_dirs/yolox_s_uav2uav/latest.pth`(~100MB)。
2. 放回本機 `work_dirs/yolox_s_uav2uav/latest.pth`。
3. 本機接手:`make export-onnx && make nef`(CPU 即可)。

## 注意事項

- **12 小時 session 上限**:預設 40 epochs(T4 估 5–8 小時,估算)。被中斷時重跑所有 cell,Cell 4 加 `--resume-from /kaggle/working/work_dirs/yolox_s_uav2uav/latest.pth`(Output 的 work_dirs 要先 Add 回 Input 或重新訓練)。
- **Anti-UAV410 在 Kaggle 用 stride=30**(本機轉檔預設 10):控制訓練時間並平衡空對空(TU2U)與地對空(Anti-UAV410)的 uav 樣本比例。
- Colab 也可以,流程相同(把 `/kaggle/input` 改成掛 Google Drive 的路徑);但免費版 GPU 時數不保證,Kaggle 額度比較可預期。
- 程式碼有改動時:重新執行 `make kaggle-pack` 產 zip,到 Kaggle dataset 頁 New Version 上傳。
