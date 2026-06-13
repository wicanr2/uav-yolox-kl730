#!/usr/bin/env bash
# 下載三個資料集到 data/raw/
# - ThermalUAV2UAV: git clone 直接拿全 (~0.97 GiB, 影像 commit 在 repo 內)
# - HIT-UAV: GitHub release zip (~775 MiB)
# - Anti-UAV410: 只能手動 (Google Drive / 百度盤),本腳本只印指引
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw
mkdir -p "$RAW"

if [ ! -d "$RAW/ThermalUAV2UAV_Dataset" ]; then
    echo "[1/3] clone ThermalUAV2UAV (~1 GiB)..."
    git clone --depth 1 https://github.com/GabryV00/ThermalUAV2UAV_Dataset "$RAW/ThermalUAV2UAV_Dataset"
else
    echo "[1/3] ThermalUAV2UAV 已存在,略過"
fi

if [ ! -d "$RAW/HIT-UAV" ]; then
    echo "[2/3] 下載 HIT-UAV release v1.2.1 (~775 MiB)..."
    wget -c -O "$RAW/HIT-UAV.zip" \
        https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset/releases/download/v1.2.1/HIT-UAV.zip
    unzip -q "$RAW/HIT-UAV.zip" -d "$RAW/"
    # 解壓後頂層目錄名以實際為準;預期為 HIT-UAV/
    [ -d "$RAW/HIT-UAV" ] || echo "警告: 解壓後找不到 $RAW/HIT-UAV,請檢查 zip 內層目錄名並改名"
    rm -f "$RAW/HIT-UAV.zip"
else
    echo "[2/3] HIT-UAV 已存在,略過"
fi

if [ ! -d "$RAW/AntiUAV410" ]; then
    cat <<'EOF'
[3/3] Anti-UAV410 需手動下載 (Google Drive 大檔需登入,無法腳本化):
  https://drive.google.com/file/d/1zsdazmKS3mHaEZWS2BnqbYHPEcIaH5WR/view
  (百度盤備援: https://pan.baidu.com/s/1R-L9gKIRowMgjjt52n48-g?pwd=a410 提取碼 a410)
解壓後放成:
  data/raw/AntiUAV410/{train,val,test}/<序列名>/{N.jpg..., IR_label.json}
注意: 此資料集 repo 無 LICENSE,僅供內部研究評估;商用需聯絡作者 (TPAMI 2023, doi:10.1109/TPAMI.2023.3335338)。
缺少它仍可先用 TU2U + HIT-UAV 跑通全流程 (merge 腳本會自動略過)。
EOF
else
    echo "[3/3] AntiUAV410 已存在"
fi
echo "完成。下一步: make data-convert"
