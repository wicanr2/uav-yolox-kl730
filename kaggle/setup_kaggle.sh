#!/usr/bin/env bash
# Kaggle 一鍵上架腳本 — 在 docker 內跑 kaggle CLI,不污染主機 (符合專案規則)
#
# 你只需要做的事 (不可自動化的部分):
#   1. 註冊 Kaggle 帳號 + 手機驗證 (開 GPU/Internet 的前提)
#   2. kaggle.com → Account → Create New API Token → 下載 kaggle.json
#   3. 把 kaggle.json 放到 ~/.kaggle/kaggle.json
#   4. 執行: bash kaggle/setup_kaggle.sh <你的kaggle帳號名>
#
# 之後本腳本自動: 建 code dataset、(可選)建 antiuav410 dataset、推 notebook kernel。
set -euo pipefail
cd "$(dirname "$0")/.."

USER="${1:?用法: bash kaggle/setup_kaggle.sh <kaggle_username> [--with-antiuav]}"
WITH_ANTI="${2:-}"
KAGGLE="docker run --rm -v $HOME/.kaggle:/root/.kaggle -v $(pwd):/work -w /work \
        ghcr.io/astral-sh/uv:python3.11-bookworm-slim bash -c"

[ -f "$HOME/.kaggle/kaggle.json" ] || { echo "缺 ~/.kaggle/kaggle.json,見本檔開頭步驟 2-3"; exit 1; }

run_kaggle() { $KAGGLE "pip install -q kaggle 2>/dev/null; chmod 600 /root/.kaggle/kaggle.json; kaggle $*"; }

echo "=== 1. 重新打包最新程式碼 ==="
make kaggle-pack

echo "=== 2. 建 code dataset (uav2uav-code) ==="
mkdir -p kaggle/_code_ds && cp kaggle/uav2uav-code.zip kaggle/_code_ds/
cat > kaggle/_code_ds/dataset-metadata.json <<EOF
{ "title": "uav2uav-code", "id": "$USER/uav2uav-code", "licenses": [{"name": "CC0-1.0"}] }
EOF
run_kaggle "datasets create -p /work/kaggle/_code_ds -q || kaggle datasets version -p /work/kaggle/_code_ds -m 'update' -q"

if [ "$WITH_ANTI" = "--with-antiuav" ]; then
  echo "=== 3. 建 antiuav410 dataset (8.8GB,上傳慢) ==="
  mkdir -p kaggle/_anti_ds && cp datasets/Anti-UAV410.zip kaggle/_anti_ds/
  cat > kaggle/_anti_ds/dataset-metadata.json <<EOF
{ "title": "antiuav410", "id": "$USER/antiuav410", "licenses": [{"name": "other"}] }
EOF
  run_kaggle "datasets create -p /work/kaggle/_anti_ds -q"
else
  echo "=== 3. 跳過 Anti-UAV410 (首跑建議: 先用 TU2U+HIT-UAV 訓出可用模型,之後再加) ==="
fi

echo "=== 4. 推 notebook kernel ==="
mkdir -p kaggle/_kernel && cp kaggle/train_uav2uav_kaggle.ipynb kaggle/_kernel/
DS_SOURCES="\"$USER/uav2uav-code\""
[ "$WITH_ANTI" = "--with-antiuav" ] && DS_SOURCES="$DS_SOURCES, \"$USER/antiuav410\""
cat > kaggle/_kernel/kernel-metadata.json <<EOF
{
  "id": "$USER/uav2uav-train",
  "title": "uav2uav-train",
  "code_file": "train_uav2uav_kaggle.ipynb",
  "language": "python", "kernel_type": "notebook",
  "enable_gpu": true, "enable_internet": true,
  "dataset_sources": [$DS_SOURCES]
}
EOF
run_kaggle "kernels push -p /work/kaggle/_kernel"

echo
echo "完成。到 kaggle.com/$USER/uav2uav-train 開啟,確認 GPU/Internet 已開,Run All。"
echo "跑完: kaggle kernels output $USER/uav2uav-train -p work_dirs/yolox_s_uav2uav/"
