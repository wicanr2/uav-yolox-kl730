# UAV2UAV 訓練環境: kneron-mmdetection (mmdet 2.25.0 fork) + YOLOX
#
# 版本鎖定依據 (2026-06-11 查證自官方 repo / 文件):
#   - kneron-mmdetection 基於 mmdetection 2.25.0,mmcv-full 限 >=1.3.17, <=1.6.0
#   - ONNX export 鎖 opset 11 (pytorch2onnx_kneron.py 內有 assert)
#
# 本機為 CPU-only (2026-06-11 決策):裝 CPU wheel,負責資料轉檔、smoke 訓練、
# ONNX export 與 NEF 轉換驗證;正式訓練在雲端 GPU 執行 (見 README「CPU 開發策略」)。
# 若改 GPU 機:把下方兩個 index 換成 cu113 (torch 1.12.1+cu113 / mmcv cu113),其餘不變。
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates build-essential \
        libgl1 libglib2.0-0 unzip \
    && rm -rf /var/lib/apt/lists/*

# uv 管理 venv (規則: Python 一律 docker 內 uv venv,不污染系統環境)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
# Python 裝到 /opt 而非 /root:容器以非 root UID 執行時才能 traverse symlink
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv venv /opt/venv --python 3.8
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:${PATH}"

RUN uv pip install torch==1.12.1+cpu torchvision==0.13.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install mmcv-full==1.6.0 \
        -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.12.0/index.html

# kneron-mmdetection (官方 YOLOX → Kneron 流程的 fork)
RUN git clone --depth 1 https://github.com/kneron/kneron-mmdetection /workspace/kneron-mmdetection
WORKDIR /workspace/kneron-mmdetection
# mmdet setup.py 會在 build 期 import torch → 必須關掉 uv 的 build 隔離;
# 而 no-build-isolation 需要 venv 內有 setuptools/wheel (uv venv 預設不含)
RUN uv pip install setuptools wheel \
    && uv pip install -r requirements/build.txt \
    && uv pip install --no-build-isolation -e . \
    && uv pip install onnx onnxoptimizer onnx-simplifier==0.4.36
    # onnx-simplifier pin 0.4.36: 0.5.x 依賴的 onnxsim 0.6.x 在 py3.8 無 wheel,源碼編譯會失敗

# yapf pin: mmcv 1.x Config.dump 呼叫 FormatCode(verify=) — yapf >=0.41 移除該參數會 TypeError
# onnxruntime: onnxsim 的 runtime 依賴 (缺了它會嘗試自己 pip install 而失敗)
RUN uv pip install 'yapf==0.40.1' onnxruntime

WORKDIR /workspace/uav2uav
