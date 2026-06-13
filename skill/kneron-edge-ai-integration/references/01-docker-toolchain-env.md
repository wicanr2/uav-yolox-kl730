# 01 · Docker / Toolchain 環境建置 (含 6 個 build 坑)

## 兩個 docker image

| image | 用途 | 來源 |
|---|---|---|
| 自建 `*-train` | 訓練、ONNX 匯出 | 自寫 Dockerfile (見下) |
| `kneron/toolchain:latest` | ONNX→優化→量化→NEF | Kneron 官方 (v0.33.x) |

主機**不裝任何 Python 套件**,全在容器內 (規則:docker first + uv venv)。

## 訓練 image Dockerfile (CPU 版;GPU 把 +cpu 換 +cu113)

```dockerfile
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential libgl1 libglib2.0-0 unzip \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python          # 坑6: 非root才進得去
RUN uv venv /opt/venv --python 3.8
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:${PATH}"
RUN uv pip install torch==1.12.1+cpu torchvision==0.13.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install mmcv-full==1.6.0 \
        -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.12.0/index.html
RUN git clone --depth 1 https://github.com/kneron/kneron-mmdetection /workspace/kneron-mmdetection
WORKDIR /workspace/kneron-mmdetection
RUN uv pip install setuptools wheel \                       # 坑2
    && uv pip install -r requirements/build.txt \
    && uv pip install --no-build-isolation -e . \           # 坑1
    && uv pip install onnx onnxoptimizer onnx-simplifier==0.4.36   # 坑3
RUN uv pip install 'yapf==0.40.1' onnxruntime               # 坑4 + onnxsim runtime 依賴
```

## 六個 build 坑 (依序會踩到)

1. **mmdet setup.py build 期 import torch** → uv 隔離 build 環境沒 torch → `--no-build-isolation`
2. `--no-build-isolation` 需 venv 內有 setuptools,uv venv 預設沒有 → 先 `uv pip install setuptools wheel`
3. **onnxsim 0.6.x 在 py3.8 無 wheel** 源碼編譯失敗 → pin `onnx-simplifier==0.4.36`
4. **新版 yapf 移除 `FormatCode(verify=)`** → mmcv 1.x Config.dump TypeError → pin `yapf==0.40.1`
5. 容器以 root 跑 → 產物變 root 所有,host 操作被拒 → `docker run -u $(id -u):$(id -g) -e HOME=/tmp`
6. **uv 把 CPython 裝在 /root/.local,非 root 進不去** (`python: command not found`) → `UV_PYTHON_INSTALL_DIR=/opt/uv-python`

額外:`onnxruntime` 要先裝,否則 onnxsim 會在執行期自己 `pip install` 而失敗。
偵錯提醒:**背景指令別接 `| tail`**,pipe 會吃掉 exit code 讓你把失敗當成功。

## toolchain 容器用法

```bash
# platform 是字串 "730";python 在 conda env onnx1.13 (base 不支援 730),非互動須用完整路徑
KTC_PY=/workspace/miniconda/envs/onnx1.13/bin/python
docker run --rm -i --cpus=8 --memory=20g --memory-swap=20g -v $PWD:/data1 \
    kneron/toolchain:latest $KTC_PY /data1/convert.py ...
```
- **heredoc 餵 python 要加 `-i`**(`docker run -i ... python - <<PY`),否則 stdin 不轉發、python 收到空程式。
- `--memory`/`--cpus` 一律加 (見 SKILL §2 記憶體當機事故)。

## 版本鎖定總表

| 項目 | 值 | 依據 |
|---|---|---|
| kneron-mmdetection | mmdetection 2.25.0 fork | repo mmdet/version.py |
| mmcv-full | 1.6.0 (允許 1.3.17–1.6.0) | mmdet __init__ assert |
| PyTorch | 1.12.1 (+cpu/+cu113) | mmcv-full 1.6.0 prebuilt 最新組合 |
| ONNX opset | 11 (export 內 assert) | pytorch2onnx_kneron.py |
| toolchain | kneron/toolchain (v0.33.x), conda env onnx1.13 | base 不支援 730 |
| yapf / onnx-simplifier | 0.40.1 / 0.4.36 | 見坑 3、4 |
