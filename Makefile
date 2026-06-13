# UAV2UAV — YOLOX 訓練 → KL730 NEF 全流程
# 本機 CPU-only:訓練相關目標分「smoke (本機可跑)」與「train (雲端 GPU 用)」
# 規則: 所有 Python 都在 docker 內執行 (訓練 image 用 uv venv;轉換用官方 toolchain)
IMAGE      := uav2uav-train
TOOLCHAIN  := kneron/toolchain:latest
PWD_       := $(shell pwd)
CFG        := /workspace/uav2uav/training/configs/yolox_s_uav2uav.py
WORK_DIR   := /workspace/uav2uav/work_dirs/yolox_s_uav2uav
ANN        := /workspace/uav2uav/data/coco/annotations
# 以本機 UID 跑容器,避免產物變 root 所有;HOME=/tmp 讓 torch/matplotlib cache 可寫
RUN        := docker run --rm --cpus=8 -u $(shell id -u):$(shell id -g) -e HOME=/tmp \
	-v $(PWD_):/workspace/uav2uav $(IMAGE)

.PHONY: help env-build data-download data-convert smoke train-smoke train \
        zoo-export export-onnx nef nef-zoo kaggle-pack

# 打包程式碼給 Kaggle dataset (排除資料/產物/大 zip)
kaggle-pack:
	cd $(PWD_) && zip -qr kaggle/uav2uav-code.zip \
		datasets training conversion Makefile CONTEXT.md PLAN.md README.md docs \
		-x 'datasets/Anti-UAV410.zip' && ls -lh kaggle/uav2uav-code.zip

help:
	@echo "── 本機 (CPU) ──────────────────────────────────────────"
	@echo "make env-build      建訓練 docker image (CPU wheel)"
	@echo "make data-download  下載 TU2U + HIT-UAV (Anti-UAV410 印手動指引)"
	@echo "make data-convert   三來源 → COCO → merge + 產 smoke 子集"
	@echo "make smoke          資料/設定能被 mmdet 載入的快速檢查"
	@echo "make train-smoke    1 epoch × 200 張子集 CPU 訓練 (驗證訓練管線)"
	@echo "make zoo-export     下載 Kneron Model Zoo 預訓練 YOLOX-s → ONNX (不需訓練)"
	@echo "make nef-zoo        Zoo ONNX → KL730 NEF (驗證整條轉換鏈, 不需 GPU)"
	@echo "make export-onnx    自訓 ckpt → ONNX"
	@echo "make nef            自訓 ONNX → KL730 NEF"
	@echo "── 雲端 GPU (Colab/Kaggle, 見 README) ──────────────────"
	@echo "make train          正式訓練 (在 GPU 機上跑;本機跑會極慢)"

env-build:
	docker build -t $(IMAGE) training/

data-download:
	bash datasets/download.sh

data-convert:
	$(RUN) bash -c "cd /workspace/uav2uav/datasets && \
		python convert_tu2u_to_coco.py --root ../data/raw/ThermalUAV2UAV_Dataset --out-dir ../data/coco/annotations && \
		python convert_hituav_to_coco.py --root ../data/raw/HIT-UAV --out-dir ../data/coco/annotations && \
		python convert_antiuav410_to_coco.py --root ../data/raw/AntiUAV410 --out-dir ../data/coco/annotations && \
		python merge_coco.py --ann-dir ../data/coco/annotations && \
		python make_subset.py --ann-dir ../data/coco/annotations"

# 用 mmdet 載一次資料集確認 json/路徑/類別都對
smoke:
	$(RUN) python -c "\
	from mmdet.datasets import build_dataset; \
	from mmcv import Config; \
	cfg = Config.fromfile('$(CFG)'); \
	ds = build_dataset(cfg.data.val); \
	print('val OK:', len(ds), 'images,', ds.CLASSES)"

# CPU 1 epoch 小子集:驗證 loss 會下降、checkpoint 會產出 (估 10–30 分鐘)
train-smoke:
	$(RUN) bash -c "cd /workspace/kneron-mmdetection && CUDA_VISIBLE_DEVICES=-1 \
		python tools/train.py $(CFG) \
		--work-dir /workspace/uav2uav/work_dirs/smoke \
		--cfg-options runner.max_epochs=1 data.samples_per_gpu=2 data.workers_per_gpu=2 \
		data.train.dataset.ann_file=$(ANN)/train_subset.json \
		data.val.ann_file=$(ANN)/val_subset.json \
		evaluation.interval=1 custom_hooks.0.num_last_epochs=1"

# 正式訓練 — 設計給雲端 GPU 機 (本機 CPU 跑完要數週,不建議)
train:
	$(RUN) bash -c "cd /workspace/kneron-mmdetection && \
		python tools/train.py $(CFG) --work-dir $(WORK_DIR)"

# ── 不需訓練的轉換鏈驗證:官方 COCO 80 類預訓練權重 ──────────────
zoo-export:
	mkdir -p artifacts work_dirs/zoo
	$(RUN) bash -c "cd /workspace/kneron-mmdetection && \
		{ [ -f /workspace/uav2uav/work_dirs/zoo/latest.zip ] || wget -O /workspace/uav2uav/work_dirs/zoo/latest.zip \
			https://github.com/kneron/Model_Zoo/raw/main/mmdetection/yolox_s/latest.zip ; } && \
		unzip -o /workspace/uav2uav/work_dirs/zoo/latest.zip -d /workspace/uav2uav/work_dirs/zoo && \
		python tools/deployment/pytorch2onnx_kneron.py \
		configs/yolox/yolox_s_8x8_300e_coco_img_norm.py \
		\$$(ls /workspace/uav2uav/work_dirs/zoo/*.pth | head -1) \
		--output-file /workspace/uav2uav/artifacts/yolox_s_coco_zoo.onnx \
		--skip-postprocess --shape 640 640 || true; \
		python -c 'import onnx; m=onnx.load(\"/workspace/uav2uav/artifacts/yolox_s_coco_zoo.onnx\"); onnx.checker.check_model(m); print(\"ONNX OK (export 腳本尾段的舊版優化器 crash 可忽略,kneronnxopt 會接手)\")'"

# toolchain 的 python 在 conda env onnx1.13 內 (唯一支援 730 的環境),非互動執行需用完整路徑
KTC_PY := /workspace/miniconda/envs/onnx1.13/bin/python
# [事故記錄 2026-06-11] mix 模式量化 threads=8 × 200 張把 30GB RAM 吃爆 → 主機 swap thrash 當機。
# 對策: 容器硬上限 20g (爆了 kill 容器不拖垮主機) + threads=4 + 校正 100 張。
TOOLCHAIN_RUN := docker run --rm --cpus=8 --memory=20g --memory-swap=20g -v $(PWD_):/data1 $(TOOLCHAIN)

nef-zoo:
	$(TOOLCHAIN_RUN) \
		$(KTC_PY) /data1/conversion/convert_to_nef.py \
		--onnx /data1/artifacts/yolox_s_coco_zoo.onnx \
		--calib-dir /data1/data/raw/ThermalUAV2UAV_Dataset/val/images \
		--out-dir /data1/artifacts --threads 4 --num 100 --range-method mmse --check

export-onnx:
	mkdir -p artifacts
	$(RUN) bash -c "cd /workspace/kneron-mmdetection && \
		python tools/deployment/pytorch2onnx_kneron.py $(CFG) \
		$(WORK_DIR)/latest.pth \
		--output-file /workspace/uav2uav/artifacts/yolox_s_uav2uav.onnx \
		--skip-postprocess --shape 640 640 || true; \
		python -c 'import onnx; m=onnx.load(\"/workspace/uav2uav/artifacts/yolox_s_uav2uav.onnx\"); onnx.checker.check_model(m); print(\"ONNX OK (export 腳本尾段的舊版優化器 crash 可忽略,kneronnxopt 會接手)\")'"

nef:
	$(TOOLCHAIN_RUN) \
		$(KTC_PY) /data1/conversion/convert_to_nef.py \
		--onnx /data1/artifacts/yolox_s_uav2uav.onnx \
		--calib-dir /data1/data/raw/ThermalUAV2UAV_Dataset/val/images \
		--out-dir /data1/artifacts --threads 4 --num 100 --range-method mmse --check
