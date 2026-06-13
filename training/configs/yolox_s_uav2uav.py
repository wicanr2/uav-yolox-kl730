# YOLOX-s, UAV2UAV 4 類 (uav/person/vehicle/bicycle), 熱影像
#
# 繼承 Kneron 官方 img_norm 版 config — 關鍵差異 (相對上游 YOLOX):
#   - activation 全改 LeakyReLU(0.1) 取代 SiLU (Kneron NPU 硬體友善)
#   - img_norm: mean=128 / std=256 → 等效 x/256 - 0.5
#     (量化校正集前處理必須與此一致,見 conversion/convert_to_nef.py)
# 路徑可由環境變數覆寫: 本機 docker 用預設值;Kaggle/Colab 設 KMM_ROOT / UAV2UAV_ROOT
import os
KMM_ROOT = os.environ.get('KMM_ROOT', '/workspace/kneron-mmdetection')
UAV2UAV_ROOT = os.environ.get('UAV2UAV_ROOT', '/workspace/uav2uav')
del os  # 避免 module 物件混進 mmcv Config dict

_base_ = KMM_ROOT + '/configs/yolox/yolox_s_8x8_300e_coco_img_norm.py'

classes = ('uav', 'person', 'vehicle', 'bicycle')
data_root = UAV2UAV_ROOT + '/data/'

model = dict(bbox_head=dict(num_classes=4))

# base 的 train 是 MultiImageMixDataset(Mosaic/MixUp) 包一層 CocoDataset,
# 因此 train 要改的是 dataset 內層
train_dataset = dict(
    dataset=dict(
        classes=classes,
        ann_file=data_root + 'coco/annotations/train.json',
        img_prefix=data_root + 'raw/'))

data = dict(
    samples_per_gpu=8,   # 單卡;base 預設 8 GPU × 8
    workers_per_gpu=4,
    train=train_dataset,
    val=dict(
        classes=classes,
        ann_file=data_root + 'coco/annotations/val.json',
        img_prefix=data_root + 'raw/'),
    test=dict(
        classes=classes,
        ann_file=data_root + 'coco/annotations/test.json',
        img_prefix=data_root + 'raw/'))

# lr 線性縮放: base 0.01 對應總 batch 64;單卡 batch 8 → /8
optimizer = dict(lr=0.01 / 8)

# 資料量 (~數萬張) 不需 300 epochs;第一輪求收斂可審閱
runner = dict(max_epochs=100)
# classwise=True: 逐類別 AP。類別不平衡 (train uav:bicycle ~6:1,Kaggle stride=30 後 ~2.6:1),
# 必須分類別看,否則 bicycle/vehicle 的弱表現會被 uav 高分蓋過。
# 策略: 先看第一輪 per-class AP,若 bicycle/vehicle 明顯落後再導入 class-balanced sampling
# (YOLOX 的 MultiImageMixDataset 不易直接套 sampler,屆時再評估)。
evaluation = dict(interval=5, metric='bbox', classwise=True)
checkpoint_config = dict(interval=5)
