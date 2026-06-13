"""YOLOX 後處理:把 NEF 的 9 個 raw 輸出 decode 成 bounding box + NMS。

KL730 上這段跑在 A55 (CPU) — NPU 只算到 raw 張量。純 numpy 實作,
與 kneron-mmdetection YOLOXHead 的 decode 語意一致:

  NEF 輸出 (--skip-postprocess, 無 sigmoid):
    cls: (1, num_cls, H, W)  raw logit   ← 需 sigmoid
    reg: (1, 4,       H, W)  grid 偏移+log 尺度  ← 需 decode
    obj: (1, 1,       H, W)  raw logit   ← 需 sigmoid
  三個尺度 (stride 8/16/32 對應 640 輸入的 80/40/20)。

  decode (cell (gx,gy), stride s):
    cx = (reg_x + gx) * s ;  cy = (reg_y + gy) * s
    w  = exp(reg_w) * s   ;  h  = exp(reg_h) * s
    score = sigmoid(obj) * sigmoid(cls_c)

不依賴 torch,可在 toolchain container / 板上 Python 直接跑。
"""
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _group_outputs(outputs):
    """依 channel 數把 9 個輸出分成 cls/reg/obj,再依空間大小配對成尺度。

    不靠固定 index — 對輸出順序變動穩健。
    回傳 [(cls, reg, obj, stride), ...] 由大尺度到小尺度。
    """
    arrs = [np.asarray(o) for o in outputs]
    arrs = [a[0] if a.ndim == 4 else a for a in arrs]  # 去 batch 維 → (C,H,W)
    cls, reg, obj = {}, {}, {}
    for a in arrs:
        c, h, w = a.shape
        key = (h, w)
        if c == 4:
            reg[key] = a
        elif c == 1:
            obj[key] = a
        else:
            cls[key] = a  # num_classes
    levels = []
    for key in sorted(cls, key=lambda hw: -hw[0]):  # 大 → 小
        levels.append((cls[key], reg[key], obj[key], key))
    return levels


def decode(outputs, input_size=640, conf_thres=0.3):
    """回傳 (boxes_xyxy[N,4], scores[N], classes[N]),座標在 input_size 尺度。"""
    boxes, scores, classes = [], [], []
    for cls_l, reg_l, obj_l, (h, w) in _group_outputs(outputs):
        stride = input_size // h
        # grid
        gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        cx = (reg_l[0] + gx) * stride
        cy = (reg_l[1] + gy) * stride
        bw = np.exp(reg_l[2]) * stride
        bh = np.exp(reg_l[3]) * stride
        obj_s = _sigmoid(obj_l[0])                     # (H,W)
        cls_s = _sigmoid(cls_l)                        # (num_cls,H,W)
        # 每格取最高分類別
        best_cls = np.argmax(cls_s, axis=0)            # (H,W)
        best_cls_s = np.max(cls_s, axis=0)             # (H,W)
        score = obj_s * best_cls_s                     # (H,W)
        mask = score >= conf_thres
        if not mask.any():
            continue
        x0 = (cx - bw / 2)[mask]
        y0 = (cy - bh / 2)[mask]
        x1 = (cx + bw / 2)[mask]
        y1 = (cy + bh / 2)[mask]
        boxes.append(np.stack([x0, y0, x1, y1], axis=1))
        scores.append(score[mask])
        classes.append(best_cls[mask])
    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)


def nms(boxes, scores, iou_thres=0.45):
    """標準 NMS,回傳保留的 index。"""
    if len(boxes) == 0:
        return []
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x1 - x0) * (y1 - y0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx0 = np.maximum(x0[i], x0[order[1:]])
        yy0 = np.maximum(y0[i], y0[order[1:]])
        xx1 = np.minimum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        inter = np.maximum(0, xx1 - xx0) * np.maximum(0, yy1 - yy0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def postprocess(outputs, input_size=640, conf_thres=0.3, iou_thres=0.45, per_class=True):
    """完整後處理:decode → NMS。回傳 list of (x0,y0,x1,y1,score,cls)。"""
    boxes, scores, classes = decode(outputs, input_size, conf_thres)
    if len(boxes) == 0:
        return []
    keep_all = []
    if per_class:
        for c in np.unique(classes):
            idx = np.where(classes == c)[0]
            for k in nms(boxes[idx], scores[idx], iou_thres):
                keep_all.append(idx[k])
    else:
        keep_all = nms(boxes, scores, iou_thres)
    return [(*boxes[i], float(scores[i]), int(classes[i])) for i in keep_all]
