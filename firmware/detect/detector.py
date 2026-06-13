"""detect 模組對外窄介面 (PLAN §4 deep module)。

    det = Detector(nef_path, backend="sim"|"board")
    results = det.infer(frame_rgb)   # frame: HxWx3 uint8 → [(x0,y0,x1,y1,score,cls), ...]

內部藏起來的複雜度:前處理 (resize + img_norm)、NPU 推論、decode+NMS、座標還原。
呼叫端只需要知道「丟一張圖、拿一組框」。

兩個 backend:
  - sim:   用 ktc.kneron_inference 在 PC 模擬器跑 NEF (開發/驗證用,本機可跑)
  - board: 用 Kneron PLUS (kp) 在實體 KL730 跑 (待硬體;此處為介面 stub)

decode/NMS 與前處理在兩個 backend 完全共用 — 上板時只換「NPU 推論」那一塊,
其餘行為一致,降低 sim→board 的行為漂移風險。
"""
import numpy as np
from PIL import Image

from postprocess import postprocess

_INPUT = 640
_INPUT_NAME = "input"


def _preprocess(frame_rgb, size=_INPUT):
    """HxWx3 uint8 → (1,3,size,size) float32, 與訓練/量化一致的 x/256-0.5。"""
    img = Image.fromarray(frame_rgb).resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 256.0 - 0.5
    return np.transpose(arr, (2, 0, 1))[None]


class Detector:
    def __init__(self, nef_path, backend="sim", platform=730,
                 conf_thres=0.3, iou_thres=0.45, input_size=_INPUT):
        self.nef_path = nef_path
        self.backend = backend
        self.platform = platform
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = input_size
        if backend == "sim":
            import ktc  # noqa: F401 — 只在 toolchain container 內可用
            self._ktc = ktc
        elif backend == "board":
            self._init_board()
        else:
            raise ValueError(f"未知 backend: {backend}")

    def _init_board(self):
        """KL730 + Kneron PLUS 初始化 (待硬體驗證的 stub)。

        實機流程 (見 docs/kneron-npu-deploy-notes.md §1):
            import kp
            self._dev = kp.core.connect_devices(usb_port_ids=[...])
            self._model = kp.core.load_model_from_file(self._dev, self.nef_path)
        """
        raise NotImplementedError(
            "board backend 需在實體 KL730 + Kneron PLUS 環境實作;"
            "decode/NMS 已就緒 (postprocess.py),屆時只接 kp 推論輸出即可。")

    def _run_npu(self, x):
        """回傳 NEF 的 raw 輸出 list。"""
        if self.backend == "sim":
            return self._ktc.kneron_inference(
                [x], nef_file=self.nef_path,
                input_names=[_INPUT_NAME], platform=self.platform)
        # board: 收 kp 推論結果 → 反量化成 float list (待硬體)
        raise NotImplementedError

    def infer(self, frame_rgb):
        """主介面:一張 RGB 圖 → [(x0,y0,x1,y1,score,cls)],座標在原圖尺度。"""
        h0, w0 = frame_rgb.shape[:2]
        x = _preprocess(frame_rgb, self.input_size)
        outs = self._run_npu(x)
        dets = postprocess(outs, input_size=self.input_size,
                           conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        sx, sy = w0 / self.input_size, h0 / self.input_size
        return [(x0 * sx, y0 * sy, x1 * sx, y1 * sy, s, c)
                for x0, y0, x1, y1, s, c in dets]
