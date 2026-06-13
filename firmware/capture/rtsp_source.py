"""RTSP (H.265/H.264) 影像源 — 即時取流,最新幀勝出 (PLAN §4.1 capture 站延伸)。

即時辨識的核心紀律:**解碼率與推論率解耦,推論永遠拿最新幀,舊幀丟棄**。
若排隊累積 → 延遲無限增長 → 崩盤。本模組用單槽 LatestFrameBuffer 實現。

兩種解碼 backend:
  - ffmpeg 子程序 (本模組預設):rtsp → rawvideo bgr24 → stdout pipe。
    可攜、相依少 (只要 ffmpeg);軟解 H.265 吃 CPU。適合開發/地面端。
  - GStreamer (board 用,見 board_pipeline()):KL730 上接 HW 解碼器卸載 A55。

⚠️ 部署前務必確認 KL730 是否有 HW HEVC 解碼 (見 docs/architecture/rtsp-realtime-ingestion.md §2)。
"""
import subprocess
import threading

import numpy as np


class LatestFrameBuffer:
    """執行緒安全單槽緩衝:set 覆寫舊幀,get 取最新並標記已消費。即時管線的丟幀紀律核心。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0          # 已寫入的總幀數
        self._last_got = 0     # 上次 get 拿到的 seq
        self.produced = 0
        self.dropped = 0       # 被新幀覆蓋、從未被消費的幀數

    def set(self, frame):
        with self._lock:
            if self._frame is not None and self._seq > self._last_got:
                self.dropped += 1   # 舊幀還沒被取走就被覆蓋 → 丟棄
            self._frame = frame
            self._seq += 1
            self.produced += 1

    def get_latest(self):
        """回傳 (frame, seq) 或 (None, 0)。同一幀不會被取兩次。"""
        with self._lock:
            if self._frame is None or self._seq == self._last_got:
                return None, 0
            self._last_got = self._seq
            f = self._frame
            return f, self._seq


def board_pipeline(rtsp_url, width=640, height=512, hwdec=True):
    """KL730 board 用的 GStreamer pipeline 字串 (接 HW 解碼器卸載 A55)。

    hwdec=True 時用平台 HW HEVC 解碼器 (名稱依 BSP,如 v4l2slh265dec / kmssink 系);
    fallback avdec_h265 為軟解。實際 element 名以 KL730 BSP 的 gst-inspect 為準。
    """
    dec = "v4l2slh265dec" if hwdec else "avdec_h265"
    return (
        f"rtspsrc location={rtsp_url} latency=50 protocols=udp ! "
        f"rtph265depay ! h265parse ! {dec} ! "
        f"videoscale ! videoconvert ! video/x-raw,format=BGR,width={width},height={height} ! "
        f"appsink drop=true max-buffers=1 sync=false"   # drop=true + max-buffers=1 = 硬體層也最新幀勝出
    )


class RtspSource:
    """ffmpeg 子程序解碼 RTSP → BGR 幀 → LatestFrameBuffer (reader thread)。"""

    def __init__(self, rtsp_url, width=640, height=512, transport="udp", probe_timeout=5):
        self.url = rtsp_url
        self.w, self.h = width, height
        self.transport = transport
        self.probe_timeout = probe_timeout
        self.buf = LatestFrameBuffer()
        self._proc = None
        self._thread = None
        self._stop = threading.Event()

    def _ffmpeg_cmd(self):
        # rtsp/rtp/sdp/file → 縮放到模型輸入 → rawvideo bgr24 → stdout
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-fflags", "nobuffer", "-flags", "low_delay"]   # 低延遲:不要 ffmpeg 端 buffer
        if self.url.startswith("rtsp://"):
            cmd += ["-rtsp_transport", self.transport]
        else:
            cmd += ["-protocol_whitelist", "file,udp,rtp,crypto,data"]  # 允許 .sdp/RTP 輸入
        cmd += ["-i", self.url, "-vf", f"scale={self.w}:{self.h}",
                "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        return cmd

    def _reader(self):
        frame_bytes = self.w * self.h * 3
        self._proc = subprocess.Popen(self._ffmpeg_cmd(),
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        while not self._stop.is_set():
            raw = self._proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break   # 流結束或斷線
            frame = np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 3)
            self.buf.set(frame)
        if self._proc:
            self._proc.terminate()

    def start(self):
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def read_latest(self):
        return self.buf.get_latest()

    def stats(self):
        return {"produced": self.buf.produced, "dropped": self.buf.dropped}

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=2)
