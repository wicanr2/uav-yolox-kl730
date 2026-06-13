#!/usr/bin/env python3
"""RTSP/RTP H.265 ingestion 自我驗證 (本機 ffmpeg loopback,不需真攝影機/板)。

跑法: python3 firmware/test_rtsp_loopback.py
驗證: ① LatestFrameBuffer 丟幀紀律 ② 真 H.265-over-RTP/UDP 解碼→幀→緩衝全鏈。
依賴: ffmpeg (含 libx265 編 + hevc 解)。
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "capture"))
import numpy as np
from rtsp_source import LatestFrameBuffer, RtspSource


def test_buffer_drop():
    buf = LatestFrameBuffer()
    for i in range(5):
        buf.set(np.full((2, 2, 3), i, np.uint8))
    f, seq = buf.get_latest()
    assert seq == 5 and int(f[0, 0, 0]) == 4
    for i in range(5, 10):
        buf.set(np.full((2, 2, 3), i, np.uint8))
    f, seq = buf.get_latest()
    assert seq == 10 and int(f[0, 0, 0]) == 9
    assert buf.get_latest()[0] is None          # 不重取
    assert buf.produced == 10 and buf.dropped == 8
    print("[1] 最新幀緩衝丟幀紀律 PASS (產10/取2最新/丟8)")


def test_h265_rtp_loopback():
    sdp = "/tmp/_uav_stream.sdp"
    srv = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-re",
         "-f", "lavfi", "-i", "testsrc=size=640x512:rate=30",
         "-c:v", "libx265", "-preset", "ultrafast",
         "-x265-params", "keyint=15:log-level=none",
         "-f", "rtp", "-sdp_file", sdp, "rtp://127.0.0.1:5004"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(2.0)
        src = RtspSource(sdp, 640, 512).start()
        t0, seen, sample = time.time(), 0, None
        while time.time() - t0 < 4.0:
            f, _ = src.read_latest()
            if f is not None:
                seen += 1
                sample = f
            time.sleep(0.03)
        src.stop()
        st = src.stats()
        assert st["produced"] > 10, f"解碼太少 {st['produced']}"
        assert sample is not None and sample.shape == (512, 640, 3)
        assert sample.std() > 1
        print(f"[2] H.265/RTP loopback PASS (解碼{st['produced']}/取{seen}/丟{st['dropped']})")
    finally:
        srv.terminate()
        Path(sdp).unlink(missing_ok=True)


if __name__ == "__main__":
    test_buffer_drop()
    test_h265_rtp_loopback()
    print("ALL PASS — RTSP/RTP H.265 ingestion 架構驗證通過")
