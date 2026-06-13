"""即時 UAV 辨識主迴圈:RTSP → 最新幀 → NPU 推論 → 追蹤 → 輸出。

體現即時紀律:每輪拉「最新幀」(舊幀已被 LatestFrameBuffer 丟掉),
推論 fps 即有效偵測 fps,延遲有界。偵測之間用 tracker 維持 UAV 身分、容忍漏幀。

detector / tracker / on_result 皆可注入 (duck-typed) → 無 NPU/硬體也能跑骨架測試。
實機:detector = firmware.detect.detector.Detector(backend="board")。
"""
import time

from capture.rtsp_source import RtspSource


class PassThroughTracker:
    """佔位 tracker。實機換 ByteTrack:跨幀關聯、容忍漏偵測、給穩定 UAV track id。"""

    def update(self, detections):
        return detections


def run(rtsp_url, detector, tracker=None, on_result=None,
        width=640, height=512, max_frames=None, idle_sleep=0.002):
    """detector: 物件需有 .infer(frame_bgr) → [(x0,y0,x1,y1,score,cls), ...]
       tracker:  .update(dets) → tracks (預設 PassThrough)
       on_result(seq, tracks, latency_ms): 每次推論完回呼 (輸出 MAVLink/RTSP overlay 等)
    """
    tracker = tracker or PassThroughTracker()
    src = RtspSource(rtsp_url, width, height).start()
    n_infer = 0
    try:
        while max_frames is None or n_infer < max_frames:
            frame, seq = src.read_latest()
            if frame is None:
                time.sleep(idle_sleep)        # 無新幀,讓出 CPU (不忙等)
                continue
            t0 = time.perf_counter()
            dets = detector.infer(frame)       # NPU 推論 + decode + NMS (跑在 A55)
            tracks = tracker.update(dets)
            latency_ms = (time.perf_counter() - t0) * 1000
            n_infer += 1
            if on_result:
                on_result(seq, tracks, latency_ms)
    finally:
        src.stop()
    st = src.stats()
    st["inferred"] = n_infer
    return st   # {produced, dropped, inferred} — dropped 高 = 串流快過推論 (正常的丟幀)


if __name__ == "__main__":
    import sys
    # 煙霧用:假 detector (不需 NPU),驗證 capture→loop→輸出串得起來
    class DummyDetector:
        def infer(self, frame):
            return [(10, 10, 50, 50, 0.9, 0)]   # 固定回一個假框
    url = sys.argv[1] if len(sys.argv) > 1 else "rtsp://127.0.0.1:8554/test"
    stats = run(url, DummyDetector(),
                on_result=lambda seq, t, ms: print(f"seq={seq} tracks={len(t)} {ms:.1f}ms"),
                max_frames=int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    print("stats:", stats)
