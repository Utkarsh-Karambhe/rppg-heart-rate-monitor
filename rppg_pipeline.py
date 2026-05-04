import cv2
import numpy as np
import time
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.signal import butter, filtfilt, detrend


@dataclass
class ChunkResult:
    index: int
    start_time: float
    end_time: float
    bpm: Optional[float]
    resp_bpm: Optional[float]
    latency_sec: float
    quality: float


# ── Face detection (runs on small frame for speed) ──────────────
def detect_face_roi(frame_bgr, face_cascade, scale=0.5) -> Optional[Tuple[int,int,int,int]]:
    small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
    roi_x = int(x + 0.25 * w)
    roi_y = int(y + 0.15 * h)
    roi_w = int(0.50 * w)
    roi_h = int(0.50 * h)
    return roi_x, roi_y, roi_w, roi_h


# ── Signal extraction (face detect every 10 frames only) ────────
def extract_green_signal(
    frames: List[np.ndarray],
    timestamps: List[float],
    face_cascade,
    detect_every: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    times, values = [], []
    last_roi = None

    for i, (frame_bgr, t) in enumerate(zip(frames, timestamps)):
        if i % detect_every == 0:
            new_roi = detect_face_roi(frame_bgr, face_cascade)
            if new_roi is not None:
                last_roi = new_roi

        roi = last_roi
        if roi is None:
            continue

        x, y, w, h = roi
        H, W = frame_bgr.shape[:2]
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        roi_bgr = frame_bgr[y : y + h, x : x + w]
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

        r_mean = float(np.mean(roi_rgb[:, :, 0]))
        g_mean = float(np.mean(roi_rgb[:, :, 1]))
        b_mean = float(np.mean(roi_rgb[:, :, 2]))

        total  = r_mean + g_mean + b_mean + 1e-6
        norm_g = g_mean / total

        times.append(t)
        values.append(norm_g)

    return np.array(times), np.array(values)


# ── Band-pass filter ─────────────────────────────────────────────
def bandpass_filter(signal, times, low_hz, high_hz):
    if len(signal) < 15:
        return None, None
    duration = times[-1] - times[0]
    if duration <= 0:
        return None, None

    n = len(signal)
    fs = n / duration

    t_uniform = np.linspace(times[0], times[-1], n)
    x = np.interp(t_uniform, times, signal)
    if not np.all(np.isfinite(x)):
        return None, None
    x = detrend(x)

    nyq  = 0.5 * fs
    low  = max(low_hz  / nyq, 0.001)
    high = min(high_hz / nyq, 0.990)
    if low >= high:
        return None, None

    b, a = butter(3, [low, high], btype="band")
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        return None, None

    try:
        x_filt = filtfilt(b, a, x)
    except ValueError:
        return None, None
    return x_filt, fs


# ── FFT peak BPM ─────────────────────────────────────────────────
def estimate_peak_bpm(x_filt, fs, low_hz, high_hz):
    n = len(x_filt)
    if n < 10:
        return None, None

    freqs   = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(x_filt))

    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band_mask):
        return None, None

    freqs_band = freqs[band_mask]
    mag_band   = fft_mag[band_mask]

    peak_idx = np.argmax(mag_band)
    f_peak   = freqs_band[peak_idx]
    quality  = float(mag_band[peak_idx] / (np.mean(mag_band) + 1e-8))
    bpm      = float(f_peak * 60.0)
    return bpm, quality


# ── Process one 5-second chunk ───────────────────────────────────
def process_chunk(chunk_index, frames, timestamps, face_cascade) -> ChunkResult:
    t0 = time.perf_counter()

    times, signal = extract_green_signal(frames, timestamps, face_cascade)

    bpm        = None
    resp_bpm   = None
    quality_hr = 0.0

    if len(times) >= 15:
        # Heart rate: 0.7–3 Hz → 42–180 BPM
        x_hr, fs_hr = bandpass_filter(signal, times, 0.7, 3.0)
        if x_hr is not None:
            bpm_est, q = estimate_peak_bpm(x_hr, fs_hr, 0.7, 3.0)
            if bpm_est is not None and 40 <= bpm_est <= 180:
                bpm        = bpm_est
                quality_hr = q if q else 0.0

        # Respiratory rate: 0.1–0.5 Hz → 6–30 breaths/min
        x_rr, fs_rr = bandpass_filter(signal, times, 0.1, 0.5)
        if x_rr is not None:
            rr_est, _ = estimate_peak_bpm(x_rr, fs_rr, 0.1, 0.5)
            if rr_est is not None and 6 <= rr_est <= 30:
                resp_bpm = rr_est

    t1 = time.perf_counter()

    return ChunkResult(
        index=chunk_index,
        start_time=timestamps[0] if timestamps else 0.0,
        end_time=timestamps[-1]  if timestamps else 0.0,
        bpm=bpm,
        resp_bpm=resp_bpm,
        latency_sec=t1 - t0,
        quality=quality_hr,
    )


# ── Aggregate overall estimates ──────────────────────────────────
def aggregate_overall(chunks: List[ChunkResult]):
    bpms = np.array([c.bpm      for c in chunks if c.bpm      is not None and c.quality > 2.0])
    rrs  = np.array([c.resp_bpm for c in chunks if c.resp_bpm is not None])

    overall_bpm = float(np.median(bpms)) if bpms.size > 0 else None
    overall_rr  = float(np.median(rrs))  if rrs.size  > 0 else None
    return overall_bpm, overall_rr


# ── Main pipeline ────────────────────────────────────────────────
def run_pipeline(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps

    print(f"\n{'='*60}")
    print(f"  rPPG Near Real-Time Pipeline")
    print(f"{'='*60}")
    print(f"  Video    : {video_path}")
    print(f"  FPS      : {fps:.1f}")
    print(f"  Duration : {duration_s:.1f}s  ({total_frames} frames)")
    print(f"  Strategy : Face detect every 10 frames, reuse ROI between")
    print(f"{'='*60}\n")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("OpenCV could not load haarcascade_frontalface_default.xml")

    chunk_results: List[ChunkResult] = []
    chunk_frames, chunk_times = [], []
    frame_idx   = 0
    chunk_index = 0
    global_start = time.perf_counter()
    first_t = None

    while True:
        ret, frame = cap.read()
        if not ret:
            if chunk_times and (chunk_times[-1] - chunk_times[0]) >= 3.0:
                result = process_chunk(chunk_index, chunk_frames, chunk_times, face_cascade)
                chunk_results.append(result)
            break

        t_video = frame_idx / fps
        if first_t is None:
            first_t = t_video
        t_rel = t_video - first_t

        chunk_frames.append(frame)
        chunk_times.append(t_rel)
        frame_idx += 1

        if chunk_times[-1] - chunk_times[0] >= 5.0:
            result = process_chunk(chunk_index, chunk_frames, chunk_times, face_cascade)
            chunk_results.append(result)
            chunk_index += 1
            chunk_frames, chunk_times = [], []

    cap.release()
    global_end = time.perf_counter()

    # ── Print table ──────────────────────────────────────────────
    print("CHUNK-LEVEL RESULTS:")
    print(f"{'Chunk':<7}{'Window':<18}{'HR (BPM)':<12}{'RR (br/min)':<15}{'Quality':<10}Latency")
    print("─" * 68)
    for c in chunk_results:
        hr_str = f"{c.bpm:.1f}"      if c.bpm      is not None else "N/A"
        rr_str = f"{c.resp_bpm:.1f}" if c.resp_bpm is not None else "N/A"
        window = f"{c.start_time:.1f}s–{c.end_time:.1f}s"
        flag   = " ✓" if c.quality > 2.0 else " (low quality)"
        print(
            f"  {c.index:02d}   {window:<18}{hr_str:<12}{rr_str:<15}"
            f"{c.quality:<10.2f}{c.latency_sec*1000:.0f} ms{flag}"
        )

    overall_bpm, overall_rr = aggregate_overall(chunk_results)
    total_time = global_end - global_start

    print(f"\n{'='*60}")
    print("OVERALL ESTIMATES  (median of ✓ quality chunks only):")
    print(f"  Heart Rate  : {f'{overall_bpm:.1f} BPM' if overall_bpm else 'N/A'}")
    print(f"  Resp. Rate  : {f'{overall_rr:.1f} breaths/min' if overall_rr else 'N/A'}")

    print(f"\nPERFORMANCE METRICS:")
    print(f"  Total runtime       : {total_time:.2f} s")
    if chunk_results:
        avg_lat = np.mean([c.latency_sec for c in chunk_results])
        max_lat = np.max( [c.latency_sec for c in chunk_results])
        print(f"  Avg chunk latency   : {avg_lat*1000:.0f} ms")
        print(f"  Max chunk latency   : {max_lat*1000:.0f} ms")
        print(f"  Real-time factor    : {5.0/avg_lat:.1f}x  (>1 = faster than real-time ✓)")
        print(f"  Valid chunks used   : {sum(1 for c in chunk_results if c.quality > 2.0)}/{len(chunk_results)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Near Real-Time rPPG Prototype")
    parser.add_argument("--video", required=True, help="Path to face video (ideally 60s)")
    args = parser.parse_args()
    run_pipeline(args.video)
