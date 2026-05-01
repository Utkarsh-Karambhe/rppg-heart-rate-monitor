<div align="center">

<br/>

```
██████╗ ██████╗ ██████╗  ██████╗
██╔══██╗██╔══██╗██╔══██╗██╔════╝
██████╔╝██████╔╝██████╔╝██║  ███╗
██╔══██╗██╔═══╝ ██╔═══╝ ██║   ██║
██║  ██║██║     ██║     ╚██████╔╝
╚═╝  ╚═╝╚═╝     ╚═╝      ╚═════╝
```

# Remote Photoplethysmography · Heart Rate Monitor

**Measure your heart — no contact required.**

Estimates heart rate and respiratory rate from a face video using computer vision and signal processing — running 17–21× faster than real time.

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![SciPy](https://img.shields.io/badge/SciPy-Signal%20Processing-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

<br/>

<!-- Replace demo.gif with your actual GIF/screenshot -->
<img src="demo.gif" alt="rPPG Demo" width="780" style="border-radius: 12px;" />

<br/><br/>

</div>

---

## What Is rPPG?

Remote photoplethysmography (rPPG) detects subtle colour changes in skin caused by blood flow — invisible to the eye, but measurable through a camera. Every heartbeat pushes blood through your face, slightly changing the light reflected off it. This system captures those micro-variations from the **green channel** of your camera feed and extracts physiological signals using signal processing.

No wearables. No contact. Just your face and a camera.

---

## Features

| | |
|---|---|
| ❤️ **Heart Rate** | BPM estimation per 5-second chunk |
| 🌬️ **Respiratory Rate** | Breaths/min extracted from the same signal |
| ⚡ **Near Real-Time** | 17–21× faster than real-time processing |
| 📊 **Live Dashboard** | Interactive Streamlit + Plotly visualisation |
| 🎯 **Signal Quality Score** | Per-chunk reliability indicator |
| 🧍 **Optimised Face Tracking** | Detection runs every 10 frames to cut overhead |

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         rPPG Pipeline                               │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  Video   │───▶│     Face     │───▶│     RGB      │              │
│  │  Input   │    │  Detection   │    │  Extraction  │              │
│  └──────────┘    │ (Haarcascade)│    │  (ROI mean)  │              │
│                  └──────────────┘    └──────┬───────┘              │
│                                             │                       │
│                  ┌──────────────────────────▼──────────────────┐   │
│                  │              Signal Processing               │   │
│                  │                                             │   │
│                  │  Detrend → Interpolate → Bandpass Filter    │   │
│                  │                                             │   │
│                  │   HR band:  0.7 – 3.0 Hz  (42 – 180 BPM)  │   │
│                  │   RR band:  0.1 – 0.5 Hz  (6 – 30 br/min) │   │
│                  └──────────────────┬──────────────────────────┘   │
│                                     │                               │
│                  ┌──────────────────▼──────────────────────────┐   │
│                  │           FFT Frequency Analysis             │   │
│                  │   Peak frequency → BPM / Breaths per min    │   │
│                  └──────────────────┬──────────────────────────┘   │
│                                     │                               │
│                  ┌──────────────────▼──────────────────────────┐   │
│                  │         Streamlit Dashboard Output           │   │
│                  │   Live BPM · RR · Quality · Performance     │   │
│                  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step

**1 · Face Detection**
Haarcascade classifier locates the face ROI. Detection runs every 10 frames — the bounding box is reused in between for speed.

**2 · Signal Extraction**
Mean RGB values are sampled from the face region each frame. The green channel carries the strongest rPPG signal (peak absorption by haemoglobin).

**3 · Signal Processing**
The raw trace is detrended to remove drift, resampled to a uniform time grid, then passed through a zero-phase Butterworth bandpass filter — separately tuned for heart rate and respiratory rate.

**4 · Frequency Analysis**
FFT is applied on each 5-second window. The dominant peak frequency maps directly to BPM or breaths/min.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Computer Vision | OpenCV |
| Signal Processing | NumPy · SciPy |
| Frontend / UI | Streamlit |
| Visualisation | Plotly |
| Data Handling | Pandas |

---

## Project Structure

```
rppg-heart-rate-monitor/
│
├── app.py                  # Streamlit UI & dashboard
├── rppg_pipeline.py        # Core signal processing logic
├── demo.gif                # Demo animation (README preview)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- A face video (MP4, AVI, etc.) or webcam

### Installation

```bash
git clone https://github.com/Utkarsh-Karambhe/rppg-heart-rate-monitor.git
cd rppg-heart-rate-monitor
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser, upload a video, and watch the dashboard populate in real time.

---

## Output & Dashboard

The dashboard displays:

- **Live BPM graph** — heart rate across each 5-second chunk
- **Respiratory rate trend** — breaths/min over time  
- **Signal quality bar chart** — per-chunk reliability score
- **Overall median BPM** — aggregate summary
- **Performance panel** — chunk latency and real-time factor

---

## Performance

| Metric | Value |
|---|---|
| Chunk latency | ~250 – 400 ms |
| Real-time factor | ~17 – 21× |
| Optimal conditions | Stable lighting, frontal face |

---

## Limitations

- Sensitive to uneven or changing lighting
- Requires a frontal, clearly visible face
- Motion (head movement, talking) reduces signal quality
- Not validated for clinical or medical use

---

## Roadmap

- [ ] Live webcam streaming support
- [ ] Deep learning–based face tracking (MediaPipe / dlib)
- [ ] Mobile deployment (WASM or native)
- [ ] ML-based noise reduction
- [ ] Multi-face simultaneous estimation
- [ ] Clinical-grade accuracy benchmarking

---

## Local GIF Conversion (for GitHub preview)

GitHub does not autoplay `.mp4` files in READMEs. Convert your demo to GIF first:

| Platform | Tool |
|---|---|
| macOS | [Kap](https://getkap.co/) |
| Windows | [ScreenToGif](https://www.screentogif.com/) |
| Any | `ffmpeg -i demo.mp4 -vf "fps=15,scale=780:-1" demo.gif` |

Then commit:

```bash
git add demo.gif
git commit -m "Add demo GIF for README preview"
git push
```

---

## Author

**Utkarsh Karambhe**  
B.Tech CSE (Data Science) · Nagpur, India  
[GitHub](https://github.com/Utkarsh-Karambhe)

---

<div align="center">

If this project helped you, consider giving it a ⭐ — it genuinely helps with visibility.

<br/>

*Built with signal processing, caffeine, and curiosity about what a camera can silently know about you.*

</div>
