<div align="center">

<br/>

<img src="https://em-content.zobj.net/source/apple/391/anatomical-heart_1fac0.png" width="80" alt="heart" />

<br/>

# Remote Photoplethysmography · Heart Rate Monitor

**Measure your heart — no contact required.**

Estimates heart rate and respiratory rate from a face video using<br/>
computer vision and signal processing — running **17–21× faster** than real time.

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![SciPy](https://img.shields.io/badge/SciPy-Signal_Processing-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

<img src="demo.gif" alt="rPPG Demo" width="760" style="border-radius: 12px;" />

<br/><br/>

</div>

---

## 🧬 What Is rPPG?

Remote photoplethysmography (rPPG) detects subtle colour changes in skin caused by blood flow — invisible to the naked eye, but measurable through any standard camera. Every heartbeat pushes blood through your face, slightly altering the light reflected off your skin.

This system captures those micro-variations from the **green channel** of your camera feed and extracts physiological signals using band-limited signal processing and FFT frequency analysis.

> [!NOTE]
> No wearables. No contact. Just your face and a camera.

---

## ✨ Features

| | Feature | Description |
|:---:|---|---|
| ❤️ | **Heart Rate** | BPM estimation per 5-second sliding window |
| 🌬️ | **Respiratory Rate** | Breaths/min extracted from the same rPPG signal |
| ⚡ | **Near Real-Time** | Processes 17–21× faster than video duration |
| 📊 | **Live Dashboard** | Interactive Streamlit + Plotly visualisation |
| 🎯 | **Signal Quality Score** | Per-chunk reliability indicator |
| 🧍 | **Optimised Face Tracking** | Haarcascade runs every 10 frames to cut overhead |

---

## 🔬 How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         rPPG Pipeline                               │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Video   │───▶│     Face     │───▶│     RGB      │               │
│  │  Input   │    │  Detection   │    │  Extraction  │               │
│  └──────────┘    │ (Haarcascade)│    │  (ROI mean)  │               │
│                  └──────────────┘    └──────┬───────┘               │
│                                             │                       │
│                  ┌──────────────────────────▼──────────────────┐    │
│                  │              Signal Processing              │    │
│                  │                                             │    │
│                  │  Detrend → Interpolate → Bandpass Filter    │    │
│                  │                                             │    │
│                  │   HR band:  0.7 – 3.0 Hz  (42 – 180 BPM)    │    │
│                  │   RR band:  0.1 – 0.5 Hz  (6 – 30 br/min)   │    │
│                  └──────────────────┬──────────────────────────┘    │
│                                     │                               │
│                  ┌──────────────────▼──────────────────────────┐    │
│                  │           FFT Frequency Analysis            │    │
│                  │   Peak frequency → BPM / Breaths per min    │    │
│                  └──────────────────┬──────────────────────────┘    │
│                                     │                               │
│                  ┌──────────────────▼──────────────────────────┐    │
│                  │         Streamlit Dashboard Output          │    │
│                  │   Live BPM · RR · Quality · Performance     │    │
│                  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

<details>
<summary><b>📖 Step-by-Step Breakdown</b></summary>

<br/>

**1 · Face Detection**
Haarcascade classifier locates the face ROI. Detection runs every 10 frames — the bounding box is reused in between for speed.

**2 · Signal Extraction**
Mean RGB values are sampled from the face region each frame. The green channel carries the strongest rPPG signal due to peak absorption by haemoglobin.

**3 · Signal Processing**
The raw trace is detrended to remove drift, resampled to a uniform time grid, then passed through a zero-phase Butterworth bandpass filter — separately tuned for heart rate and respiratory rate.

**4 · Frequency Analysis**
FFT is applied on each 5-second window. The dominant peak frequency maps directly to BPM or breaths/min.

</details>

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Computer Vision | OpenCV |
| Signal Processing | NumPy · SciPy |
| Frontend / UI | Streamlit |
| Visualisation | Plotly |
| Data Handling | Pandas |

---

## 📁 Project Structure

```
rppg-heart-rate-monitor/
│
├── app.py                  # Streamlit UI & dashboard
├── rppg_pipeline.py        # Core signal processing logic
├── .streamlit/config.toml  # Streamlit server settings for deploys
├── Dockerfile              # Container image for the Streamlit app
├── docker-compose.yml      # App + optional nginx reverse proxy
├── deploy/nginx/           # nginx reverse-proxy template
├── demo.gif                # Demo animation
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT license
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or 3.12 recommended
- A face video (MP4 or MOV for the Streamlit uploader)
- Docker, if you want containerized deployment

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

Open [`http://localhost:8501`](http://localhost:8501) in your browser, upload a face video, and watch the dashboard populate in real time.

---

## 🐳 Docker Deployment

Build and run only the Streamlit app:

```bash
docker compose up --build app
```

Open [`http://localhost:8501`](http://localhost:8501).
The direct Streamlit port is bound to `127.0.0.1`; use nginx for public/domain traffic.

Run the app behind nginx:

```bash
NGINX_HOST=your-domain.com docker compose up --build -d
```

Point your domain's DNS `A` record to the server IP, then open `http://your-domain.com`.
For HTTPS, terminate TLS at your cloud load balancer, Cloudflare, or a host-level Certbot setup that forwards traffic to nginx on port 80.

Useful environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `APP_PORT` | `8501` | Direct Streamlit host port |
| `NGINX_HTTP_PORT` | `80` | Public nginx HTTP port |
| `NGINX_HOST` | `localhost` | nginx `server_name` |
| `CLIENT_MAX_BODY_SIZE` | `500m` | Max upload size accepted by nginx |

---

## ☁️ Streamlit Community Cloud

Use `app.py` as the entrypoint and keep `requirements.txt` in the repo root.
Community Cloud defaults to Python 3.12; Python 3.11 also works if selected from Advanced settings.

This repo pins OpenCV, NumPy, SciPy, pandas, Plotly, and Streamlit to versions that resolve cleanly on Linux deploy images.
Do not add `packages.txt` for this app on Community Cloud unless a new native dependency is introduced; the current video path uses the `opencv-python-headless` wheel and avoids the apt dependency conflicts shown by `ffmpeg`/`libglib`.

---

## 📊 Dashboard

The live dashboard displays:

- 📈 **BPM graph** — heart rate across each 5-second chunk
- 🌬️ **Respiratory rate trend** — breaths/min over time
- 🎯 **Signal quality chart** — per-chunk reliability score
- ❤️ **Median BPM** — aggregate summary across the full video
- ⚡ **Performance panel** — chunk latency and real-time factor
- 📥 **CSV export** — download all chunk-level data

---

## ⚡ Performance

| Metric | Value |
|---|---|
| Chunk latency | ~250 – 400 ms |
| Real-time factor | ~17 – 21× |
| Optimal conditions | Stable lighting, frontal face, minimal motion |

> Benchmarked on MacBook Air M2. Performance may vary by hardware.

---

## ⚠️ Limitations

> [!WARNING]
> This tool is **not validated for clinical or medical use**. Do not use it for health decisions.

- Sensitive to uneven or rapidly changing lighting
- Requires a frontal, clearly visible face
- Head movement and talking reduce signal quality

---

## 🗺️ Roadmap

- [ ] Live webcam streaming support
- [ ] Deep learning–based face tracking (MediaPipe / dlib)
- [ ] Mobile deployment (WASM or native)
- [ ] ML-based noise reduction
- [ ] Multi-face simultaneous estimation
- [ ] Clinical-grade accuracy benchmarking

---

## 🙋 Author

**Utkarsh Karambhe**
B.Tech CSE (Data Science) · Nagpur, India
[GitHub](https://github.com/Utkarsh-Karambhe)

---

<div align="center">

If this project helped you, consider giving it a ⭐ — it genuinely helps with visibility.

<br/>


</div>
