# 🫀 Near Real-Time rPPG Heart Rate Monitor

![Demo](demo.mp4)

A **computer vision + signal processing system** that estimates:

- ❤️ **Heart Rate (BPM)**
- 🌬️ **Respiratory Rate (breaths/min)**

using **remote photoplethysmography (rPPG)** from a face video.

---

## 🔥 Key Highlights

- ⚡ Near real-time processing (17–21x faster than real-time)
- 🎯 5-second chunk-based estimation
- 🧠 Signal processing using FFT + bandpass filtering
- 🧍 Face tracking optimized (detect every 10 frames)
- 📊 Interactive dashboard (Streamlit + Plotly)

---

## 🎥 Demo

👉 If video doesn't load, download it or open directly.

https://github.com/YOUR-USERNAME/rppg-heart-rate-monitor/blob/main/demo.mp4

---

## 🧠 How It Works

### 1. Face Detection
- Haarcascade-based detection (OpenCV)
- Runs every 10 frames for speed

### 2. Signal Extraction
- Extract mean RGB values from face ROI
- Normalize green channel (rPPG signal)

### 3. Signal Processing
- Detrending
- Interpolation to uniform sampling
- Bandpass filtering:
  - HR: 0.7–3 Hz
  - RR: 0.1–0.5 Hz

### 4. Frequency Analysis
- FFT applied on filtered signal
- Peak frequency → BPM

---

## ⚙️ Tech Stack

| Category            | Tools Used |
|--------------------|-----------|
| Computer Vision     | OpenCV |
| Signal Processing   | NumPy, SciPy |
| Frontend UI         | Streamlit |
| Visualization       | Plotly |
| Data Handling       | Pandas |

---

## 📂 Project Structure


rppg-heart-rate-monitor/
│
├── app.py # Streamlit UI
├── rppg_pipeline.py # Core signal processing
├── demo.mp4 # Demo video
├── requirements.txt
└── README.md


---

## ▶️ Run Locally

```bash
git clone https://github.com/YOUR-USERNAME/rppg-heart-rate-monitor.git
cd rppg-heart-rate-monitor

pip install -r requirements.txt
streamlit run app.py
📊 Output
Heart Rate per 5-second chunk
Respiratory Rate per chunk
Overall median BPM
Signal quality score
Real-time performance metrics
⚡ Performance
⏱️ Chunk latency: ~250–400 ms
🚀 Real-time factor: ~17–21x
✅ Robust under stable lighting conditions
📈 Sample Output (What You’ll See)
Live BPM graph
Respiratory rate trend
Signal quality bar chart
Performance dashboard
⚠️ Limitations
Sensitive to lighting variations
Requires visible face (frontal)
Motion can reduce accuracy
🚀 Future Improvements
🎥 Real-time webcam support
🤖 Deep learning-based face tracking
📱 Mobile deployment
🧠 ML-based noise reduction
🏥 Clinical-grade validation
👨‍💻 Author

Utkarsh Karambhe
📍 Nagpur, India
🎓 B.Tech CSE (Data Science)

⭐ If you found this useful

Give this repo a ⭐ — it helps!


---

# 🔥 Now let me upgrade you further (important)

## ⚠️ Your demo currently won’t autoplay

GitHub does NOT autoplay `.mp4` inside README.

---

## 💥 BEST UPGRADE (do this)

### Convert to GIF:

Use:
- **Kap (Mac)**
- **ScreenToGif (Windows)**

Then:

```bash
git add demo.gif
git commit -m "Added demo GIF"
git push
