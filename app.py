
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import cv2
import numpy as np
import time as _time
from rppg_pipeline import process_chunk, aggregate_overall

st.set_page_config(page_title="rPPG Heart Rate Monitor", layout="centered")

# ── Custom CSS Design System ─────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root variables ──────────────────────────────────────────── */
:root {
    --bg-primary:      #0a0e17;
    --bg-secondary:    #111827;
    --bg-card:         rgba(17, 24, 39, 0.7);
    --border-subtle:   rgba(255, 255, 255, 0.06);
    --text-primary:    #f1f5f9;
    --text-secondary:  #94a3b8;
    --accent-red:      #ef4444;
    --accent-red-glow: rgba(239, 68, 68, 0.3);
    --accent-blue:     #3b82f6;
    --accent-green:    #22c55e;
    --accent-amber:    #f59e0b;
    --glass-bg:        rgba(15, 23, 42, 0.6);
    --glass-border:    rgba(255, 255, 255, 0.08);
}

/* ── Global resets ───────────────────────────────────────────── */
.stApp {
    background: linear-gradient(145deg, #0a0e17 0%, #0f172a 40%, #111827 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}

/* Remove default Streamlit header and footer */
header[data-testid="stHeader"] {
    background: rgba(10, 14, 23, 0.8) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}
footer { display: none !important; }

/* ── Main container ──────────────────────────────────────────── */
.block-container {
    padding-top: 2rem !important;
    max-width: 780px !important;
}

/* ── Typography ──────────────────────────────────────────────── */
h1 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.1rem !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
    padding-bottom: 0.3rem !important;
    color: var(--text-primary) !important;
}
.gradient-title {
    background: linear-gradient(135deg, #f1f5f9 0%, #e74c3c 50%, #ef4444 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

h2, h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.01em !important;
}

p, li, span, div {
    font-family: 'Inter', sans-serif !important;
}

/* Caption styling */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-secondary) !important;
    font-size: 0.92rem !important;
    line-height: 1.55 !important;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b2e 50%, #0f172a 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li {
    color: var(--text-secondary) !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
}

section[data-testid="stSidebar"] strong {
    color: #e2e8f0 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: var(--glass-border) !important;
    margin: 1.2rem 0 !important;
}

/* ── File uploader ───────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--glass-bg) !important;
    border: 1.5px dashed rgba(239, 68, 68, 0.25) !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(239, 68, 68, 0.5) !important;
    background: rgba(239, 68, 68, 0.03) !important;
    box-shadow: 0 0 30px rgba(239, 68, 68, 0.06) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploader"] small {
    color: rgba(148, 163, 184, 0.6) !important;
}
/* Inner drop zone area */
[data-testid="stFileUploader"] section {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1.5px dashed rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    padding: 1.5rem !important;
    color: var(--text-secondary) !important;
}
[data-testid="stFileUploader"] section span {
    color: var(--text-secondary) !important;
}
[data-testid="stFileUploader"] section small {
    color: rgba(148, 163, 184, 0.5) !important;
}
/* Browse files button inside uploader */
[data-testid="stFileUploader"] section button,
[data-testid="stFileUploadDropzone"] button {
    background: rgba(239, 68, 68, 0.15) !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"] section button:hover,
[data-testid="stFileUploadDropzone"] button:hover {
    background: rgba(239, 68, 68, 0.25) !important;
    border-color: rgba(239, 68, 68, 0.5) !important;
}
/* File upload dropzone overall */
[data-testid="stFileUploadDropzone"] {
    background: rgba(15, 23, 42, 0.8) !important;
    border-color: rgba(255, 255, 255, 0.1) !important;
    color: var(--text-secondary) !important;
}
/* Upload icon color */
[data-testid="stFileUploadDropzone"] svg {
    fill: var(--text-secondary) !important;
    stroke: var(--text-secondary) !important;
}

/* ── Video player container ──────────────────────────────────── */
video {
    border-radius: 12px !important;
    border: 1px solid var(--glass-border) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
}

/* ── Primary button ──────────────────────────────────────────── */
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #dc2626 0%, #ef4444 50%, #f87171 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.7rem 1.5rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.25) !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(239, 68, 68, 0.4) !important;
}

/* ── Secondary / Download button ─────────────────────────────── */
.stDownloadButton > button,
button[data-testid="stBaseButton-secondary"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
    transition: all 0.3s ease !important;
    backdrop-filter: blur(8px) !important;
}
.stDownloadButton > button:hover,
button[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(239, 68, 68, 0.1) !important;
    border-color: rgba(239, 68, 68, 0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── Metric cards ────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 14px !important;
    padding: 1.1rem 1.2rem !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="stMetric"]::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 3px !important;
    background: linear-gradient(90deg, var(--accent-red), var(--accent-blue)) !important;
    border-radius: 14px 14px 0 0 !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(239, 68, 68, 0.2) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
}

/* ── Info / Success alerts ───────────────────────────────────── */
[data-testid="stAlert"] {
    background: var(--glass-bg) !important;
    border-radius: 12px !important;
    border: 1px solid var(--glass-border) !important;
    backdrop-filter: blur(8px) !important;
}
div[data-testid="stAlert"] > div {
    color: var(--text-secondary) !important;
}

/* ── Success alert special ───────────────────────────────────── */
.stSuccess, [data-baseweb="notification"][kind="positive"] {
    border-left: 3px solid var(--accent-green) !important;
}

/* ── Progress bar ────────────────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #dc2626, #ef4444, #f87171) !important;
    border-radius: 8px !important;
}
.stProgress {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 8px !important;
}

/* ── Dividers ────────────────────────────────────────────────── */
hr {
    border-color: var(--glass-border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Dataframe / table ───────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--glass-border) !important;
}

/* ── Plotly chart containers ─────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 14px !important;
    padding: 0.5rem !important;
    backdrop-filter: blur(8px) !important;
}

/* ── Subheader styling ───────────────────────────────────────── */
[data-testid="stSubheader"] h2,
[data-testid="stSubheader"] h3 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}

/* ── Animated heart pulse on title ───────────────────────────── */
@keyframes heartPulse {
    0%, 100% { transform: scale(1); }
    15% { transform: scale(1.15); }
    30% { transform: scale(1); }
    45% { transform: scale(1.1); }
}

/* ── Subtle glow border animation for hero section ───────────── */
@keyframes glowShift {
    0%   { box-shadow: 0 0 20px rgba(239, 68, 68, 0.05); }
    50%  { box-shadow: 0 0 30px rgba(239, 68, 68, 0.1); }
    100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.05); }
}

/* ── Smooth scrollbar ────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.2);
}

/* ── Link styling ────────────────────────────────────────────── */
a {
    color: var(--accent-red) !important;
    text-decoration: none !important;
    transition: color 0.2s ease !important;
}
a:hover {
    color: #f87171 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────
st.markdown(
    "<h1>🫀 <span class='gradient-title'>Near Real-Time rPPG Heart Rate Monitor</span></h1>",
    unsafe_allow_html=True
)
st.caption(
    "Upload a 60-second face video. Heart Rate and Respiratory Rate "
    "are estimated in 5-second chunks."
)

# ── Sidebar info ─────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
**Biomarkers estimated:**
- ❤️ **Heart Rate** (BPM) — FFT peak of green-channel signal (0.7–3 Hz)
- 🌬️ **Respiratory Rate** (breaths/min) — low-band FFT (0.1–0.5 Hz)

**Pipeline:**
- Face detected every 10 frames (reuse ROI = fast)
- Each 5s chunk processes in ~250–400 ms
- Real-time factor ≈ 17–21x on MacBook Air M2
    """)
    st.divider()
    st.caption("Built with OpenCV · SciPy · Streamlit")

# ── Upload ───────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload face video (.mp4 / .mov)", type=["mp4", "mov"])

if uploaded:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded.read())

    st.video("temp_video.mp4")
    st.divider()

    if st.button("▶ Run rPPG Analysis", type="primary", use_container_width=True):

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        cap          = cv2.VideoCapture("temp_video.mp4")
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s   = total_frames / fps

        st.info(f"📹 Video: {duration_s:.1f}s · {fps:.0f} fps · {total_frames} frames")

        chunk_frames, chunk_times, chunk_results = [], [], []
        frame_idx, chunk_index = 0, 0
        first_t = None

        progress = st.progress(0, text="Starting...")
        chart_ph = st.empty()

        wall_start = _time.perf_counter()

        # ── Frame loop ────────────────────────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                if chunk_times and (chunk_times[-1] - chunk_times[0]) >= 3.0:
                    r = process_chunk(chunk_index, chunk_frames, chunk_times, face_cascade)
                    chunk_results.append(r)
                break

            t_video = frame_idx / fps
            if first_t is None:
                first_t = t_video
            t_rel = t_video - first_t

            chunk_frames.append(frame)
            chunk_times.append(t_rel)
            frame_idx += 1

            progress.progress(
                min(frame_idx / total_frames, 1.0),
                text=f"Processing frames… {frame_idx}/{total_frames}  "
                     f"({len(chunk_results)} chunks done)"
            )

            if chunk_times[-1] - chunk_times[0] >= 5.0:
                r = process_chunk(chunk_index, chunk_frames, chunk_times, face_cascade)
                chunk_results.append(r)
                chunk_index += 1
                chunk_frames, chunk_times = [], []

                # ── Update live line chart after every chunk ──────
                labels = [f"{c.start_time:.0f}s" for c in chunk_results]
                bpms   = [c.bpm      if c.bpm      is not None else None for c in chunk_results]
                rrs    = [c.resp_bpm if c.resp_bpm is not None else None for c in chunk_results]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=labels, y=bpms,
                    mode="lines+markers",
                    name="Heart Rate (BPM)",
                    line=dict(color="#ef4444", width=2.5, shape="spline"),
                    marker=dict(size=8, color="#ef4444",
                                line=dict(width=2, color="rgba(239,68,68,0.3)"))
                ))
                fig.add_trace(go.Scatter(
                    x=labels, y=rrs,
                    mode="lines+markers",
                    name="Resp. Rate (br/min)",
                    line=dict(color="#3b82f6", width=2, dash="dot", shape="spline"),
                    marker=dict(size=7, color="#3b82f6",
                                line=dict(width=2, color="rgba(59,130,246,0.3)"))
                ))
                fig.update_layout(
                    title=dict(text="Live Biomarker Estimates — 5-second chunks",
                               font=dict(size=14, color="#94a3b8")),
                    xaxis_title="Chunk start time",
                    yaxis_title="Value",
                    height=380,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1, font=dict(color="#94a3b8")),
                    margin=dict(t=60, b=40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8", family="Inter, sans-serif"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                               linecolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                               linecolor="rgba(255,255,255,0.1)"),
                )
                chart_ph.plotly_chart(fig, use_container_width=True)

        cap.release()
        wall_end  = _time.perf_counter()
        wall_time = wall_end - wall_start

        progress.progress(1.0, text="✅ Done!")

        # ── Overall aggregation ───────────────────────────────────
        overall_bpm, overall_rr = aggregate_overall(chunk_results)

        st.success("✅ Analysis complete!")
        st.divider()

        # ── 3 metric cards ────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        col1.metric("❤️ Heart Rate",
                    f"{overall_bpm:.1f} BPM" if overall_bpm else "N/A")
        col2.metric("🌬️ Resp. Rate",
                    f"{overall_rr:.1f} br/min" if overall_rr else "N/A")
        col3.metric("✅ Valid Chunks",
                    f"{sum(1 for c in chunk_results if c.quality > 2.0)}/{len(chunk_results)}")

        # ── BPM Gauge ─────────────────────────────────────────────
        if overall_bpm:
            st.divider()
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=overall_bpm,
                title={"text": "Overall Heart Rate (BPM)",
                       "font": {"size": 16, "color": "#94a3b8", "family": "Inter, sans-serif"}},
                number={"font": {"size": 52, "color": "#ef4444", "family": "Inter, sans-serif"},
                        "suffix": " BPM", "valueformat": ".1f"},
                gauge={
                    "axis": {
                        "range": [40, 160],
                        "tickwidth": 1,
                        "tickcolor": "rgba(255,255,255,0.2)",
                        "tickvals": [40, 60, 80, 100, 120, 140, 160],
                        "tickfont": {"color": "#64748b", "size": 11},
                    },
                    "bar":  {"color": "#ef4444", "thickness": 0.3},
                    "bgcolor": "rgba(255,255,255,0.02)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [40,  60], "color": "rgba(59,130,246,0.15)"},
                        {"range": [60, 100], "color": "rgba(34,197,94,0.12)"},
                        {"range": [100, 160], "color": "rgba(239,68,68,0.12)"},
                    ],
                    "threshold": {
                        "line": {"color": "#f1f5f9", "width": 3},
                        "thickness": 0.8,
                        "value": overall_bpm
                    }
                }
            ))
            fig_gauge.update_layout(
                height=300,
                margin=dict(t=50, b=10, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f1f5f9", "family": "Inter, sans-serif"}
            )

            # Zone label below gauge
            if overall_bpm < 60:
                zone_msg = "🔵 Low — below normal resting range"
                zone_color = "rgba(59,130,246,0.15)"
            elif overall_bpm <= 100:
                zone_msg = "🟢 Normal — healthy resting heart rate"
                zone_color = "rgba(34,197,94,0.12)"
            else:
                zone_msg = "🔴 Elevated — above normal resting range"
                zone_color = "rgba(239,68,68,0.12)"

            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(
                f"<div style='text-align:center; padding:12px 20px; "
                f"background:{zone_color}; border-radius:12px; "
                f"font-size:0.9rem; color:#e2e8f0; font-family:Inter,sans-serif; "
                f"font-weight:500; border:1px solid rgba(255,255,255,0.06); "
                f"backdrop-filter:blur(8px); margin-top:0.5rem;'>{zone_msg}</div>",
                unsafe_allow_html=True
            )

        # ── Performance stats ─────────────────────────────────────
        if chunk_results:
            avg_lat = float(np.mean([c.latency_sec for c in chunk_results]))
            max_lat = float(np.max( [c.latency_sec for c in chunk_results]))
            rtf     = 5.0 / avg_lat

            st.divider()
            st.subheader("⚡ Performance Metrics")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Total Runtime",     f"{wall_time:.2f} s")
            p2.metric("Avg Chunk Latency", f"{avg_lat*1000:.0f} ms")
            p3.metric("Max Chunk Latency", f"{max_lat*1000:.0f} ms")
            p4.metric("Real-time Factor",  f"{rtf:.1f}x ✓")

        # ── Quality bar chart ─────────────────────────────────────
        st.divider()
        st.subheader("📊 Signal Quality per Chunk")

        q_labels = [f"#{c.index:02d}" for c in chunk_results]
        q_values = [c.quality for c in chunk_results]
        q_colors = ["#22c55e" if c.quality > 2.0 else "#ef4444" for c in chunk_results]

        fig_quality = go.Figure(go.Bar(
            x=q_labels,
            y=q_values,
            marker=dict(color=q_colors, line=dict(width=0),
                        cornerradius=4),
            text=[f"{q:.2f}" for q in q_values],
            textposition="outside",
            textfont=dict(size=11, color="#94a3b8", family="Inter, sans-serif")
        ))
        fig_quality.add_hline(
            y=2.0,
            line_dash="dot",
            line_color="rgba(255,255,255,0.3)",
            line_width=1.5,
            annotation_text="  Quality threshold (2.0)",
            annotation_position="top left",
            annotation_font_size=11,
            annotation_font_color="#94a3b8"
        )
        fig_quality.update_layout(
            xaxis_title="Chunk",
            yaxis_title="Quality Score (SNR ratio)",
            height=320,
            margin=dict(t=40, b=30),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter, sans-serif"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                       linecolor="rgba(255,255,255,0.1)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                       linecolor="rgba(255,255,255,0.1)")
        )
        st.plotly_chart(fig_quality, use_container_width=True)
        st.caption(
            "🟢 Green bars = valid (quality > 2.0) — included in Overall BPM  ·  "
            "🔴 Red bars = low quality — excluded from final estimate"
        )

        # ── Chunk detail table ────────────────────────────────────
        st.divider()
        st.subheader("📋 Chunk-Level Detail")

        rows = []
        for c in chunk_results:
            rows.append({
                "Chunk":        f"#{c.index:02d}",
                "Window":       f"{c.start_time:.1f}s – {c.end_time:.1f}s",
                "HR (BPM)":     f"{c.bpm:.1f}"      if c.bpm      is not None else "N/A",
                "RR (br/min)":  f"{c.resp_bpm:.1f}" if c.resp_bpm is not None else "N/A",
                "Quality":      f"{c.quality:.2f}",
                "Latency (ms)": str(round(c.latency_sec * 1000)),
                "Valid":        "✓" if c.quality > 2.0 else "✗"
            })

        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.apply(
                lambda row: ["background-color: #1a3a1a" if row["Valid"] == "✓"
                             else "background-color: #3a1a1a"] * len(row),
                axis=1
            ),
            use_container_width=True,
            hide_index=True
        )

        # ── Download button ───────────────────────────────────────
        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv,
            file_name="rppg_results.csv",
            mime="text/csv",
            use_container_width=True
        )