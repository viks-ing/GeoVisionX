"""
app.py
OrbitalMind — Streamlit Demo
On-orbit satellite intelligence: TerraMind multi-head inference + TiM + validation.
"""

import io
import json
import time

import numpy as np
import streamlit as st
from PIL import Image

# ── Local modules ─────────────────────────────────────────────────
from src.pipeline import OrbitalMindPipeline
from src.util import generate_sample_image, compute_bandwidth_saving, format_output_json
from src.Visualizer import (
    render_ndvi_colormap,
    render_change_heatmap,
    render_metrics_chart,
    render_edge_specs,
)

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OrbitalMind — On-Orbit Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global dark theme ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #12151e;
        border-right: 1px solid #1e2230;
    }
    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #1a1d2e;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 12px 16px;
    }
    /* ── Priority badges ── */
    .badge-HIGH   { background:#ff4b4b; color:#fff; border-radius:6px; padding:3px 10px; font-weight:700; }
    .badge-MEDIUM { background:#ffa500; color:#000; border-radius:6px; padding:3px 10px; font-weight:700; }
    .badge-LOW    { background:#21c354; color:#000; border-radius:6px; padding:3px 10px; font-weight:700; }
    /* ── Section headings ── */
    h2, h3 { color: #a0c4ff; }
    /* ── Code block ── */
    pre { background: #1a1d2e !important; border-radius: 8px; }
    /* ── Divider ── */
    hr { border-color: #2a2d3e; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <h1 style='font-size:2.4rem; color:#a0c4ff; letter-spacing:1px; margin-bottom:0;'>
            🛰️ OrbitalMind
        </h1>
        <p style='color:#6a7fa8; font-size:1.05rem; margin-top:6px;'>
            On-orbit satellite intelligence · TerraMind encoder · TiM · Multi-head prediction
        </p>
    </div>
    <hr>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR — CONTROLS
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Controls")
    st.markdown("---")

    scene_type = st.selectbox(
        "🌍 Scene Type",
        ["Agricultural", "Urban / Coastal", "Forest / Wildfire"],
        index=0,
    )

    task = st.selectbox(
        "🎯 Inference Task",
        [
            "Multi-Task (All)",
            "Flood Detection",
            "Crop Stress Detection",
            "Change Detection",
        ],
        index=0,
    )

    use_tim = st.toggle("🧠 Enable TiM (Thinking-in-Modalities)", value=True)
    use_scorer = st.toggle("✅ Enable Self-Validation Scorer", value=True)

    st.markdown("---")
    image_source = st.radio(
        "📡 Image Source",
        ["Generate Synthetic Scene", "Upload Custom Image"],
        index=0,
    )

    uploaded_file = None
    if image_source == "Upload Custom Image":
        uploaded_file = st.file_uploader(
            "Upload satellite image (PNG / JPG / TIF)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
        )

    st.markdown("---")
    run_btn = st.button("🚀 Run Inference", width="stretch", type="primary")

# ─────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "img_array" not in st.session_state:
    st.session_state.img_array = None

# ─────────────────────────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────────────────────────
if run_btn:
    # Load image
    if image_source == "Upload Custom Image" and uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB").resize((256, 256))
        img_array = np.array(pil_img, dtype=np.uint8)
    else:
        with st.spinner("Generating synthetic scene…"):
            img_array = generate_sample_image(scene_type)

    # Run pipeline
    pipeline = OrbitalMindPipeline(
        use_tim=use_tim,
        use_scorer=use_scorer,
        task=task,
        scene_type=scene_type,
    )
    with st.spinner("Running on-orbit inference…"):
        result = pipeline.run(img_array)

    st.session_state.result = result
    st.session_state.img_array = img_array

# ─────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────
result = st.session_state.result
img_array = st.session_state.img_array

if result is None:
    st.info("Configure parameters in the sidebar and click **🚀 Run Inference** to begin.")
    st.markdown("---")
    # Show comparison metrics even before inference
    st.markdown("## 📊 Model Performance Benchmarks")
    fig = render_metrics_chart()
    st.pyplot(fig)
    st.markdown("---")
    render_edge_specs()
    st.stop()

# ── Top summary row ───────────────────────────────────────────────
pred = result["prediction"]
val  = result["validation"]
bw   = compute_bandwidth_saving(img_array, result["output_json"])

priority_class = f"badge-{pred['priority']}"
col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    st.metric("🔔 Event", pred["event"])
with col_b:
    st.metric("📊 Confidence", f"{pred['confidence']:.1%}")
with col_c:
    st.metric("⏱️ Latency", f"{result['latency_ms']} ms")
with col_d:
    st.metric("📡 Bandwidth Saved", f"{bw['saving_pct']:.1f}%")

priority_label = f"<span class='{priority_class}'>{pred['priority']} PRIORITY</span>"
st.markdown(f"**Alert Priority:** {priority_label}", unsafe_allow_html=True)

if val:
    val_color = "#21c354" if val["validation_level"] == "Strong" else (
        "#ffa500" if val["validation_level"] == "Moderate" else "#ff4b4b"
    )
    st.markdown(
        f"<span style='color:{val_color}; font-weight:600;'>"
        f"🔍 Validation: {val['validation_level']} ({val['validation_score']}/100)</span> — "
        f"{val['validation_reason']}",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Scene imagery + spectral maps ─────────────────────────────────
st.markdown("## 🖼️ Scene Analysis")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Input Image**")
    st.image(img_array, width="stretch", caption=f"Scene: {scene_type}")

with col2:
    st.markdown("**NDVI Colormap** *(TiM synthetic)*")
    ndvi_img = render_ndvi_colormap(img_array)
    st.image(ndvi_img, width="stretch", caption="Red=Stressed · Green=Healthy")

with col3:
    st.markdown("**Change Heatmap**")
    change_img = render_change_heatmap(img_array)
    st.image(change_img, width="stretch", caption="Bright=High change probability")

st.markdown("---")

# ── Multi-head scores ─────────────────────────────────────────────
st.markdown("## 🧠 Multi-Head Prediction Scores")
mh = result["multi_head"]
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("💧 Flood Score",       f"{mh['flood']:.3f}")
    st.progress(float(mh["flood"]))

with c2:
    st.metric("🌾 Crop Stress Score", f"{mh['crop_stress']:.3f}")
    st.progress(float(mh["crop_stress"]))

with c3:
    st.metric("🔄 Change Score",      f"{mh['change']:.3f}")
    st.progress(float(mh["change"]))

st.markdown("---")

# ── Explanation + baseline comparison ────────────────────────────
col_exp, col_base = st.columns([3, 2])

with col_exp:
    st.markdown("## 📝 Explanation")
    st.info(pred["explanation"])

    trigger = result["trigger_status"]
    trigger_icon = "🟢" if trigger == "CHANGE_DETECTED" else "🟡"
    st.caption(f"{trigger_icon} Adaptive Trigger: **{trigger}**")

with col_base:
    st.markdown("## 🔬 Baseline Comparison")
    base = result["baseline"]
    st.markdown(
        f"| | **OrbitalMind** | **Baseline** |\n"
        f"|---|---|---|\n"
        f"| Event | {pred['event']} | {base['event']} |\n"
        f"| Confidence | {pred['confidence']:.1%} | {base['confidence']:.1%} |\n"
        f"| Priority | {pred['priority']} | {base['priority']} |\n"
        f"| Validation | {val['validation_score'] if val else 'N/A'}/100 | N/A |"
    )

st.markdown("---")

# ── Features summary ──────────────────────────────────────────────
with st.expander("🔭 Raw Feature Embeddings (TerraMind Encoder Output)"):
    fs = result["features_summary"]
    col_f1, col_f2, col_f3 = st.columns(3)
    items = [(k, v) for k, v in fs.items() if not isinstance(v, list)]
    third = len(items) // 3 + 1
    for idx, (k, v) in enumerate(items):
        col = [col_f1, col_f2, col_f3][min(idx // third, 2)]
        col.metric(k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else str(v))

# ── Downlink JSON ─────────────────────────────────────────────────
with st.expander("📡 Semantic Downlink Payload (< 2 KB JSON)"):
    payload_str = format_output_json(result)
    st.code(payload_str, language="json")
    raw_kb  = bw["raw_kb"]
    out_b   = bw["output_bytes"]
    ratio   = bw["compression_ratio"]
    st.caption(
        f"Raw image: **{raw_kb:.1f} KB** → Payload: **{out_b} bytes** "
        f"(×{ratio:.0f} compression · {bw['saving_pct']:.2f}% saved)"
    )

st.markdown("---")

# ── Benchmark charts ──────────────────────────────────────────────
st.markdown("## 📊 Performance Benchmarks")
fig = render_metrics_chart()
st.pyplot(fig)

st.markdown("---")
render_edge_specs()
