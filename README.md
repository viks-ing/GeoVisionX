# GeoVisionX — On-Orbit Satellite Intelligence

> **Multi-task geospatial AI that runs inference on the satellite, not the ground.**

---

## 1. What Problem Are We Solving?

Every day, Earth-observation satellites generate terabytes of raw imagery that must be transmitted to ground stations for analysis — a bandwidth bottleneck that delays disaster response by hours and costs operators tens of thousands of dollars per gigabyte of downlink. Emergency managers waiting for flood extent maps, agricultural agencies tracking crop stress across a continent, and defence operators monitoring land-use change all share the same frustration: the data exists in orbit, but the insight arrives too late to act on. GeoVisionX targets this gap by moving inference on-board: a satellite running OrbitalMind processes raw imagery in-situ and downlinks a compact <2 KB semantic JSON (event label, confidence, priority, and an actionable explanation) instead of a full image. The paying customer is any organisation that operates or licences EO satellites and needs near-real-time analytics — commercial constellation operators, national disaster-management agencies, and precision-agriculture platforms — for whom every hour of delay translates directly to crop losses, lives at risk, or a missed operational window.

---

## 2. What Did We Build?

GeoVisionX is a full-stack geospatial AI system built around **OrbitalMind**, a physics-informed multi-task inference pipeline designed to run under the compute and power envelope of a Jetson-class edge device. The encoder extracts spectral (NDVI, MNDWI, BSI), textural, and statistical features from satellite imagery in a single forward pass, inspired by IBM's **TerraMind** foundation model architecture. Three lightweight prediction heads — flood detection, crop-stress classification, and scene-change detection — share the same feature embedding, enabling true multi-task inference without redundant compute. A **TiM (Thinking-in-Modalities)** module generates a synthetic NDVI map as an intermediate reasoning step, improving head accuracy without requiring an actual NIR channel on-board. An **Adaptive Trigger Engine** skips inference entirely when scene-change signal is below threshold, conserving CPU cycles between passes. A **TerraMind-Small Scorer** acts as a self-validation layer, checking internal feature consistency before the result is committed to downlink. The final output is serialised by a **Semantic Compressor** into a structured JSON payload. The project ships with a React + Vite frontend for live demo visualisation and a FastAPI backend serving the pipeline; deployment targets Netlify (frontend) and any Python-capable edge runtime (backend).

---

## 3. How Did We Measure It?

OrbitalMind is benchmarked against a **rule-based NDVI-threshold baseline** (`BaselineClassifier`) that applies a single spectral cut-off with no multi-modal fusion and no self-validation. Every inference call returns both the OrbitalMind result and the baseline result side-by-side, so the delta is always visible in the UI.

| Metric | Baseline | OrbitalMind |
|---|---|---|
| Modalities fused | 1 (NDVI only) | 4 (NDVI, MNDWI, BSI, texture) |
| Tasks per forward pass | 1 | 3 (flood / crop stress / change) |
| Self-validation score | None | 60–93 / 100 |
| Output payload size | ~0.5 KB | <2 KB (all tasks + validation) |
| Inference latency (CPU) | <5 ms | <50 ms |
| Downlink vs. raw image | ~750 KB (256×256 PNG) | <2 KB JSON → **>99% bandwidth saving** |

> **Caveat:** these numbers come from the simulated pipeline running on synthetic demo data, not from a labelled held-out test set. See §5 for what's missing.

---

## 4. Orbital-Compute Story

OrbitalMind is designed from first principles to fit inside the power and memory budget of a **NVIDIA Jetson Orin NX** (the current state-of-the-art for small-sat edge AI modules) or equivalent ARM-class co-processor.

| Constraint | Target | Current Status |
|---|---|---|
| Model size (encoder + 3 heads) | <200 MB | ~0 MB (simulated; real TerraMind-Small checkpoint ~120 MB with INT8 quantisation) |
| Single-image latency | <200 ms end-to-end | <50 ms on CPU (simulation) |
| Peak RAM footprint | <512 MB | <100 MB (NumPy pipeline) |
| Power draw | <10 W sustained | N/A (not yet measured on hardware) |
| Downlink reduction | >95% | >99% on 256×256 RGB input |

The pipeline avoids PyTorch at inference time (pure NumPy), which eliminates the framework overhead and makes it portable to bare-metal C++ transpilation for production deployments. The Adaptive Trigger Engine provides a second layer of compute savings: when consecutive passes show no scene change (edge energy + texture variance below threshold 0.08), the entire inference stack is skipped, reducing average duty cycle in stable scenes.

---

## 5. What Doesn't Work Yet

Be honest about the gaps — they are the most useful part of this document for anyone evaluating the project.

- **No real model weights.** The encoder is a physics-informed heuristic, not a fine-tuned TerraMind checkpoint. LoRA adapter training on TerraTorch with labelled Sentinel-2 scenes is the immediate next step.
- **No held-out evaluation dataset.** Accuracy numbers (precision, recall, F1 per head) against a labelled benchmark (e.g., WorldFloods, Sen12MS, or a custom crop-stress dataset) do not yet exist. The validation score is self-consistency, not ground-truth accuracy.
- **RGB only.** The pipeline accepts 3-band RGB imagery and synthesises NDVI internally. True SAR (Sentinel-1) and multi-spectral (Bands 8/11/12) fusion — the core TerraMind value proposition — is not yet wired in.
- **No hardware test.** Latency and power figures have not been measured on a Jetson or any real space-qualified compute board. The <200 ms target is an estimate.
- **Stochastic noise in demo predictions.** A `random.gauss(0, 0.04)` term is added to each head score to make demo outputs feel realistic. This must be removed before any real evaluation.
- **No temporal baseline.** Change detection compares texture statistics within a single image rather than across a time series. Multi-temporal change detection (T₀ vs. T₁ pairs) is not implemented.
- **Security / auth not hardened.** The FastAPI backend runs with `allow_origins=["*"]`; fine for demo, not for production.

---

## Quick Start

```bash
# Backend
cd GeoVisionX-main
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:5173`, select a scene type and task, and click **Run Analysis**.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Encoder / pipeline | Python, NumPy (TerraMind-inspired heuristics) |
| API | FastAPI + Uvicorn |
| Frontend | React 18, Vite, Vanilla CSS |
| Deployment | Netlify (frontend), any Python edge runtime (backend) |
| Target hardware | NVIDIA Jetson Orin NX / Jetson AGX Orin |
| Foundation model reference | IBM TerraMind (TerraTorch) |
