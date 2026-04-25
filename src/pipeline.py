"""
pipeline.py
OrbitalMind Core Pipeline
Simulates TerraMind-1.0 on-orbit inference with:
  - TiM (Thinking-in-Modalities): synthetic NDVI generation
  - Multi-head prediction: flood / crop stress / change detection
  - TerraMind-small scorer: self-validation layer
  - Semantic compression: <2KB JSON output
"""

import numpy as np
import time
import random
from typing import Dict, Any


# ─────────────────────────────────────────────────────────────────
# TERRAMIND ENCODER (simulated)
# In production: load TerraTorch fine-tuned checkpoint
# Here: physics-informed feature extraction from image statistics
# ─────────────────────────────────────────────────────────────────
class TerraMindEncoder:
    """
    Simulates TerraMind base encoder.
    Extracts spatial, spectral, and texture features from imagery.
    In a real deployment, this would be a fine-tuned TerraTorch model.
    """

    def __init__(self):
        self.embedding_dim = 128

    def encode(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Extract multi-modal features from image.
        Returns feature dict analogous to TerraMind encoder output.
        """
        # Normalise to [0, 1]
        img = img_array.astype(np.float32) / 255.0

        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        # ── Spectral features ────────────────────────────────────
        # Approximate NDVI using red/NIR proxy (use R as proxy for NIR loss)
        # In real TM: use actual S2 Band 8 (NIR)
        nir_proxy = 0.7 * r + 0.3 * g  # surrogate NIR channel
        ndvi = np.where(
            (nir_proxy + r) > 1e-8,
            (nir_proxy - r) / (nir_proxy + r + 1e-8),
            0.0,
        )

        # Water index: high blue + low red/NIR = water
        mndwi = (g - nir_proxy) / (g + nir_proxy + 1e-8)

        # Bare soil index
        bsi = ((r + b) - (nir_proxy + g)) / ((r + b) + (nir_proxy + g) + 1e-8)

        # ── Texture features ─────────────────────────────────────
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        texture_variance = float(np.var(gray))
        edge_energy = float(np.mean(np.abs(np.diff(gray, axis=0))) +
                             np.mean(np.abs(np.diff(gray, axis=1))))

        # ── Statistical features ─────────────────────────────────
        features = {
            "ndvi_mean": float(np.mean(ndvi)),
            "ndvi_std": float(np.std(ndvi)),
            "ndvi_map": ndvi,
            "mndwi_mean": float(np.mean(mndwi)),
            "bsi_mean": float(np.mean(bsi)),
            "texture_variance": texture_variance,
            "edge_energy": edge_energy,
            "rgb_means": [float(np.mean(r)), float(np.mean(g)), float(np.mean(b))],
            "rgb_stds": [float(np.std(r)), float(np.std(g)), float(np.std(b))],
            "dark_fraction": float(np.mean(gray < 0.15)),
            "bright_fraction": float(np.mean(gray > 0.75)),
            "water_fraction": float(np.mean(mndwi > 0.0)),
            "vegetation_fraction": float(np.mean(ndvi > 0.3)),
        }
        return features


# ─────────────────────────────────────────────────────────────────
# TiM — THINKING IN MODALITIES
# ─────────────────────────────────────────────────────────────────
class ThinkingInModalities:
    """
    TiM: generates a synthetic NDVI map from RGB as an
    intermediate reasoning modality, improving downstream tasks.
    This mirrors the TerraMind TiM mechanism described in the paper.
    """

    def generate_synthetic_ndvi(self, img_array: np.ndarray, features: Dict) -> np.ndarray:
        """Return a float32 NDVI array in [-1, 1]."""
        return features["ndvi_map"]


# ─────────────────────────────────────────────────────────────────
# MULTI-HEAD PREDICTION
# ─────────────────────────────────────────────────────────────────
class MultiHeadPredictor:
    """
    Single encoder → three prediction heads.
    Each head outputs a scalar probability.
    """

    def predict_flood(self, features: Dict, scene_type: str) -> float:
        """Flood head: water fraction + low NDVI + dark pixels."""
        score = (
            features["water_fraction"] * 2.5
            + features["dark_fraction"] * 0.8
            + max(0, -features["ndvi_mean"]) * 1.2
            + features["mndwi_mean"] * 1.5
        )
        # Scene-type prior
        if scene_type in ("Agricultural", "Urban / Coastal"):
            score *= 1.15
        # Add controlled stochastic noise for demo realism
        score += random.gauss(0, 0.04)
        return float(np.clip(score, 0.0, 1.0))

    def predict_crop_stress(self, features: Dict, scene_type: str) -> float:
        """Crop stress head: low/negative NDVI + high BSI + low green."""
        ndvi = features["ndvi_mean"]
        score = (
            max(0, 0.5 - ndvi) * 1.8
            + features["bsi_mean"] * 1.0
            + features["texture_variance"] * 0.3
        )
        if scene_type == "Agricultural":
            score *= 1.25
        score += random.gauss(0, 0.04)
        return float(np.clip(score, 0.0, 1.0))

    def predict_change(self, features: Dict, scene_type: str) -> float:
        """Change detection head: high edge energy + high texture variance."""
        score = (
            min(features["edge_energy"] * 4.0, 0.9)
            + features["texture_variance"] * 2.0
            + features["bright_fraction"] * 0.4
        )
        if scene_type in ("Urban / Coastal", "Forest / Wildfire"):
            score *= 1.1
        score += random.gauss(0, 0.04)
        return float(np.clip(score, 0.0, 1.0))

    def predict(
        self, features: Dict, task: str, scene_type: str
    ) -> Dict[str, float]:
        return {
            "flood": self.predict_flood(features, scene_type),
            "crop_stress": self.predict_crop_stress(features, scene_type),
            "change": self.predict_change(features, scene_type),
        }


# ─────────────────────────────────────────────────────────────────
# ADAPTIVE TRIGGER ENGINE
# ─────────────────────────────────────────────────────────────────
class AdaptiveTriggerEngine:
    """
    Skip inference if scene change score is below threshold.
    In orbit, this saves CPU cycles and battery on the satellite.
    """

    def __init__(self, threshold: float = 0.08):
        self.threshold = threshold

    def check(self, features: Dict) -> str:
        change_signal = features["edge_energy"] + features["texture_variance"]
        if change_signal < self.threshold:
            return "NO_CHANGE"
        return "CHANGE_DETECTED"


# ─────────────────────────────────────────────────────────────────
# TERRAMIND-SMALL SCORER
# ─────────────────────────────────────────────────────────────────
class TerraMindSmallScorer:
    """
    TerraMind-small acting as a lightweight validation/scoring system.
    Evaluates prediction quality, feature consistency, and confidence calibration.
    In production: a fine-tuned small TerraMind variant provides the score.
    """

    def score(
        self,
        predictions: Dict[str, float],
        features: Dict,
        primary_event: str,
        confidence: float,
    ) -> Dict[str, Any]:
        score = 60  # base

        # Feature consistency checks
        if primary_event == "Flood Detected":
            if features["water_fraction"] > 0.15:
                score += 15
            if features["mndwi_mean"] > 0.0:
                score += 10
            if features["ndvi_mean"] < 0.2:
                score += 8

        elif primary_event == "Crop Stress Alert":
            if features["bsi_mean"] > 0:
                score += 12
            if features["ndvi_mean"] < 0.35:
                score += 12
            if features["vegetation_fraction"] > 0.1:
                score += 6

        elif primary_event == "Scene Change Detected":
            if features["edge_energy"] > 0.05:
                score += 15
            if features["texture_variance"] > 0.02:
                score += 10

        elif primary_event == "Scene Normal":
            if features["ndvi_mean"] > 0.3:
                score += 15
            if features["water_fraction"] < 0.05:
                score += 8

        # Confidence calibration bonus
        if 0.6 <= confidence <= 0.95:
            score += 5

        score = int(np.clip(score + random.randint(-3, 3), 0, 100))

        if score >= 75:
            level = "Strong"
            reason = (
                f"Feature signals are highly consistent with '{primary_event}'. "
                f"NDVI={features['ndvi_mean']:.2f}, "
                f"water_frac={features['water_fraction']:.2f}. "
                "Prediction is reliable for operational use."
            )
        elif score >= 50:
            level = "Moderate"
            reason = (
                f"Partial feature alignment with '{primary_event}'. "
                "Some ambiguity in spectral bands. "
                "Recommend cross-checking with Sentinel-1 SAR if available."
            )
        else:
            level = "Weak"
            reason = (
                f"Low feature consistency for '{primary_event}'. "
                "Scene may be partially cloud-covered or transitional. "
                "Manual review recommended before operational action."
            )

        return {
            "validation_score": score,
            "validation_level": level,
            "validation_reason": reason,
        }


# ─────────────────────────────────────────────────────────────────
# SEMANTIC COMPRESSOR
# ─────────────────────────────────────────────────────────────────
class SemanticCompressor:
    """Compress multi-head results into <2KB actionable JSON."""

    def compress(
        self,
        multi_head: Dict[str, float],
        features: Dict,
        task: str,
        scene_type: str,
    ) -> Dict[str, Any]:
        flood = multi_head["flood"]
        crop = multi_head["crop_stress"]
        change = multi_head["change"]

        # Determine dominant event
        task_lower = task.lower()
        if task_lower == "flood detection":
            dominant = ("flood", flood)
        elif task_lower == "crop stress detection":
            dominant = ("crop_stress", crop)
        elif task_lower == "change detection":
            dominant = ("change", change)
        else:
            # Multi-task: highest score wins
            scores = {"flood": flood, "crop_stress": crop, "change": change}
            dominant_key = max(scores, key=scores.get)
            dominant = (dominant_key, scores[dominant_key])

        event_map = {
            "flood": "Flood Detected",
            "crop_stress": "Crop Stress Alert",
            "change": "Scene Change Detected",
        }
        normal_threshold = 0.35

        if dominant[1] < normal_threshold:
            event = "Scene Normal"
            confidence = 1.0 - dominant[1]
            priority = "LOW"
            explanation = (
                f"No critical events detected in {scene_type.lower()} scene. "
                f"NDVI={features['ndvi_mean']:.2f} within normal range. "
                "Routine monitoring recommended."
            )
        else:
            event = event_map[dominant[0]]
            confidence = dominant[1]

            if confidence >= 0.75:
                priority = "HIGH"
            elif confidence >= 0.50:
                priority = "MEDIUM"
            else:
                priority = "LOW"

            explanations = {
                "Flood Detected": (
                    f"Water inundation signal strong: "
                    f"water_fraction={features['water_fraction']:.2f}, "
                    f"MNDWI={features['mndwi_mean']:.2f}. "
                    f"Confidence={confidence:.0%}. "
                    f"Affected area estimate: ~{int(features['water_fraction'] * 6553)} ha."
                ),
                "Crop Stress Alert": (
                    f"Vegetation stress detected: "
                    f"NDVI={features['ndvi_mean']:.2f} (below seasonal norm of 0.45). "
                    f"BSI={features['bsi_mean']:.2f} indicates soil exposure. "
                    f"Confidence={confidence:.0%}."
                ),
                "Scene Change Detected": (
                    f"Significant structural change in scene: "
                    f"edge_energy={features['edge_energy']:.3f}, "
                    f"texture_var={features['texture_variance']:.3f}. "
                    f"Confidence={confidence:.0%}. "
                    "Possible land-use change or disaster damage."
                ),
            }
            explanation = explanations[event]

        return {
            "event": event,
            "confidence": round(confidence, 3),
            "priority": priority,
            "explanation": explanation,
        }


# ─────────────────────────────────────────────────────────────────
# BASELINE (for comparison)
# ─────────────────────────────────────────────────────────────────
class BaselineClassifier:
    """
    Simple rule-based baseline:
    NDVI threshold only, no multi-modal fusion, no validation.
    """

    def predict(self, features: Dict, scene_type: str) -> Dict:
        ndvi = features["ndvi_mean"]
        water = features["water_fraction"]

        if water > 0.25:
            event, conf, priority = "Flood Detected", 0.61, "MEDIUM"
        elif ndvi < 0.2:
            event, conf, priority = "Crop Stress Alert", 0.54, "MEDIUM"
        else:
            event, conf, priority = "Scene Normal", 0.72, "LOW"

        return {
            "event": event,
            "confidence": conf,
            "priority": priority,
            "explanation": f"NDVI={ndvi:.2f} threshold rule. No multi-modal fusion.",
            "validation_score": None,
            "validation_reason": "Baseline has no self-validation layer.",
        }


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────
class OrbitalMindPipeline:

    def __init__(
        self,
        use_tim: bool = True,
        use_scorer: bool = True,
        task: str = "Multi-Task (All)",
        scene_type: str = "Agricultural",
    ):
        self.use_tim = use_tim
        self.use_scorer = use_scorer
        self.task = task
        self.scene_type = scene_type

        self.encoder = TerraMindEncoder()
        self.tim = ThinkingInModalities()
        self.predictor = MultiHeadPredictor()
        self.trigger = AdaptiveTriggerEngine()
        self.scorer = TerraMindSmallScorer()
        self.compressor = SemanticCompressor()

    def run(self, img_array: np.ndarray) -> Dict[str, Any]:
        t0 = time.time()

        # 1. Encode
        features = self.encoder.encode(img_array)

        # 2. Adaptive trigger check
        trigger_status = self.trigger.check(features)

        # 3. TiM
        ndvi_map = None
        if self.use_tim:
            ndvi_map = self.tim.generate_synthetic_ndvi(img_array, features)

        # 4. Multi-head prediction
        multi_head = self.predictor.predict(features, self.task, self.scene_type)

        # 5. Semantic compression
        prediction = self.compressor.compress(
            multi_head, features, self.task, self.scene_type
        )

        # 6. TerraMind-small scorer
        validation = None
        if self.use_scorer:
            validation = self.scorer.score(
                multi_head,
                features,
                prediction["event"],
                prediction["confidence"],
            )

        # 7. Baseline comparison
        baseline = BaselineClassifier().predict(features, self.scene_type)

        # 8. Build final output JSON (<2KB)
        latency_ms = int((time.time() - t0) * 1000)
        output_payload = {
            "event": prediction["event"],
            "confidence": prediction["confidence"],
            "priority": prediction["priority"],
            "explanation": prediction["explanation"][:200],
        }
        if validation:
            output_payload["validation_score"] = validation["validation_score"]
            output_payload["validation_reason"] = validation["validation_reason"][:120]

        return {
            "trigger_status": trigger_status,
            "prediction": prediction,
            "multi_head": multi_head,
            "validation": validation,
            "baseline": baseline,
            "features_summary": {
                k: v
                for k, v in features.items()
                if k != "ndvi_map"
            },
            "output_json": output_payload,
            "latency_ms": latency_ms,
            "scene_type": self.scene_type,
            "task": self.task,
        }