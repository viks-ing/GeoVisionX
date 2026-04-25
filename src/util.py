"""
utils.py
OrbitalMind Utilities
- Synthetic satellite image generation (realistic scene types)
- Bandwidth computation
- JSON formatting
"""

import numpy as np
import json
from typing import Dict, Any


def generate_sample_image(scene_type: str) -> np.ndarray:
    """
    Generate a realistic synthetic 256×256 RGB satellite image
    for each scene type. Uses noise + spatial gradients to simulate
    real Sentinel-2 RGB composites.
    """
    np.random.seed(42)
    h, w = 256, 256
    img = np.zeros((h, w, 3), dtype=np.float32)

    if scene_type == "Agricultural":
        # Green fields with dry patches and field boundaries
        base_green = np.random.normal(0.38, 0.08, (h, w))
        base_red = np.random.normal(0.22, 0.06, (h, w))
        base_blue = np.random.normal(0.15, 0.04, (h, w))

        # Add field-like patches (dry / stressed areas)
        for _ in range(12):
            cx, cy = np.random.randint(30, 220), np.random.randint(30, 220)
            r_size = np.random.randint(15, 45)
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < r_size ** 2
            if np.random.rand() > 0.5:
                # Stressed brown patch
                base_green[mask] *= 0.65
                base_red[mask] *= 1.35
            else:
                # Lush area
                base_green[mask] *= 1.25
                base_red[mask] *= 0.80

        img[:, :, 0] = np.clip(base_red, 0, 1)
        img[:, :, 1] = np.clip(base_green, 0, 1)
        img[:, :, 2] = np.clip(base_blue, 0, 1)

    elif scene_type == "Urban / Coastal":
        # Grey urban + bright coastline + dark water body
        base = np.random.normal(0.45, 0.12, (h, w))

        # Water body (lower half-left)
        water_mask = np.zeros((h, w), dtype=bool)
        water_mask[140:, :110] = True
        for row in range(130, 170):
            col = int(110 + 15 * np.sin((row - 130) * 0.25))
            water_mask[row, :max(0, col)] = True

        img[:, :, 0] = np.clip(base * 0.85, 0, 1)
        img[:, :, 1] = np.clip(base * 0.82, 0, 1)
        img[:, :, 2] = np.clip(base * 0.90, 0, 1)

        img[water_mask, 0] = 0.05 + np.random.normal(0, 0.02, water_mask.sum())
        img[water_mask, 1] = 0.12 + np.random.normal(0, 0.02, water_mask.sum())
        img[water_mask, 2] = 0.35 + np.random.normal(0, 0.03, water_mask.sum())

    elif scene_type == "Forest / Wildfire":
        # Dense forest with burn scar areas
        base_green = np.random.normal(0.30, 0.07, (h, w))
        base_red = np.random.normal(0.18, 0.05, (h, w))
        base_blue = np.random.normal(0.12, 0.04, (h, w))

        # Burn scar patch (dark, reddish)
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        scar_mask = (((xx - 160) ** 2 / 60 ** 2) + ((yy - 80) ** 2 / 45 ** 2)) < 1
        base_red[scar_mask] = 0.55 + np.random.normal(0, 0.04, scar_mask.sum())
        base_green[scar_mask] = 0.08 + np.random.normal(0, 0.02, scar_mask.sum())
        base_blue[scar_mask] = 0.06 + np.random.normal(0, 0.01, scar_mask.sum())

        img[:, :, 0] = np.clip(base_red, 0, 1)
        img[:, :, 1] = np.clip(base_green, 0, 1)
        img[:, :, 2] = np.clip(base_blue, 0, 1)

    else:
        # Fallback: mixed scene
        img = np.random.uniform(0.1, 0.8, (h, w, 3)).astype(np.float32)

    # Apply spatial low-pass smoothing for realism
    from scipy.ndimage import gaussian_filter
    for c in range(3):
        img[:, :, c] = gaussian_filter(img[:, :, c], sigma=1.5)

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def compute_bandwidth_saving(img_array: np.ndarray, output_json: Dict) -> Dict:
    """Compare raw image bytes vs compressed JSON output."""
    raw_bytes = img_array.nbytes  # uncompressed
    json_str = json.dumps(output_json, separators=(",", ":"))
    json_bytes = len(json_str.encode("utf-8"))

    saving_pct = (1.0 - json_bytes / raw_bytes) * 100

    return {
        "raw_kb": raw_bytes / 1024,
        "output_bytes": json_bytes,
        "saving_pct": saving_pct,
        "compression_ratio": raw_bytes / max(json_bytes, 1),
    }


def format_output_json(result: Dict[str, Any]) -> str:
    """Pretty-format the final downlink JSON payload."""
    payload = result.get("output_json", {})
    return json.dumps(payload, indent=2, ensure_ascii=False)