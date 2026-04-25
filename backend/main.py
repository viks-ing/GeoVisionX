from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import base64
from typing import Optional

from src.pipeline import OrbitalMindPipeline
from src.util import generate_sample_image, compute_bandwidth_saving
from src.Visualizer import render_ndvi_colormap, render_change_heatmap

app = FastAPI(title="GeoVisionX API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.get("/api/health")
def health_check():
    return {"status": "online", "system": "GeoVisionX On-Orbit Processor"}

@app.post("/api/inference")
async def run_inference(
    scene_type: str = "Agricultural",
    task: str = "Multi-Task (All)",
    use_tim: bool = True,
    use_scorer: bool = True,
    file: Optional[UploadFile] = File(None)
):
    try:
        # 1. Load Image
        if file:
            content = await file.read()
            pil_img = Image.open(io.BytesIO(content)).convert("RGB").resize((256, 256))
            img_array = np.array(pil_img, dtype=np.uint8)
        else:
            img_array = generate_sample_image(scene_type)

        # 2. Run Pipeline
        pipeline = OrbitalMindPipeline(
            use_tim=use_tim,
            use_scorer=use_scorer,
            task=task,
            scene_type=scene_type,
        )
        result = pipeline.run(img_array)

        # 3. Generate Visualizations
        ndvi_img = render_ndvi_colormap(img_array)
        change_img = render_change_heatmap(img_array)

        # 4. Prepare Response
        bw = compute_bandwidth_saving(img_array, result["output_json"])
        
        response = {
            "success": True,
            "prediction": result["prediction"],
            "multi_head": result["multi_head"],
            "validation": result["validation"],
            "baseline": result["baseline"],
            "latency_ms": result["latency_ms"],
            "trigger_status": result["trigger_status"],
            "bandwidth": bw,
            "images": {
                "input": image_to_base64(Image.fromarray(img_array)),
                "ndvi": image_to_base64(ndvi_img),
                "change": image_to_base64(change_img),
            },
            "output_json": result["output_json"]
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
