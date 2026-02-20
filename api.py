from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io

from RAG_pneumonia_agent import run_agent, app #import the full agent

app = FastAPI(
    title="Chest X-Ray AI Diagnostic API",
    description=" Diffusion-Augmented ViT + RAG Medical Agent with Heatmap Output",
    version="1.0.0"
)   

#response schema
class DiagnosisResponse(BaseModel):
    report: str
    prediction: dict
    heatmap_base64: str


#health check
@app.get("/")
def root():
    return{"status": "API is running"}

#diagnostic endpoint
@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        result = run_agent(image=image)

        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))