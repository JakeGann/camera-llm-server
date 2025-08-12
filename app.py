from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import requests

app = FastAPI()

HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY not set in environment variables.")

HF_MODEL = "nlpconnect/vit-gpt2-image-captioning"

@app.post("/describe")
async def describe_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        )
        if response.status_code != 200:
            return JSONResponse(content={"error": response.text}, status_code=response.status_code)

        return response.json()

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def home():
    return {"message": "Image analysis server is running"}

@app.get("/healthz")
def health_check():
    return {"status": "ok"}
