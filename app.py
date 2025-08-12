from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import requests

app = FastAPI()

# Load the Hugging Face API key from environment variables
HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY not set in environment variables.")

# Using a free model like LLaVA for image description
HF_MODEL = "liuhaotian/llava-v1.5-7b"

@app.post("/describe")
async def describe_image(file: UploadFile = File(...)):
    try:
        # Read file into memory
        image_bytes = await file.read()

        # Send to Hugging Face Inference API
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            files={"image": image_bytes}
        )

        if response.status_code != 200:
            return JSONResponse(
                content={"error": f"Hugging Face API error: {response.text}"},
                status_code=response.status_code
            )

        return response.json()

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/")
def home():
    return {"message": "Image analysis server is running"}
