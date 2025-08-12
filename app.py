from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from google.cloud import vision

app = FastAPI()

# Initialize the Google Vision client once
client = vision.ImageAnnotatorClient()

@app.post("/describe")
async def describe_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = vision.Image(content=image_bytes)

        response = client.label_detection(image=image)
        labels = response.label_annotations

        if response.error.message:
            return JSONResponse(content={"error": response.error.message}, status_code=400)

        descriptions = [label.description for label in labels]
        return {"labels": descriptions}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def home():
    return {"message": "Image analysis server is running"}
