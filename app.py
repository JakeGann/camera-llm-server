from fastapi import FastAPI, File, UploadFile
import requests
import os

app = FastAPI()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"  # Fast + supports vision

@app.post("/describe")
async def describe_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    # Send image to OpenAI
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    import base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image briefly for a security alert"},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_image}"}
            ]}
        ],
        "max_tokens": 100
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
    if r.status_code != 200:
        return {"error": r.text}
    data = r.json()
    description = data["choices"][0]["message"]["content"]
    return {"description": description}
