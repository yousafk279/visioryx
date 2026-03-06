from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import model_engine

app = FastAPI(title="Visioryx Pro Neural Server")

# Enable communication between HTML and Python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    """Receives image and returns the Neural Insight"""
    content = await file.read()
    return model_engine.predict_image(content)

if __name__ == "__main__":
    print("VisioryX Intelligence Server is now LIVE.")
    uvicorn.run(app, host="127.0.0.1", port=8000)