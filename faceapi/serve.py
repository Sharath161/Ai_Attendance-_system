"""Launch the Face API.  python -m faceapi.serve  (or uvicorn faceapi.api:app)"""
import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run("faceapi.api:app",
                host=os.environ.get("FACEAPI_HOST", "0.0.0.0"),
                port=int(os.environ.get("FACEAPI_PORT", "8080")),
                workers=1)
