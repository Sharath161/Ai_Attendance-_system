"""Launch the attendance system.  python -m attendance.serve"""
import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run("attendance.app:app",
                host=os.environ.get("ATT_HOST", "0.0.0.0"),
                port=int(os.environ.get("ATT_PORT", "8000")),
                workers=1)
