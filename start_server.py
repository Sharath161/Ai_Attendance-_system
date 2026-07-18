"""Launcher script — changes to project root before starting uvicorn."""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import uvicorn
uvicorn.run("api.main:app", host="0.0.0.0", port=8000, workers=1)
