"""
dimensionality_reduction.py
REST-API serving dimensionality reduction techniques
"""
from . import __version__
from fastapi import FastAPI

app = FastAPI()


@app.get("/version")
def read_version():
    return {"version": __version__}
