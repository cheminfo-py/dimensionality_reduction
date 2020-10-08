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


@app.post("/pca")
def run_pca(array: list, n_components: int = 2, standardize: bool = False):
    ...


@app.post("/kernelpca")
def run_kernelpca(
    array: list,
    n_components: int = 2,
    kernel: str = "rbf",
    gamma: float = None,
    degree: float = None,
    coef: float = None,
    alpha: float = None,
    standardize: bool = False,
):
    ...


@app.post("/umap")
def run_umap(
    array: list,
    n_components=2,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    standardize: bool = False,
):
    ...


@app.post("/tsne")
def run_tsne(array: list, perplexity: float = 0.02, standardize: bool = False):
    ...


@app.post("/isomap")
def run_isomap(
    array: list, n_neighbors: int = 5, n_components: int = 2, standardize: bool = False
):
    ...


@app.post("/lle")
def run_lle(
    array: list,
    n_neighbors: int = 5,
    n_components: int = 2,
    reg: float = 0.001,
    standardize: bool = False,
):
    ...
