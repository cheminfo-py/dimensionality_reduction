# -*- coding: utf-8 -*-
"""
dimensionality_reduction.py
REST-API serving dimensionality reduction techniques
"""
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import __version__
from .manifold import tsne, umap
from .model import DimensionalityReduction
from .pca import KernelPCAModel, kernel_pca, pca
from .preprocess import convert_to_array

app = FastAPI()

logger = logging.getLogger("api")


@app.get("/version")
def read_version():
    return {"version": __version__}


@app.post("/pca")
def run_pca(parameters: DimensionalityReduction):
    try:
        array = convert_to_array(parameters.array, parameters.standardize)
    except ValueError as execp:
        logger.error("Not a valid array {}".format(execp))
        raise HTTPException(status_code=400, detail="Not a valid array")

    try:
        return pca(array, parameters.n_components)
    except Exception:
        raise HTTPException(status_code=400, detail="PCA failed")


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
    try:
        array = convert_to_array(array, standardize)
    except ValueError:
        raise HTTPException(status_code=400, detail="Not a valid array")

    try:
        return kernel_pca(array, n_components, kernel, gamma, degree, coef, alpha)
    except Exception:
        raise HTTPException(status_code=400, detail="Kernel PCA failed")


@app.post("/umap")
def run_umap(
    array: list,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    standardize: bool = False,
):
    try:
        array = convert_to_array(array, standardize)
    except ValueError:
        raise HTTPException(status_code=400, detail="Not a valid array")

    try:
        return umap(array, n_components, n_neighbors, min_dist, metric)
    except Exception:
        raise HTTPException(status_code=400, detail="UMAP failed")


@app.post("/tsne")
def run_tsne(
    array: list,
    n_components: int = 2,
    perplexity: float = 30.0,
    early_exaggeration: float = 12.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    n_iter_without_progress: int = 300,
    min_grad_norm: float = 1e-07,
    metric: str = "euclidean",
    standardize: bool = False,
):
    try:
        array = convert_to_array(array, standardize)
    except ValueError:
        raise HTTPException(status_code=400, detail="Not a valid array")

    try:
        return tsne(
            array,
            n_components,
            perplexity,
            early_exaggeration,
            learning_rate,
            n_iter,
            n_iter_without_progress,
            min_grad_norm,
            metric,
        )
    except Exception:
        raise HTTPException(status_code=400, detail="UMAP failed")


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
