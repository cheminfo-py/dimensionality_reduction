# -*- coding: utf-8 -*-
"""
dimensionality_reduction.py
REST-API serving dimensionality reduction techniques
"""
import logging

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from fastapi.responses import HTMLResponse

from . import __version__
from .manifold import (
    tsne,
    umap,
    isomap,
    lle,
    UMAPModel,
    TSNEModel,
    IsomapModel,
    LLEModel,
)
from .model import DimensionalityReduction, DimensionalityReductionResponse
from .pca import KernelPCAModel, kernel_pca, pca
from .preprocess import convert_to_array

app = FastAPI(
    title="Dimensionality Reduction",
    description="Offers dimensionality reduction and manifold learning techniques",
    version=__version__,
)

logger = logging.getLogger("api")


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>Dimensionality reduction</title>
        </head>
        <h1> Dimensionality reduction </h1>
        <body>
            <p>This webservice provides dimensionality reduction and manifold learning techniques.</p>
            <p>Find the docs at <a href="./docs">/docs</a> and the openAPI specfication at <a href="./openapi.json">/openapi.json</a>.</p>
        </body>
    </html>
    """


@app.get("/version")
def read_version():
    return {"version": __version__}


def validate_array(array, standardize):
    try:
        array = convert_to_array(array, standardize)
    except ValueError as excep:
        logger.error("Not a valid array {}".format(excep))
        raise HTTPException(status_code=400, detail="Not a valid array")
    except ValidationError as excep:
        logger.error("Not a valid array {}".format(excep))
        raise HTTPException(status_code=400, detail="Not a valid array")

    return array


@app.post("/pca", response_model=DimensionalityReductionResponse)
def run_pca(parameters: DimensionalityReduction):

    array = validate_array(parameters.array, parameters.standardize)

    try:
        return pca(array, parameters.n_components)
    except Exception as excep:
        logger.error("PCA failed {}".format(excep))
        raise HTTPException(status_code=400, detail="PCA failed")


@app.post("/kernelpca", response_model=DimensionalityReductionResponse)
def run_kernelpca(parameters: KernelPCAModel):
    array = validate_array(parameters.array, parameters.standardize)

    try:
        return kernel_pca(
            array,
            parameters.n_components,
            parameters.kernel,
            parameters.gamma,
            parameters.degree,
            parameters.coef,
            parameters.alpha,
        )
    except Exception as excep:
        logger.error("Kernel PCA failed {}".format(excep))
        raise HTTPException(status_code=400, detail="Kernel PCA failed")


@app.post("/umap", response_model=DimensionalityReductionResponse)
def run_umap(parameters: UMAPModel):
    array = validate_array(parameters.array, parameters.standardize)

    try:
        return umap(
            array,
            parameters.n_components,
            parameters.n_neighbors,
            parameters.min_dist,
            parameters.metric,
        )
    except Exception:
        raise HTTPException(status_code=400, detail="UMAP failed")


@app.post("/tsne", response_model=DimensionalityReductionResponse)
def run_tsne(parameters: TSNEModel):
    array = validate_array(parameters.array, parameters.standardize)

    try:
        return tsne(
            array,
            parameters.n_components,
            parameters.perplexity,
            parameters.early_exaggeration,
            parameters.learning_rate,
            parameters.n_iter,
            parameters.n_iter_without_progress,
            parameters.min_grad_norm,
            parameters.metric,
        )
    except Exception:
        raise HTTPException(status_code=400, detail="TSNE failed")


@app.post("/isomap", response_model=DimensionalityReductionResponse)
def run_isomap(parameters: IsomapModel):
    array = validate_array(parameters.array, parameters.standardize)
    try:
        return isomap(array, parameters.n_components, parameters.n_neighbors)
    except Exception:
        raise HTTPException(status_code=400, detail="Isomap failed")


@app.post("/lle", response_model=DimensionalityReductionResponse)
def run_lle(parameters: LLEModel,):
    array = validate_array(parameters.array, parameters.standardize)
    try:
        return lle(
            array, parameters.n_components, parameters.n_neighbors, parameters.reg
        )
    except Exception:
        raise HTTPException(status_code=400, detail="LLE failed")
