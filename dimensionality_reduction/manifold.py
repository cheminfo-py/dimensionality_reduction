# -*- coding: utf-8 -*-
from typing import Optional
import numpy as np
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from umap import UMAP
from .model import DimensionalityReduction
from . import __version__


class UMAPModel(DimensionalityReduction):
    n_neighbors: Optional[int] = 15
    min_dist: Optional[float] = 0.1
    metric: Optional[str] = "euclidean"


class TSNEModel(DimensionalityReduction):
    perplexity: Optional[float] = 30.0
    early_exaggeration: Optional[float] = 12.0
    learning_rate: Optional[float] = 200.0
    n_iter: Optional[int] = 1000
    n_iter_without_progress: Optional[int] = 300
    min_grad_norm: Optional[float] = 1e-07
    metric: Optional[str] = "euclidean"


class IsomapModel(DimensionalityReduction):
    n_neighbors: Optional[int] = 5


class LLEModel(DimensionalityReduction):
    n_neighbors: Optional[int] = 5
    reg: Optional[float] = 0.001


def umap(
    array: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
):

    umap_instance = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )

    X_transformed = umap_instance.fit_transform(array)

    return {"projection": X_transformed.tolist(), "api_version": __version__}


def tsne(
    array: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    early_exaggeration: float = 12.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    n_iter_without_progress: int = 300,
    min_grad_norm: float = 1e-07,
    metric: str = "euclidean",
):
    tsne_instance = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        n_iter=n_iter,
        n_iter_without_progress=n_iter_without_progress,
        min_grad_norm=min_grad_norm,
        metric=metric,
    )

    X_transformed = tsne_instance.fit_transform(array)

    return {"projection": X_transformed.tolist(), "api_version": __version__}


def isomap(
    array: np.ndarray, n_components: int = 2, n_neighbors: int = 5,
):
    isomap_instance = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    X_transformed = isomap_instance.fit_transform(array)
    return {"projection": X_transformed.tolist(), "api_version": __version__}


def lle(array, n_components: int = 2, n_neighbors: int = 4, reg: float = 0.0001):

    lle_instance = LocallyLinearEmbedding(
        n_components=n_components, n_neighbors=n_neighbors, reg=reg
    )
    
    X_transformed = lle_instance.fit_transform(array)
    return {"projection": X_transformed.tolist(), "api_version": __version__}

