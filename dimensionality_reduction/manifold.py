# -*- coding: utf-8 -*-
from math import e

import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP


def umap(
    array: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
):

    umap_instance = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)

    X_transformed = umap_instance.fit_transform()

    return {"projection": X_transformed.tolist()}


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

    X_transformed = tsne_instance.fit_transform()

    return {"projection": X_transformed.tolist()}
