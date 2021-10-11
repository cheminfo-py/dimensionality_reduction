# -*- coding: utf-8 -*-
from typing import Literal, Optional

import numpy as np
from sklearn.decomposition import PCA, KernelPCA

from . import __version__
from .model import DimensionalityReduction


class KernelPCAModel(DimensionalityReduction):
    kernel: Optional[str] = Literal["linear", "poly", "rbf", "sigmoid", "cosine"]
    gamma: Optional[float] = None
    degree: Optional[float] = None
    coef: Optional[float] = None
    alpha: Optional[float] = None


def kernel_pca(
    array: np.ndarray,
    n_components: int,
    kernel: str,
    gamma: float,
    degree: int,
    coef0: float,
    alpha: float,
):

    kernel_pca_instance = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        alpha=alpha,
    )

    X_transformed = kernel_pca_instance.fit_transform(array)

    return {"projection": X_transformed.tolist(), "api_version": __version__}


def pca(array: np.ndarray, n_components: int):

    pca_instance = PCA(n_components=n_components)

    X_transformed = pca_instance.fit_transform(array)

    return {"projection": X_transformed.tolist(), "api_version": __version__}
