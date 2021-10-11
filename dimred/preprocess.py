# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import StandardScaler


def convert_to_array(array: list, standardize: bool = False) -> np.ndarray:
    try:
        array = np.matrix(array)
        assert len(array.shape) == 2

    except Exception as execp:
        raise ValueError(
            f"Make sure the input is a list of list that can be mapped to a numpy array with two dimensions, {execp}"
        )

    if standardize:
        scaler = StandardScaler()
        array = scaler.fit_transform(array)

    return array
