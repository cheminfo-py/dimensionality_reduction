import numpy as np
from sklearn.preprocessing import StandardScaler


def convert_to_array(array: list, standardize: bool = False) -> np.ndarray:
    try:
        array = np.array(list)

        assert len(np.array.shape) == 2

    except Exception:
        raise ValueError(
            "Make sure the input is a list of list that can be mapped to a numpy array with two dimensions"
        )

    if standardize:
        scaler = StandardScaler()
        array = scaler.fit_transform(array)

    return array
