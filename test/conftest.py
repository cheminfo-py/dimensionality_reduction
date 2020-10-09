# -*- coding: utf-8 -*-
import pytest
from sklearn.datasets import make_regression


@pytest.fixture(scope="module")
def get_data():
    return make_regression()[0].tolist()
