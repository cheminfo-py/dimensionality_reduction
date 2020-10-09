# -*- coding: utf-8 -*-
from typing import Optional

from pydantic import BaseModel


class DimensionalityReduction(BaseModel):
    array: list
    n_components: Optional[int] = 2
    standardize: Optional[bool] = True
