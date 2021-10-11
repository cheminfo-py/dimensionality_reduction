# -*- coding: utf-8 -*-
from typing import Optional

from pydantic import BaseModel, validator

from . import __version__


class DimensionalityReduction(BaseModel):
    matrix: list
    n_components: Optional[int] = 2
    standardize: Optional[bool] = True

    @validator("n_components")
    def n_components_must_be_greater_0(cls, v):
        if v < 1:
            raise ValueError("n_components must be >= 1")
        return v


class DimensionalityReductionResponse(BaseModel):
    projection: list
    api_version: Optional[str] = __version__
