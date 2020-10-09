# -*- coding: utf-8 -*-
from . import __version__
from typing import Optional

from pydantic import BaseModel, validator


class DimensionalityReduction(BaseModel):
    array: list
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

