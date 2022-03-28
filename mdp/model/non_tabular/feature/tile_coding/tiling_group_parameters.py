from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np

from mdp.model.non_tabular.environment.dimension.dim_enum import DimEnum


@dataclass
class TilingGroupParameters:
    included_dims: set[DimEnum] = field(default_factory=set)
    tile_size_per_dim: Optional[dict[DimEnum, float]] = None,
    tiles_per_dim: Optional[dict[DimEnum, int]] = None,
    tilings: Optional[int] = None,
    offset_per_dimension_fn: Optional[Callable[[int], np.ndarray]] = None
