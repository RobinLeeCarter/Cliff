from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field

if TYPE_CHECKING:
    from mdp.model.non_tabular.environment.dimension.dim_enum import DimEnum
    from mdp.model.non_tabular.environment.dimension.float_dimension import FloatDimension
    from mdp.model.non_tabular.environment.dimension.category_dimension import CategoryDimension


@dataclass
class Dims:
    """
    structure to holds all the information about the dimensions of the environment
    note that dicts are documented as ordered in insertion order from Python 3.7 and this is relied upon
    """
    state_float_dims: dict[DimEnum, FloatDimension] = field(default_factory=dict)
    state_categories: dict[DimEnum, CategoryDimension] = field(default_factory=dict)
    action_categories: dict[DimEnum, CategoryDimension] = field(default_factory=dict)
