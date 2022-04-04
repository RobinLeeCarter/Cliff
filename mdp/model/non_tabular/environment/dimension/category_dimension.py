from __future__ import annotations
from dataclasses import dataclass

from mdp.model.non_tabular.environment.dimension.dimension import Dimension


@dataclass
class CategoryDimension(Dimension):
    possible_values: list
