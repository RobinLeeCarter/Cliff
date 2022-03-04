from __future__ import annotations
import dataclasses

from mdp.model.non_tabular.environment.dimension.dimension import Dimension


@dataclasses.dataclass
class CategoryDimension(Dimension):
    possible_values: int
