from __future__ import annotations
import dataclasses

from mdp.model.environment.non_tabular.dimension import Dimension


@dataclasses.dataclass
class CategoryDimension(Dimension):
    possible_values: int = 0
