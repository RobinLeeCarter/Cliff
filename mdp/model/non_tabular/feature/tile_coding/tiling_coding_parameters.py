from __future__ import annotations
from dataclasses import dataclass, field

from mdp import common
from mdp.model.non_tabular.feature.tile_coding.tiling_group_parameters import TilingGroupParameters


@dataclass
class TileCodingParameters(common.FeatureParameters):
    tiling_groups: list[TilingGroupParameters] = field(default_factory=list)
    use_dict: bool = True
