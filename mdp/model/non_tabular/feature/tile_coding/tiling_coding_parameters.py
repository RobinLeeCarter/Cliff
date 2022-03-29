from __future__ import annotations
from dataclasses import dataclass, field

from mdp import common
from mdp.model.non_tabular.feature.tile_coding.tiling_group_parameters import TilingGroupParameters


@dataclass
class TileCodingParameters(common.FeatureParameters):
    # specify either tiling_group or tiling_groups but not both
    tiling_group: TilingGroupParameters = field(default_factory=TilingGroupParameters)
    tiling_groups: list[TilingGroupParameters] = field(default_factory=list)
    use_dict: bool = True   # pass False if wish to [Hash mod max_size] and avoid using a dict

    def __post_init__(self):
        if not self.tiling_groups:
            self.tiling_groups = [self.tiling_group]
