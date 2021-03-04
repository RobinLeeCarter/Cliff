from __future__ import annotations
from typing import Optional
import dataclasses
import copy

import numpy as np

from mdp import common


@dataclasses.dataclass
class EnvironmentParameters(common.EnvironmentParameters):
    track: Optional[np.ndarray] = None
    min_velocity: Optional[int] = None
    max_velocity: Optional[int] = None
    min_acceleration: Optional[int] = None
    max_acceleration: Optional[int] = None
    extra_reward_for_failure: Optional[float] = None
    skid_probability: Optional[float] = None

    def apply_default_to_nones(self, default_: EnvironmentParameters):
        super().apply_default_to_nones(default_)
        attribute_names: list[str] = [
            'track',
            'min_velocity',
            'max_velocity',
            'verbose'
        ]
        for attribute_name in attribute_names:
            attribute = self.__getattribute__(attribute_name)
            if attribute is None:
                default_value = default_.__getattribute__(attribute_name)
                self.__setattr__(attribute_name, default_value)


default: EnvironmentParameters = EnvironmentParameters(
    actions_list=common.ActionsList.FOUR_MOVES,
    random_wind=False,
    verbose=False,
)


def default_factory() -> EnvironmentParameters:
    return copy.deepcopy(default)
