from __future__ import annotations
from typing import Optional
import dataclasses
import copy


from common import enums


@dataclasses.dataclass
class EnvironmentParameters:
    environment_type: Optional[enums.EnvironmentType] = None
    actions_list: enums.ActionsList = enums.ActionsList.FOUR_MOVES
    random_wind: Optional[bool] = None
    verbose: Optional[bool] = None

    def apply_default_to_nones(self, default_: EnvironmentParameters):
        attribute_names: list[str] = [
            'environment_type',
            'actions_list',
            'random_wind',
            'verbose'
        ]
        for attribute_name in attribute_names:
            attribute = self.__getattribute__(attribute_name)
            if attribute is None:
                default_value = default_.__getattribute__(attribute_name)
                self.__setattr__(attribute_name, default_value)


default: EnvironmentParameters = EnvironmentParameters(
    actions_list=enums.ActionsList.FOUR_MOVES,
    random_wind=False,
    verbose=False,
)


def default_factory() -> EnvironmentParameters:
    return copy.deepcopy(default)
