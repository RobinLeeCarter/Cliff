from __future__ import annotations
from typing import Optional
import dataclasses
import copy

from common import enums


@dataclasses.dataclass
class BreakdownParameters:
    breakdown_type: Optional[enums.BreakdownType] = None
    verbose: Optional[bool] = None

    def apply_default_to_nones(self, default_: BreakdownParameters):
        attribute_names: list[str] = [
            'breakdown_type',
            'verbose'
        ]
        for attribute_name in attribute_names:
            attribute = self.__getattribute__(attribute_name)
            if attribute is None:
                default_value = default_.__getattribute__(attribute_name)
                self.__setattr__(attribute_name, default_value)


default: BreakdownParameters = BreakdownParameters(
    verbose=False
)


def default_factory() -> BreakdownParameters:
    return copy.deepcopy(default)
