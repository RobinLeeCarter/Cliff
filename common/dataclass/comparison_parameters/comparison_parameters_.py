from __future__ import annotations
from typing import Optional
import dataclasses
import copy

from common import enums


@dataclasses.dataclass
class ComparisonParameters:
    comparison_type: Optional[enums.ComparisonType] = None
    verbose: Optional[bool] = None

    def apply_default_to_nones(self, default_: ComparisonParameters):
        attribute_names: list[str] = [
            'comparison_type',
            'verbose'
        ]
        for attribute_name in attribute_names:
            attribute = self.__getattribute__(attribute_name)
            if attribute is None:
                default_value = default_.__getattribute__(attribute_name)
                self.__setattr__(attribute_name, default_value)


default: ComparisonParameters = ComparisonParameters(
    verbose=False
)


def default_factory() -> ComparisonParameters:
    return copy.deepcopy(default)
