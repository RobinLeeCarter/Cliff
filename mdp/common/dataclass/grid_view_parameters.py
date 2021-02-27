from __future__ import annotations
from typing import Optional
import dataclasses
import copy


@dataclasses.dataclass
class GridViewParameters:
    window_title: Optional[str] = None
    show_trail: Optional[bool] = None
    show_values: Optional[bool] = None

    def apply_default_to_nones(self, default_: GridViewParameters):
        attribute_names: list[str] = [
            'window_title',
            'show_trail',
            'show_values',
        ]
        for attribute_name in attribute_names:
            attribute = self.__getattribute__(attribute_name)
            if attribute is None:
                default_value = default_.__getattribute__(attribute_name)
                self.__setattr__(attribute_name, default_value)


default: GridViewParameters = GridViewParameters(
    window_title="Grid World",
    show_trail=False,
    show_values=True
)


def default_factory() -> GridViewParameters:
    return copy.deepcopy(default)
