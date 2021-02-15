from __future__ import annotations
from typing import Optional
import dataclasses
import copy


@dataclasses.dataclass
class GraphParameters:
    moving_average_window_size: Optional[int] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None

    def apply_default_to_nones(self, default_: GraphParameters):
        attribute_names: list[str] = [
            'moving_average_window_size',
            'y_min',
            'y_max'
        ]
        for attribute_name in attribute_names:
            attribute = self.__getattribute__(attribute_name)
            if attribute is None:
                default_value = default_.__getattribute__(attribute_name)
                self.__setattr__(attribute_name, default_value)


default: GraphParameters = GraphParameters()


def default_factory() -> GraphParameters:
    return copy.deepcopy(default)
