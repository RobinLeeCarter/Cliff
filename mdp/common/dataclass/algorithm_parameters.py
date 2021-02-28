from __future__ import annotations
from typing import Optional
import dataclasses
# import copy

from mdp.common import enums


@dataclasses.dataclass(eq=False)    # sacrifice so it can be hashed (using id is bad if eq is defined)
class AlgorithmParameters:
    algorithm_type: Optional[enums.AlgorithmType] = None
    alpha: Optional[float] = None
    alpha_variable: Optional[bool] = None
    initial_v_value: Optional[float] = None
    initial_q_value: Optional[float] = None
    verbose: Optional[bool] = None

    def apply_default_to_nones(self, default_: AlgorithmParameters):
        attribute_names: list[str] = [
            'algorithm_type',
            'alpha',
            'alpha_variable',
            'initial_v_value',
            'initial_q_value',
            'verbose'
        ]
        for attribute_name in attribute_names:
            attribute = self.__getattribute__(attribute_name)
            if attribute is None:
                default_value = default_.__getattribute__(attribute_name)
                self.__setattr__(attribute_name, default_value)


default: AlgorithmParameters = AlgorithmParameters(
    initial_v_value=0.0,
    initial_q_value=0.0,
    verbose=False,
)


def none_factory() -> AlgorithmParameters:
    return AlgorithmParameters()
