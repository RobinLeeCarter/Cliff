from __future__ import annotations
import dataclasses
from typing import Optional

from common import enums


@dataclasses.dataclass
class AlgorithmParameters:
    algorithm_type: Optional[enums.AlgorithmType] = None
    alpha: Optional[float] = None
    alpha_variable: Optional[bool] = None
    initial_v_value: Optional[float] = None
    initial_q_value: Optional[float] = None
    verbose: Optional[bool] = None


precedence_attribute_names: list[str] = [
    'algorithm_type',
    'alpha',
    'alpha_variable',
    'initial_v_value',
    'initial_q_value',
    'verbose'
]


def default_factory() -> AlgorithmParameters:
    return AlgorithmParameters()
