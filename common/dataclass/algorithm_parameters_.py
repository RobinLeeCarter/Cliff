from __future__ import annotations
import dataclasses
from typing import Optional


@dataclasses.dataclass
class AlgorithmParameters:
    alpha: Optional[float] = None
    alpha_variable: Optional[bool] = None
    initial_v_value: Optional[float] = None
    initial_q_value: Optional[float] = None
    verbose: Optional[bool] = None


precedence_attribute_names: list[str] = [
    'alpha',
    'alpha_variable',
    'initial_v_value',
    'initial_q_value',
    'verbose'
]


def default_factory() -> AlgorithmParameters:
    return AlgorithmParameters()
