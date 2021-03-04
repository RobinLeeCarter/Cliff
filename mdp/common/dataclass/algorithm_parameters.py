from __future__ import annotations
from typing import Optional
import dataclasses

from mdp.common import enums


@dataclasses.dataclass(eq=False)    # sacrifice so it can be hashed (using id is bad if eq is defined)
class AlgorithmParameters:
    algorithm_type: Optional[enums.AlgorithmType] = None
    alpha: Optional[float] = None
    alpha_variable: Optional[bool] = None
    initial_v_value: Optional[float] = None
    initial_q_value: Optional[float] = None
    verbose: Optional[bool] = None


default: AlgorithmParameters = AlgorithmParameters(
    initial_v_value=0.0,
    initial_q_value=0.0,
    verbose=False,
)


def none_factory() -> AlgorithmParameters:
    return AlgorithmParameters()
