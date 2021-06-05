from __future__ import annotations
from typing import Optional
import dataclasses


@dataclasses.dataclass
class ResultParameters:
    return_recorder: Optional[bool] = None
    return_algorithm_title: Optional[bool] = None
    return_policy_vector: Optional[bool] = None
    return_v_vector: Optional[bool] = None
    return_q_matrix: Optional[bool] = None
    return_cum_timestep: Optional[bool] = None


default: ResultParameters = ResultParameters(
    return_recorder=False,
    return_algorithm_title=False,
    return_policy_vector=False,
    return_v_vector=False,
    return_q_matrix=False,
    return_cum_timestep=False
)


def none_factory() -> ResultParameters:
    return ResultParameters()
