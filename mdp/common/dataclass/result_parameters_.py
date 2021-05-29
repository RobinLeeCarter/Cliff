from __future__ import annotations
from typing import Optional
import dataclasses


@dataclasses.dataclass(eq=False)    # sacrifice so it can be hashed (using id is bad if eq is defined)
class ResultParameters:
    return_recorder: Optional[bool] = None
    return_policy_vector: Optional[bool] = None
    return_v: Optional[bool] = None
    return_q: Optional[bool] = None


default: ResultParameters = ResultParameters(
    return_recorder=True,
    return_policy_vector=False,
    return_v=False,
    return_q=False
)


def none_factory() -> ResultParameters:
    return ResultParameters()
