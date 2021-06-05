from __future__ import annotations
import dataclasses


@dataclasses.dataclass
class ResultParameters:
    return_recorder: bool = False
    return_algorithm_title: bool = False
    return_policy_vector: bool = False
    return_v_vector: bool = False
    return_q_matrix: bool = False
    return_cum_timestep: bool = False
