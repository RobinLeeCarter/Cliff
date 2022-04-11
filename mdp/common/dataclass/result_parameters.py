from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ResultParameters:
    return_recorder: bool = False
    return_policy_vector: bool = False
    return_v_vector: bool = False
    return_q_matrix: bool = False
    return_cum_timestep: bool = False
    return_delta_w_vector: bool = False
    return_trajectories: bool = False
