from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from mdp.common.enums import BatchEpisodes


@dataclass
class ResultParameters:
    return_recorder: bool = False
    return_policy_vector: bool = False
    return_v_vector: bool = False
    return_q_matrix: bool = False
    return_cum_timestep: bool = False
    return_batch_episodes: Optional[BatchEpisodes] = None
