from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from mdp.common.enums import AlgorithmType


@dataclass(unsafe_hash=True)    # needed for multiprocessing where results may differ, potentially pickle
class AlgorithmParameters:
    algorithm_type: Optional[AlgorithmType] = None
    verbose: bool = False
    alpha: float = 0.1
    alpha_variable: bool = False
    initial_v_value: float = 0.0
    initial_q_value: float = 0.0

    theta: float = 0.1
    iteration_timeout: int = 1000

    first_visit: bool = False
    exploring_starts: bool = False

    derive_v_from_q_as_final_step: bool = False
