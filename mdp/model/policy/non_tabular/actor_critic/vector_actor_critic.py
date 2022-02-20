from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC

import numpy as np

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.environment.non_tabular.non_tabular_environment import NonTabularEnvironment
    from mdp.model.algorithm.non_tabular.value_function.state_action_function import StateActionFunction
    from mdp.model.feature.feature import Feature
from mdp.model.policy.non_tabular.non_tabular_policy import NonTabularPolicy


class VectorActorCritic(NonTabularPolicy, ABC):
    def __init__(self,
                 environment: NonTabularEnvironment,
                 policy_parameters: common.PolicyParameters,
                 state_action_function: StateActionFunction,
                 feature: Feature,
                 initial_theta: float = 0.0,
                 ):
        super().__init__(environment, policy_parameters)
        self._state_action_function: StateActionFunction = state_action_function
        self._feature: Feature = feature
        self._theta: np.ndarray = np.full(shape=feature.max_size, fill_value=initial_theta, dtype=float)
