from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar
from abc import ABC

import numpy as np

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment
    from mdp.model.non_tabular.value_function.state_action_function import StateActionFunction
    from mdp.model.non_tabular.feature.feature import Feature
from mdp.model.non_tabular.policy.non_tabular_policy import NonTabularPolicy
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class VectorActorCritic(NonTabularPolicy[State, Action], ABC):
    def __init__(self,
                 environment: NonTabularEnvironment[State, Action],
                 policy_parameters: common.PolicyParameters,
                 state_action_function: StateActionFunction[State, Action],
                 feature: Feature,
                 initial_theta: float = 0.0,
                 ):
        super().__init__(environment, policy_parameters)
        self._state_action_function: StateActionFunction[State, Action] = state_action_function
        self._feature: Feature = feature
        self._theta: np.ndarray = np.full(shape=feature.max_size, fill_value=initial_theta, dtype=float)
