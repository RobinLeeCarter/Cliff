from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Optional
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
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment, policy_parameters)
        self._feature: Optional[Feature[State, Action]] = None
        self._initial_theta: float = policy_parameters.initial_theta
        self._theta: np.ndarray = np.empty(0, dtype=float)
        self._Q: Optional[StateActionFunction[State, Action]] = None
        self.requires_feature = True
        self.requires_q = True

    def set_feature(self, feature: Feature[State, Action]):
        self._feature: Feature[State, Action] = feature
        self._theta = np.full(shape=self._feature.max_size, fill_value=self._initial_theta, dtype=float)

    def set_state_action_function(self, state_action_function: StateActionFunction[State, Action]):
        self._Q: StateActionFunction[State, Action] = state_action_function
