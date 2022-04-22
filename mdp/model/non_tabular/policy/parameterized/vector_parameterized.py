from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Optional
from abc import ABC

import numpy as np

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment
    from mdp.model.non_tabular.feature.base_feature import BaseFeature
from mdp.model.non_tabular.policy.non_tabular_policy import NonTabularPolicy
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class VectorParameterized(NonTabularPolicy[State, Action], ABC,
                          requires_feature=True):
    def __init__(self,
                 environment: NonTabularEnvironment,
                 policy_parameters: common.PolicyParameters,
                 ):
        super().__init__(environment, policy_parameters)
        self._feature: Optional[BaseFeature[State, Action]] = None
        self._has_sparse_feature: bool = False
        self._initial_theta: float = policy_parameters.initial_theta
        self._theta: np.ndarray = np.empty(0, dtype=float)

    def set_feature(self, feature: BaseFeature[State, Action]):
        self._feature: BaseFeature[State, Action] = feature
        self._has_sparse_feature = self._feature.is_sparse
        self._theta = np.full(shape=self._feature.max_size, fill_value=self._initial_theta, dtype=float)
