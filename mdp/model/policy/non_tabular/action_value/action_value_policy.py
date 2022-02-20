from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.environment.non_tabular.non_tabular_environment import NonTabularEnvironment
    from mdp.model.algorithm.non_tabular.value_function.state_action_function import StateActionFunction
from mdp.model.policy.non_tabular.non_tabular_policy import NonTabularPolicy


class ActionValuePolicy(NonTabularPolicy, ABC):
    def __init__(self,
                 environment: NonTabularEnvironment,
                 policy_parameters: common.PolicyParameters,
                 state_action_function: StateActionFunction,
                 ):
        super().__init__(environment, policy_parameters)
        self._state_action_function: StateActionFunction = state_action_function
