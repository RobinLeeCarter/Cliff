from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar

from abc import ABC

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment
    from mdp.model.non_tabular.algorithm.value_function.state_action_function import StateActionFunction
from mdp.model.non_tabular.policy.non_tabular_policy import NonTabularPolicy
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class ActionValuePolicy(NonTabularPolicy[State, Action], ABC):
    def __init__(self,
                 environment: NonTabularEnvironment[State, Action],
                 policy_parameters: common.PolicyParameters,
                 state_action_function: StateActionFunction,
                 ):
        super().__init__(environment, policy_parameters)
        self._state_action_function: StateActionFunction = state_action_function
