from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Optional

from abc import ABC

if TYPE_CHECKING:
    from mdp import common
    from mdp.model.non_tabular.environment.non_tabular_environment import NonTabularEnvironment
    from mdp.model.non_tabular.value_function.state_action.state_action_function import StateActionFunction
from mdp.model.non_tabular.policy.non_tabular_policy import NonTabularPolicy
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction

State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class ActionValuePolicy(NonTabularPolicy[State, Action], ABC,
                        requires_q=True):
    def __init__(self,
                 environment: NonTabularEnvironment[State, Action],
                 policy_parameters: common.PolicyParameters,
                 ):
        super().__init__(environment, policy_parameters)
        self._Q: Optional[StateActionFunction[State, Action]] = None

    def set_state_action_function(self, state_action_function: StateActionFunction[State, Action]):
        self._Q: StateActionFunction[State, Action] = state_action_function
