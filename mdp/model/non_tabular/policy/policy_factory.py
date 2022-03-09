from __future__ import annotations
from typing import TYPE_CHECKING, Type, TypeVar, Generic

if TYPE_CHECKING:
    from mdp.model.tabular.environment.tabular_environment import TabularEnvironment
from mdp import common
from mdp.model.non_tabular.policy.non_tabular_policy import NonTabularPolicy
from mdp.model.non_tabular.environment.non_tabular_state import NonTabularState
from mdp.model.non_tabular.environment.non_tabular_action import NonTabularAction


State = TypeVar('State', bound=NonTabularState)
Action = TypeVar('Action', bound=NonTabularAction)


class PolicyFactory(Generic[State, Action]):
    def __init__(self, environment: TabularEnvironment[State, Action]):
        self._environment: TabularEnvironment[State, Action] = environment

        self._policy_lookup: dict[common.PolicyType, Type[NonTabularPolicy]] = {}
        # register_control_algorithms(self.register)

        p = common.PolicyType
        self._policy_lookup: dict[p, Type[NonTabularPolicy]] = {
        }

    def register(self, type_for_policy: Type[NonTabularPolicy]):
        self._policy_lookup[type_for_policy.policy_type] = type_for_policy

    def create(self, policy_parameters: common.Settings.policy_parameters) -> NonTabularPolicy:
        type_for_policy: Type[NonTabularPolicy] = self._policy_lookup[policy_parameters.policy_type]
        policy: NonTabularPolicy = type_for_policy[State, Action](self._environment, policy_parameters)
        return policy
