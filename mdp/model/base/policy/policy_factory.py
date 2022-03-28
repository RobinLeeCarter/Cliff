from __future__ import annotations
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from mdp.model.base.policy.base_policy import BasePolicy
from mdp.model.base.environment.base_environment import BaseEnvironment    # will this create a circular dep
from mdp import common
from mdp.model.tabular.policy import tabular_policy_lookups
from mdp.model.non_tabular.policy import non_tabular_policy_lookups


class PolicyFactory:
    def __init__(self, environment: BaseEnvironment):
        self._environment: BaseEnvironment = environment

        self._policy_lookup: dict[common.PolicyType, Type[BasePolicy]] = {}
        self._policy_lookup.update(tabular_policy_lookups.get_policy_lookup())
        self._policy_lookup.update(non_tabular_policy_lookups.get_policy_lookup())

        self._name_lookup: dict[common.PolicyType, str] = {}
        self._name_lookup.update(tabular_policy_lookups.get_name_lookup())
        self._name_lookup.update(non_tabular_policy_lookups.get_name_lookup())

    def create(self, policy_parameters: common.PolicyParameters) -> BasePolicy:
        policy_type: common.PolicyType = policy_parameters.policy_type
        type_of_policy: Type[BasePolicy] = self._policy_lookup[policy_type]
        # policy_name: str = self._name_lookup[policy_type]

        # TODO: pass policy_name and have it stored to be consistent with algorithm factory
        policy: BasePolicy = type_of_policy(self._environment,
                                            policy_parameters)  # policy_name)
        return policy

    def lookup_policy_name(self, policy_type: common.PolicyType) -> str:
        return self._name_lookup[policy_type]
