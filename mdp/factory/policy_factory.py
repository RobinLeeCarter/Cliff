from __future__ import annotations
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from mdp.model.base.environment.base_environment import BaseEnvironment
from mdp import common
from mdp.model.base.policy.base_policy import BasePolicy

# Tabular
from mdp.model.tabular.policy.e_greedy import EGreedy
from mdp.model.tabular.policy.random import Random
from mdp.model.tabular.policy.deterministic import Deterministic
from mdp.model.tabular.policy.no_policy import NoPolicy

# Non-tabular
from mdp.model.non_tabular.policy.action_value.non_tabular_e_greedy import NonTabularEGreedy
from mdp.model.non_tabular.policy.parameterized.softmax_linear import SoftmaxLinear


class PolicyFactory:
    def __init__(self, environment: BaseEnvironment):
        self._environment: BaseEnvironment = environment

    def create(self, policy_parameters: common.PolicyParameters) -> BasePolicy:
        policy_type: common.PolicyType = policy_parameters.policy_type
        type_of_policy: Type[BasePolicy] = BasePolicy.type_registry[policy_type]
        policy: BasePolicy = type_of_policy(self._environment, policy_parameters)
        return policy

    def lookup_policy_name(self, policy_type: common.PolicyType) -> str:
        return BasePolicy.name_registry[policy_type]


def __dummy():
    """Stops Pycharm objecting to imports. The imports are needed to generate the registry."""
    return [
        EGreedy,
        Random,
        Deterministic,
        NoPolicy,
        NonTabularEGreedy,
        SoftmaxLinear
    ]
