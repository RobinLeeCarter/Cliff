from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.tabular.tabular_environment import TabularEnvironment
    from mdp.model.policy.tabular.tabular_policy import TabularPolicy
from mdp import common
from mdp.model.policy.tabular.e_greedy import EGreedy
from mdp.model.policy.tabular.random import Random
from mdp.model.policy.tabular.deterministic import Deterministic
from mdp.model.policy.tabular.no_policy import NoPolicy


def policy_factory(environment_: TabularEnvironment, policy_parameters: common.PolicyParameters) -> TabularPolicy:
    pt = common.PolicyType
    policy_type = policy_parameters.policy_type
    if policy_type == pt.DETERMINISTIC:
        policy = Deterministic(environment_, policy_parameters)
    elif policy_type == pt.NONE:
        policy = NoPolicy(environment_, policy_parameters)
    elif policy_type == pt.RANDOM:
        policy = Random(environment_, policy_parameters)
    elif policy_type == pt.E_GREEDY:
        policy = EGreedy(environment_, policy_parameters)
    else:
        raise NotImplementedError

    return policy
