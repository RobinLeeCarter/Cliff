from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.policy.policy import Policy
from mdp import common
from mdp.model.policy.e_greedy import EGreedy
from mdp.model.policy.random import Random
from mdp.model.policy.deterministic import Deterministic
from mdp.model.policy.no_policy import NoPolicy


def policy_factory(environment_: Environment, policy_parameters: common.PolicyParameters) -> Policy:
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
