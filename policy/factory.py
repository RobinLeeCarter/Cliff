from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import environment
    from policy import policy_
import common
from policy import deterministic, e_greedy, no_policy, random


def factory(environment_: environment.Environment, policy_parameters: common.PolicyParameters) -> policy_.Policy:
    pt = common.PolicyType
    policy_type = policy_parameters.policy_type
    if policy_type == pt.DETERMINISTIC:
        policy = deterministic.Deterministic(environment_)
    elif policy_type == pt.NONE:
        policy = no_policy.NoPolicy(environment_)
    elif policy_type == pt.RANDOM:
        policy = random.Random(environment_)
    elif policy_type == pt.E_GREEDY:
        policy = e_greedy.EGreedy(environment_, policy_parameters.epsilon)
    else:
        raise NotImplementedError
    return policy
