from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model import environment
    from model.policy import policy_, deterministic
import common
from model.policy import e_greedy, no_policy, random


def factory(environment_: environment.Environment, policy_parameters: common.PolicyParameters) -> policy_.Policy:
    pt = common.PolicyType
    policy_type = policy_parameters.policy_type
    if policy_type == pt.DETERMINISTIC:
        policy = deterministic.Deterministic(environment_, policy_parameters)
    elif policy_type == pt.NONE:
        policy = no_policy.NoPolicy(environment_, policy_parameters)
    elif policy_type == pt.RANDOM:
        policy = random.Random(environment_, policy_parameters)
    elif policy_type == pt.E_GREEDY:
        policy = e_greedy.EGreedy(environment_, policy_parameters)
    else:
        raise NotImplementedError
    return policy
