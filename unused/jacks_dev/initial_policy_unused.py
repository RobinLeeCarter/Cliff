from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common

from mdp.model import policy
from mdp.scenarios.jacks.model import action, environment


def get_initial_policy(environment_: environment.Environment,
                       policy_parameters: common.PolicyParameters)\
                        -> policy.Deterministic:
    initial_policy = policy.Deterministic(environment_, policy_parameters)
    initial_action = action.Action(transfer_1_to_2=0)
    for state in environment_.states:
        initial_policy[state] = initial_action
    return initial_policy
