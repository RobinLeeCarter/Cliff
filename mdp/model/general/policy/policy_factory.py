from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from mdp.model.general.policy.general_policy import GeneralPolicy
from mdp.model.general.environment.general_environment import GeneralEnvironment    # will this create a circular dep
from mdp import common
from mdp.model.tabular.policy.e_greedy import EGreedy
from mdp.model.tabular.policy.random import Random
from mdp.model.tabular.policy.deterministic import Deterministic
from mdp.model.tabular.policy.no_policy import NoPolicy

Environment = TypeVar('Environment', bound=GeneralEnvironment)


def policy_factory(environment: Environment, policy_parameters: common.PolicyParameters) -> GeneralPolicy:
    pt = common.PolicyType
    policy_type = policy_parameters.policy_type
    if policy_type == pt.DETERMINISTIC:
        policy = Deterministic(environment, policy_parameters)
    elif policy_type == pt.NONE:
        policy = NoPolicy(environment, policy_parameters)
    elif policy_type == pt.RANDOM:
        policy = Random(environment, policy_parameters)
    elif policy_type == pt.E_GREEDY:
        policy = EGreedy(environment, policy_parameters)
    else:
        raise NotImplementedError

    return policy
