from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from mdp.model.policy.general_policy import GeneralPolicy
from mdp.model.environment.general.general_environment import GeneralEnvironment
from mdp import common
from mdp.model.policy.tabular.e_greedy import EGreedy
from mdp.model.policy.tabular.random import Random
from mdp.model.policy.tabular.deterministic import Deterministic
from mdp.model.policy.tabular.no_policy import NoPolicy

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
