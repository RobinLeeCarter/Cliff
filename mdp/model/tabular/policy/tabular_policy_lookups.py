from __future__ import annotations
from typing import Type

from mdp.model.tabular.policy.tabular_policy import TabularPolicy
from mdp import common
from mdp.model.tabular.policy.e_greedy import EGreedy
from mdp.model.tabular.policy.random import Random
from mdp.model.tabular.policy.deterministic import Deterministic
from mdp.model.tabular.policy.no_policy import NoPolicy


def get_policy_lookup() -> dict[common.PolicyType, Type[TabularPolicy]]:
    pt = common.PolicyType
    return {
        pt.TABULAR_DETERMINISTIC: Deterministic,
        pt.TABULAR_NONE: NoPolicy,
        pt.TABULAR_RANDOM: Random,
        pt.TABULAR_E_GREEDY: EGreedy
    }


def get_name_lookup() -> dict[common.PolicyType, str]:
    pt = common.PolicyType
    return {
        pt.TABULAR_DETERMINISTIC: 'Deterministic',
        pt.TABULAR_E_GREEDY: 'ε-greedy',
        pt.TABULAR_NONE: 'No policy',
        pt.TABULAR_RANDOM: 'Random'
    }
