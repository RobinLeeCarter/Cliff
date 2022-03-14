from __future__ import annotations
from typing import Type

from mdp.model.non_tabular.policy.non_tabular_policy import NonTabularPolicy
from mdp import common
from mdp.model.non_tabular.policy.action_value.e_greedy_linear import EGreedyLinear
from mdp.model.non_tabular.policy.parameterized.softmax_linear import SoftmaxLinear


def get_policy_lookup() -> dict[common.PolicyType, Type[NonTabularPolicy]]:
    pt = common.PolicyType
    return {
        pt.E_GREEDY_LINEAR: EGreedyLinear,
        pt.SOFTMAX_LINEAR: SoftmaxLinear
    }


def get_name_lookup() -> dict[common.PolicyType, str]:
    pt = common.PolicyType
    return {
        pt.E_GREEDY_LINEAR: 'ε-greedy linear',
        pt.SOFTMAX_LINEAR: 'softmax linear'
    }
