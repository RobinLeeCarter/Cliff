from __future__ import annotations

from mdp import common
from mdp.model.tabular.policy.tabular_policy import TabularPolicy


class NoPolicy(TabularPolicy,
               policy_type=common.PolicyType.TABULAR_NONE,
               policy_name="No policy"):
    def _get_a(self, s: int) -> int:
        return -1        # None

    def __setitem__(self, s: int, a: int):
        super().__setitem__(s, a)

    def _calc_probability(self, s: int, a: int) -> float:
        if a == -1:
            return 1.0
        else:
            return 0.0
