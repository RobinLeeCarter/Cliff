from __future__ import annotations
from typing import Optional

from mdp.model.policy import policy


class NoPolicy(policy.Policy):
    def _get_a(self, s: int) -> Optional[int]:
        return None

    def __setitem__(self, s: int, a: int):
        super().__setitem__(s, a)

    def _calc_probability(self, s: int, a: Optional[int]) -> float:
        if a is None:
            return 1.0
        else:
            return 0.0
