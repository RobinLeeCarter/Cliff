from __future__ import annotations

from mdp.model.policy import policy


class NoPolicy(policy.Policy):
    def _get_a(self, s: int) -> int:
        return -1        # None

    def __setitem__(self, s: int, a: int):
        super().__setitem__(s, a)

    def _calc_probability(self, s: int, a: int) -> float:
        if a == -1:
            return 1.0
        else:
            return 0.0
