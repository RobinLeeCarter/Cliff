# from __future__ import annotations
# from typing import TYPE_CHECKING
# import random
#
# import numpy as np
# from numba import njit
#
# from mdp import common
# if TYPE_CHECKING:
#     from mdp.model.environment.environment import Environment
# from mdp.model.policy.policy import Policy
# from mdp.model.policy.deterministic import Deterministic


class EGreedy(Policy):
    def __init__(self, environment_: Environment, policy_parameters: common.PolicyParameters):
        super().__init__(environment_, policy_parameters)
        self.epsilon: float = self._policy_parameters.epsilon
        greedy_policy_parameters = common.PolicyParameters(
            policy_type=common.PolicyType.DETERMINISTIC,
            store_matrix=False,
        )
        self.greedy_policy: Deterministic = Deterministic(self._environment, greedy_policy_parameters)

    @property
    def linked_policy(self) -> Deterministic:
        return self.greedy_policy

    def set_policy_vector(self, policy_vector: np.ndarray):
        self.greedy_policy.set_policy_vector(policy_vector)
        if self._store_matrix:
            self._policy_matrix = self._calc_policy_matrix()

    def get_policy_vector(self) -> np.ndarray:
        return self.greedy_policy.policy_vector

    @profile
    def _get_a(self, s: int) -> int:
        if self._store_matrix:
            # numpy_direct_choice = common.rng.choice(
            #     a=len(self._environment.actions),
            #     p=self._policy_matrix[s, :]
            # )
            # numpy_uniform_choice = self._numpy_uniform_choice(s)
            # python_choice = self._python_choice(s)
            # a_greedy = self.greedy_policy[s]
            # compatibility = self._environment.s_a_compatibility[s, :]
            # numba_uniform_choice = _numba_uniform_choice(self.epsilon, a_greedy, compatibility)

            numba_p_unsliced = _numba_p_unsliced(p=self._policy_matrix, s=s)

            numba_p_choice = _numba_p_choice(p=self._policy_matrix[s, :])

            cum_p: np.ndarray = _numba_cum_p(self._policy_matrix[s, :])
            x: float = _numba_get_x()
            numba_p_x_choice = _numba_cum_p_x_choice(cum_p, x)

            return numba_p_x_choice
        else:
            if common.rng.uniform() > self.epsilon:
                return self.greedy_policy[s]
            else:
                return common.rng.choice(
                    np.flatnonzero(self._environment.s_a_compatibility[s, :])
                )

    def __setitem__(self, s: int, a: int):
        if self._store_matrix:
            prev_a = self.greedy_policy[s]
            greedy_p = self._policy_matrix[s, prev_a]
            non_greedy_p = self._policy_matrix[s, a]
            self._policy_matrix[s, prev_a] = non_greedy_p
            self._policy_matrix[s, a] = greedy_p
        self.greedy_policy[s] = a
        # print(f"greedy_policy[{s}] = {self.greedy_policy[s]}")

    def _calc_probability(self, s: int, a: int) -> float:
        non_greedy_p = self.epsilon * self._environment.one_over_possible_actions[s]
        if a == self.greedy_policy[s]:
            greedy_p = (1 - self.epsilon) + non_greedy_p
            return greedy_p
        else:
            return non_greedy_p

    def _calc_probability_vector(self, s: int) -> np.ndarray:
        action_count: int = len(self._environment.actions)
        probability_vector: np.ndarray = np.zeros(shape=action_count, dtype=float)

        non_greedy_p: float = self.epsilon * self._environment.one_over_possible_actions[s]
        greedy_p: float = (1 - self.epsilon) + non_greedy_p

        compatible_actions: np.ndarray = self._environment.s_a_compatibility[s, :]
        probability_vector[compatible_actions] = non_greedy_p

        a = self.greedy_policy[s]
        probability_vector[a] = greedy_p

        return probability_vector

    def _calc_policy_matrix(self) -> np.ndarray:
        state_count = len(self._environment.states)
        action_count = len(self._environment.actions)
        policy_matrix = np.zeros(shape=(state_count, action_count), dtype=float)

        non_greedy_p: np.ndarray = self.epsilon * self._environment.one_over_possible_actions
        # TODO: greedy_p to zero when non_greedy_p is zero
        greedy_p: np.ndarray = (1 - self.epsilon) + non_greedy_p

        # broadcast (|S|,) to (|S|,|A|)
        non_greedy_p_broadcast = np.broadcast_to(non_greedy_p[:, np.newaxis], shape=policy_matrix.shape)
        compatible_actions: np.ndarray = self._environment.s_a_compatibility
        policy_matrix[compatible_actions] = non_greedy_p_broadcast[compatible_actions]

        i = np.arange(state_count)
        policy_vector = self.greedy_policy.policy_vector
        policy_matrix[i, policy_vector] = greedy_p

        return policy_matrix

    def _numpy_uniform_choice(self, s: int) -> int:
        if common.rng.uniform() > self.epsilon:
            return self.greedy_policy[s]
        else:
            return common.rng.choice(
                np.flatnonzero(self._environment.s_a_compatibility[s, :])
            )

    def _python_choice(self, s: int) -> int:
        return random.choices(
            population=range(len(self._environment.actions)),
            weights=list(self._policy_matrix[s, :])
        )[0]


@njit
def _numba_uniform_choice(epsilon: float,
                          a_greedy: int,
                          compatibility: np.ndarray,
                          ) -> int:
    if np.random.uniform(0.0, 1.0) > epsilon:
        return a_greedy
    else:
        return np.random.choice(
            np.flatnonzero(compatibility)
        )


@njit(cache=True)
def _numba_p_unsliced(p: np.ndarray, s: int):
    """Return an index value from a probability distribution p"""
    lo: int = 0
    hi: int = p.shape[1]
    # x: float = random.random()
    x: float = np.random.uniform(0.0, 1.0)
    cum_p: np.ndarray = np.cumsum(p[s, :])
    while lo < hi:
        mid = (lo + hi)//2
        # Use __lt__ to match the logic in list.sort() and in heapq
        if x < cum_p[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit(cache=True)
def _numba_p_choice(p: np.ndarray):
    """Return an index value from a probability distribution p"""
    lo: int = 0
    hi: int = p.shape[0]
    # x: float = random.random()
    x: float = np.random.uniform(0.0, 1.0)
    cum_p: np.ndarray = np.cumsum(p)
    while lo < hi:
        mid = (lo + hi)//2
        # Use __lt__ to match the logic in list.sort() and in heapq
        if x < cum_p[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit(cache=True)
def _numba_cum_p(p: np.ndarray):
    return np.cumsum(p)


@njit(cache=True)
def _numba_get_x() -> float:
    return random.random()


@njit(cache=True)
def _numba_cum_p_x_choice(cum_p: np.ndarray, x: float):
    """Return an index value from a probability distribution p"""
    lo: int = 0
    hi: int = cum_p.shape[0]
    # x: float = random.random()
    # x: float = np.random.uniform(0.0, 1.0)
    while lo < hi:
        mid = (lo + hi)//2
        # Use __lt__ to match the logic in list.sort() and in heapq
        if x < cum_p[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo

