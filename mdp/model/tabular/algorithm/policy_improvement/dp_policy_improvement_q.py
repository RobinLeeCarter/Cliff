from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.tabular.agent.tabular_agent import TabularAgent
from mdp import common
from mdp.model.tabular.algorithm.abstract.dynamic_programming_q import DynamicProgrammingQ


class DpPolicyImprovementQ(DynamicProgrammingQ,
                           algorithm_type=common.AlgorithmType.DP_POLICY_IMPROVEMENT_Q,
                           algorithm_name="Policy Improvement DP (Q)"):
    def __init__(self,
                 agent: TabularAgent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(agent, algorithm_parameters)

    def run(self):
        do_call_back: bool = bool(self._step_callback)
        policy_stable: bool = False
        while not policy_stable:
            policy_stable = self._policy_improvement(do_call_back)

    def _policy_improvement(self, do_call_back: bool = False) -> bool:
        if self._verbose:
            print(f"Starting Policy Improvement ...")

        # policy_vector[s] = π(s)
        policy_vector: np.ndarray = self._target_policy.get_policy_vector()

        # π'(s) = argmax_over_a( q(s, a) )
        new_policy_vector: np.ndarray = self.Q.argmax

        policy_stable: bool = np.array_equal(new_policy_vector, policy_vector)

        self._target_policy.set_policy_vector(new_policy_vector)

        if self._verbose:
            print(f"Policy Improvement completed. policy_stable = {policy_stable}")
        if do_call_back:
            self._step_callback()

        return policy_stable
