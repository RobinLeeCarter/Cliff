from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mdp.model.environment.tabular.tabular_environment import TabularEnvironment
    from mdp.model.agent.agent import Agent
from mdp import common
from mdp.model.algorithm.abstract.dynamic_programming_q import DynamicProgrammingQ


class DpPolicyImprovementQ(DynamicProgrammingQ):
    def __init__(self,
                 environment_: TabularEnvironment,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent, algorithm_parameters, policy_parameters)
        self._algorithm_type = common.AlgorithmType.DP_POLICY_IMPROVEMENT_Q
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name}"

    def run(self):
        do_call_back: bool = bool(self._step_callback)
        policy_stable: bool = False
        while not policy_stable:
            policy_stable = self._policy_improvement(do_call_back)

    def _policy_improvement(self, do_call_back: bool = False) -> bool:
        if self._verbose:
            print(f"Starting Policy Improvement ...")

        # policy_vector[s] = π(s)
        policy_vector: np.ndarray = self._agent.policy.get_policy_vector()

        # π'(s) = argmax_over_a( q(s, a) )
        new_policy_vector: np.ndarray = self.Q.argmax

        policy_stable: bool = np.array_equal(new_policy_vector, policy_vector)

        self._agent.policy.set_policy_vector(new_policy_vector)

        if self._verbose:
            print(f"Policy Improvement completed. policy_stable = {policy_stable}")
        if do_call_back:
            self._step_callback()

        return policy_stable
