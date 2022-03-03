from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.tabular.tabular_environment import TabularEnvironment
    from mdp.model.agent.tabular.agent import Agent
from mdp import common
from mdp.model.algorithm.policy_evaluation.dp_policy_evaluation_v_deterministic import DpPolicyEvaluationVDeterministic
from mdp.model.algorithm.policy_improvement.dp_policy_improvement_v import DpPolicyImprovementV


class DpPolicyIterationV(DpPolicyEvaluationVDeterministic, DpPolicyImprovementV):
    def __init__(self,
                 environment_: TabularEnvironment,
                 agent: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent, algorithm_parameters, policy_parameters)
        self._algorithm_type = common.AlgorithmType.DP_POLICY_ITERATION_V
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} θ={self._theta}"

    # @profile
    def run(self):
        if self._verbose:
            print(f"Starting Policy Iteration ...")

        iteration: int = 1
        policy_stable: bool = False
        cont: bool = True
        if self._step_callback:
            cont = self._step_callback()
        while cont and not policy_stable and iteration < self._iteration_timeout:
            if self._verbose:
                print(f"Policy Iteration. Iteration = {iteration}")
            self._policy_evaluation()
            policy_stable = self._policy_improvement()
            if self._step_callback:
                cont = self._step_callback()
            iteration += 1

        if iteration == self._iteration_timeout:
            print(f"Warning: Timed out at {iteration} iterations")
        else:
            if self._verbose:
                print(f"Policy Iteration completed ...")

        # print(f"iterations: {iteration-1}")
