from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
from mdp import common
from mdp.model.algorithm import policy_evaluation, policy_improvement


class PolicyIterationDpV(policy_evaluation.PolicyEvaluationDpV, policy_improvement.PolicyImprovementDpV):
    def __init__(self,
                 environment_: Environment,
                 agent_: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._algorithm_type = common.AlgorithmType.POLICY_ITERATION_DP_V
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} θ={self._theta}"

    def run(self):
        # policy_: policy.Policy = self._agent.target_policy
        # assert isinstance(policy_, policy.Deterministic)

        if self._verbose:
            print(f"Starting Policy Iteration ...")

        iteration: int = 1
        policy_stable: bool = False
        cont: bool = True
        if self._step_callback:
            cont = self._step_callback()
        while cont and not policy_stable and iteration < self._iteration_timeout:
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

