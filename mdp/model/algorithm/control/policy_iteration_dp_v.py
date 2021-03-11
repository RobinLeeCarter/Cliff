from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model import environment, agent, policy
from mdp import common
from mdp.model.algorithm import policy_evaluation, policy_improvement


class PolicyIterationDpV(policy_evaluation.PolicyEvaluationDpV, policy_improvement.PolicyImprovementDpV):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters)
        self._algorithm_type = common.AlgorithmType.POLICY_ITERATION_DP_V
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} θ={self._theta}"

    def run(self):
        policy_: policy.Policy = self._agent.target_policy
        assert isinstance(policy_, policy.Deterministic)

        iteration: int = 0
        policy_stable: bool = False
        while not policy_stable and iteration < self._iteration_timeout:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            iteration += 1

        if iteration == self._iteration_timeout:
            print(f"Warning: Timed out at {iteration} iterations")
