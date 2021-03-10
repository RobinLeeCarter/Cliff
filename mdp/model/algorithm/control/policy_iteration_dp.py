from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model import environment, agent, policy
from mdp import common
from mdp.model.algorithm import abstract, policy_evaluation, policy_improvement


class PolicyIterationDP(abstract.DynamicProgramming):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters)
        self._theta = self._algorithm_parameters.theta
        self._iteration_timeout = self._algorithm_parameters.iteration_timeout
        self._algorithm_type = common.AlgorithmType.POLICY_ITERATION_DP
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} θ={self._theta}"
        self._create_v()
        self._evaluation_algorithm = policy_evaluation.PolicyEvaluationDP(
            environment_, agent_, algorithm_parameters, self.V)
        self._improvement_algorithm = policy_improvement.PolicyImprovementDP(
            environment_, agent_, algorithm_parameters, self.V)

    def run(self):
        policy_: policy.Policy = self._agent.target_policy
        assert isinstance(policy_, policy.Deterministic)

        iteration: int = 0
        policy_stable: bool = False
        while not policy_stable and iteration < self._iteration_timeout:
            self._evaluation_algorithm.policy_evaluation()
            policy_stable = self._improvement_algorithm.policy_improvement()
            iteration += 1

        if iteration == self._iteration_timeout:
            print(f"Warning: Timed out at {iteration} iterations")
