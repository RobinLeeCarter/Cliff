from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model import environment, agent, policy
from mdp import common
from mdp.model.algorithm import abstract


class PolicyEvaluationDpV(abstract.DynamicProgrammingV):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters)
        self._algorithm_type = common.AlgorithmType.POLICY_EVALUATION_DP_V
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} θ={self._theta}"

    def run(self):
        self._policy_evaluation()
        if self._verbose:
            print(self.V)

    def _policy_evaluation(self):
        iteration: int = 0
        delta: float = 0.0
        policy_: policy.Policy = self._agent.policy

        while delta < self._theta and iteration < self._iteration_timeout:
            delta = 0.0
            for state in self._environment.states:
                v = self.V[state]
                new_v: float = 0.0
                for action in self._environment.actions_for_state(state):
                    # π(a|s)
                    policy_probability = policy_.get_probability(state, action)
                    if policy_probability > 0:
                        new_v += self._get_expected_return(state, action)
                self.V[state] = new_v
                delta = max(delta, abs(new_v - v))
            iteration += 1

        if iteration == self._iteration_timeout:
            print(f"Warning: Timed out at {iteration} iterations")
