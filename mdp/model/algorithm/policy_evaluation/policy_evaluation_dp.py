from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mdp.model import environment, agent, policy
from mdp import common
from mdp.model.algorithm import abstract, value_function


class PolicyEvaluationDP(abstract.DynamicProgramming):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 v: Optional[value_function.StateFunction] = None
                 ):
        super().__init__(environment_, agent_, algorithm_parameters)
        self._theta = self._algorithm_parameters.theta
        self._iteration_timeout = self._algorithm_parameters.iteration_timeout
        self._algorithm_type = common.AlgorithmType.POLICY_EVALUATION_DP
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} θ={self._theta}"
        if v:
            self.V = v
        else:
            self._create_v()

    def run(self):
        self.policy_evaluation()

    def policy_evaluation(self):
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

    def _get_expected_return(self, state: environment.State, action: environment.Action) -> float:
        expected_return: float = self._dynamics.get_expected_reward(state, action)
        for state_probability in self._dynamics.get_next_state_distribution(state, action):
            expected_return += state_probability.probability * self._agent.gamma * self.V[state_probability.state]
        return expected_return
