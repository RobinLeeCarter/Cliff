from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
from mdp import common
from mdp.model.algorithm.abstract.dynamic_programming_v import DynamicProgrammingV


class PolicyEvaluationDpV(DynamicProgrammingV):
    def __init__(self,
                 environment_: Environment,
                 agent_: Agent,
                 algorithm_parameters: common.AlgorithmParameters,
                 policy_parameters: common.PolicyParameters
                 ):
        super().__init__(environment_, agent_, algorithm_parameters, policy_parameters)
        self._algorithm_type = common.AlgorithmType.POLICY_EVALUATION_DP_V
        self.name = common.algorithm_name[self._algorithm_type]
        self.title = f"{self.name} θ={self._theta}"

    def run(self):
        do_call_back: bool = bool(self._step_callback)
        self._policy_evaluation(do_call_back)
        # if self._verbose:
        #     self.V.print_all_values()

    def _policy_evaluation(self, do_call_back: bool = False):
        iteration: int = 1
        cont: bool = True
        delta: float = float('inf')

        if self._verbose:
            print(f"Starting Policy Evaluation ...")

        while cont and delta >= self._theta and iteration < self._iteration_timeout:
            delta = 0.0
            for state in self._environment.states:
                v = self.V[state]
                new_v: float = 0.0
                for action in self._environment.actions_for_state(state):
                    # π(a|s)
                    policy_probability = self._agent.policy.get_probability(state, action)
                    if policy_probability > 0:
                        new_v += policy_probability * self._get_expected_return(state, action)
                self.V[state] = new_v
                delta = max(delta, abs(new_v - v))
            if self._verbose:
                print(f"iteration = {iteration}\tdelta={delta:.2f}")
            if do_call_back:
                cont = self._step_callback()
            iteration += 1

        if iteration == self._iteration_timeout:
            print(f"Warning: Timed out at {iteration} iterations")
        else:
            if self._verbose:
                print(f"Policy Evaluation completed. delta={delta:.2f}")
