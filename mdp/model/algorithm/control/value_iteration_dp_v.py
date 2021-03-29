from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.environment.environment import Environment
    from mdp.model.agent.agent import Agent
from mdp import common
from mdp.model.algorithm.abstract.dynamic_programming_v import DynamicProgrammingV


class ValueIterationDpV(DynamicProgrammingV):
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

    # def initialize(self):
    #     super().initialize()
    #     # self._environment.initialize_value_function(self.V)

    def run(self):
        # policy_: policy.Policy = self._agent.target_policy
        # assert isinstance(policy_, policy.Deterministic)
        iteration: int = 1
        cont: bool = True
        delta: float = float('inf')

        if self._verbose:
            print(f"Starting Value Iteration ...")

        while cont and delta >= self._theta and iteration < self._iteration_timeout:
            delta = 0.0
            for state in self._environment.states:
                if not state.is_terminal:
                    v = self.V[state]
                    new_v: float = max(self._get_expected_return(state, action)
                                       for action in self._environment.actions_for_state(state))
                    self.V[state] = new_v
                    delta = max(delta, abs(new_v - v))
            if self._verbose:
                print(f"iteration = {iteration}\tdelta={delta:.2f}")
            if self._step_callback:
                cont = self._step_callback()

            # if iteration == 2:
            #     break
            iteration += 1

        self._make_policy_greedy_wrt_v(round_first=True)

        if iteration == self._iteration_timeout:
            print(f"Warning: Timed out at {iteration} iterations")
        else:
            if self._verbose:
                print(f"Value Iteration completed ...")
