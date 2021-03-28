from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model import environment, agent
from mdp import common
from mdp.model.algorithm import abstract


class ValueIterationDpV(abstract.DynamicProgrammingV):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
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
        iteration: int = 1
        cont: bool = True
        delta: float = float('inf')

        if self._verbose:
            print(f"Starting Value Iteration ...")

        while cont and delta >= self._theta and iteration < self._iteration_timeout:
            delta = 0.0
            for state in self._environment.non_terminal_states():
                v = self.V[state]
                new_v: float = max(self._get_expected_return(state, action)
                                   for action in self._environment.actions_for_state(state))
                self.V[state] = new_v
                delta = max(delta, abs(new_v - v))
            if self._verbose:
                print(f"iteration = {iteration}\tdelta={delta:.2f}")
            if self._step_callback:
                cont = self._step_callback()
            iteration += 1

        # get greedy policy
        for state in self._environment.non_terminal_states():
            action_value: dict[environment.Action, float] = \
                {action: self._get_expected_return(state, action)
                 for action in self._environment.actions_for_state(state)}
            # argmax https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
            self._agent.policy[state] = max(action_value, key=action_value.get)
        if iteration == self._iteration_timeout:
            print(f"Warning: Timed out at {iteration} iterations")
        else:
            if self._verbose:
                print(f"Value Iteration completed ...")
