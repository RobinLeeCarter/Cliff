from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import abc

if TYPE_CHECKING:
    import environment
    import agent
    import common
from algorithm import value_function


class Episodic(abc.ABC):
    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: common.AlgorithmParameters
                 ):
        self.environment: environment.Environment = environment_
        self.agent: agent.Agent = agent_
        self.algorithm_parameters: common.AlgorithmParameters = algorithm_parameters
        self.algorithm_type: Optional[common.AlgorithmType] = None
        self.name: str = "Error: Untitled"
        self.title: str = "Error: Untitled"
        self.verbose = self.algorithm_parameters.verbose

        # assume all episodic algorithms have a Q function and initialise policy based on it
        # could add another layer of inheritance if not
        self._V = value_function.StateFunction(self.environment, self.algorithm_parameters.initial_v_value)
        self._Q = value_function.StateActionFunction(self.environment, self.algorithm_parameters.initial_q_value)
        self.gamma: float = self.agent.gamma

    def initialize(self):
        self._V.initialize_values()
        self._Q.initialize_values()
        self._make_policy_greedy_wrt_q()

    def parameter_changes(self, iteration: int):
        pass

    def _make_policy_greedy_wrt_q(self):
        for state_ in self.environment.states():
            self.agent.policy[state_] = self._Q.argmax_over_actions(state_)

    @abc.abstractmethod
    def do_episode(self, episode_length_timeout: int):
        pass

    def print_q_coverage_statistics(self):
        self._Q.print_coverage_statistics()

    def __repr__(self):
        return f"{self.title}"
