from __future__ import annotations
from typing import TYPE_CHECKING
import abc

if TYPE_CHECKING:
    import environment
    import agent
from algorithm import value_function


class Episodic(abc.ABC):
    name: str = "Error EpisodicAlgorithm.name"

    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 algorithm_parameters: dict[str, any]
                 ):
        self.environment: environment.Environment = environment_
        self.agent: agent.Agent = agent_
        self.algorithm_parameters: dict[str, any] = algorithm_parameters
        self.title: str = "Error: Untitled"
        if "verbose" in algorithm_parameters:
            self.verbose: bool = algorithm_parameters["verbose"]
        else:
            self.verbose: bool = False

        # assume all episodic algorithms have a Q function and initialise policy based on it
        # could add another layer of inheritance if not
        initial_v_value = algorithm_parameters['initial_v_value']
        initial_q_value = algorithm_parameters['initial_q_value']
        self._V = value_function.StateFunction(self.environment, initial_v_value)
        self._Q = value_function.StateActionFunction(self.environment, initial_q_value)
        self.gamma: float = self.algorithm_parameters['gamma']

    def _parameter_lookup(self, parameter_name: str, default) -> bool:
        if parameter_name in self.algorithm_parameters:
            return self.algorithm_parameters[parameter_name]
        else:
            return default

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
