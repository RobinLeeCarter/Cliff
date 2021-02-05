import abc

import constants
import environment
import agent
from algorithm.algorithms import support


class EpisodicAlgorithm(abc.ABC):
    name: str = "Error EpisodicAlgorithm.name"

    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 title: str,
                 verbose: bool = False
                 ):
        self.environment: environment.Environment = environment_
        self.agent: agent.Agent = agent_
        self.title: title = title
        self.verbose = verbose

        # assume all episodic algorithms have a Q function and initialise policy based on it
        # could add another layer of inheritance if not
        self._V = support.StateFunction(self.environment)
        self._Q = support.StateActionFunction(self.environment)

    def initialize(self):
        self._V.initialize_values()
        self._Q.initialize_values()
        self._make_policy_greedy_wrt_q()

    def parameter_changes(self, iteration: int):
        pass

    def _make_policy_greedy_wrt_q(self):
        for state_ in self.environment.states():
            self.agent.policy[state_] = self._Q.argmax_over_actions(state_)

    def do_episode(self, episode_length_timeout: int):
        self.agent.start_episode()
        self._start_episode()
        while (not self.agent.state.is_terminal) and self.agent.t < episode_length_timeout:
            self._do_training_step()

    def _start_episode(self):
        pass

    @abc.abstractmethod
    def _do_training_step(self):
        pass

    def print_q_coverage_statistics(self):
        self._Q.print_coverage_statistics()

    def __repr__(self):
        return f"{self.title}"
