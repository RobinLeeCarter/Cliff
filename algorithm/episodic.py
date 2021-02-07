import abc

import environment
import agent
from algorithm import common


class Episodic(abc.ABC):
    name: str = "Error EpisodicAlgorithm.name"

    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 verbose: bool = False
                 ):
        self.environment: environment.Environment = environment_
        self.agent: agent.Agent = agent_
        self.title: str = "Error: Untitled"
        self.verbose = verbose

        # assume all episodic algorithms have a Q function and initialise policy based on it
        # could add another layer of inheritance if not
        self._V = common.StateFunction(self.environment)
        self._Q = common.StateActionFunction(self.environment)

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
