import abc

import environment
import agent
from algorithm import episodic


class EpisodicOnline(episodic.Episodic):
    name: str = "Error EpisodicOnline.name"

    def __init__(self,
                 environment_: environment.Environment,
                 agent_: agent.Agent,
                 verbose: bool = False
                 ):
        super().__init__(environment_, agent_, verbose)

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
