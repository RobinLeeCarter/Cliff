from __future__ import annotations

import math

from mdp import common
from mdp.model.breakdown.return_by_episode import ReturnByEpisode
from mdp.model.tabular.algorithm.tabular_algorithm import TabularAlgorithm
from mdp.model.tabular.environment.tabular_environment import TabularEnvironment


class RmsByEpisode(ReturnByEpisode):
    def __init__(self, comparison: common.Comparison):
        super().__init__(comparison)
        self._y_label = "Average RMS error"

    def record(self):
        algorithm_parameters = self._trainer.settings.algorithm_parameters
        episode_counter = self._trainer.episode_counter
        self._recorder[algorithm_parameters, episode_counter] = self.rms_error()

    def rms_error(self) -> float:
        # better that it just fail if you use something with no V or an environment without get_optimum
        # if not self._algorithm.V or not hasattr(self._environment, 'get_optimum'):
        #     return None
        algorithm = self._trainer.agent.algorithm
        environment = self._trainer.agent.environment

        assert isinstance(algorithm, TabularAlgorithm)
        assert isinstance(environment, TabularEnvironment)

        squared_error: float = 0.0
        count: int = 0
        for s, state in enumerate(environment.states):
            if environment.is_valued_state(state):
                value: float = algorithm.V[s]
                # noinspection PyUnresolvedReferences
                optimum: float = environment.get_optimum(state)
                squared_error += (value - optimum)**2
                count += 1
        rms_error = math.sqrt(squared_error / count)
        return rms_error

