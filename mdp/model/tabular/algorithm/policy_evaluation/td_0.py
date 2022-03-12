from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.model.tabular.agent.tabular_agent import TabularAgent
    from mdp.model.tabular.agent.tabular_episode import TabularEpisode
from mdp import common
from mdp.model.tabular.algorithm.abstract.episodic_online import EpisodicOnline


class TD0(EpisodicOnline):
    def __init__(self,
                 agent: TabularAgent,
                 algorithm_parameters: common.AlgorithmParameters,
                 name: str
                 ):
        super().__init__(agent, algorithm_parameters, name)
        self._alpha = self._algorithm_parameters.alpha
        self.title = f"{self.name} α={self._alpha}"
        self._create_v()

    def _do_training_step(self):
        ag = self._agent
        ag.choose_action()
        ag.take_action()

        target = ag.r + self._gamma * self.V[ag.s]
        delta = target - self.V[ag.prev_s]
        self.V[ag.prev_s] += self._alpha * delta

    def _do_step_of_episode(self, episode: TabularEpisode, t: int):
        # TODO: episode is set once rather than passed in
        s: int = episode.trajectory[t].s
        s_dash: int = episode.trajectory[t+1].s
        r: float = episode.trajectory[t+1].r
        target = r + self._gamma * self.V[s_dash]
        delta = target - self.V[s]
        self.V[s] += self._alpha * delta
