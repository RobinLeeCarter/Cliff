from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable

if TYPE_CHECKING:
    from mdp import common
    from mdp.model import breakdown, agent


class Trainer:
    def __init__(self,
                 agent_: agent.Agent,
                 breakdown_: breakdown.Breakdown,
                 model_step_callback: Optional[Callable[[agent.Episode], None]] = None,
                 verbose: bool = False
                 ):
        self._agent: agent.Agent = agent_
        self._breakdown: breakdown.Breakdown = breakdown_
        self._model_step_callback: Optional[Callable[[agent.Episode], None]] = model_step_callback
        self._verbose = verbose
        self._cont: bool = True

        self.settings: Optional[common.Settings] = None
        self.run_counter: int = 0
        self.episode_counter: int = 0

        self.timestep: int = 0  # cumulative across all episodes
        self.max_timestep: int = 0  # max timestep across all runs

    @property
    def episode(self) -> agent.Episode:
        return self._agent.episode

    @property
    def agent(self) -> agent.Agent:
        return self._agent

    def train(self, settings: common.Settings):
        # process settings
        self.settings = settings
        self._agent.apply_settings(self.settings)
        # self.algorithm_ = self.agent.algorithm

        # self.algorithm_ = self.algorithm_factory[self.settings.algorithm_parameters]
        # self.algorithm_.agent.new_policy(settings.policy_parameters)
        # self.algorithm_.agent.set_gamma(settings.gamma)

        if settings.review_every_step or settings.display_every_step:
            self._agent.set_step_callback(self.step)
        settings.algorithm_title = self._agent.algorithm_title
        print(f"{settings.algorithm_title}: {settings.runs} runs")

        self.max_timestep = 0
        for self.run_counter in range(1, settings.runs + 1):
            if self._verbose or self.run_counter % settings.run_print_frequency == 0:
                print(f"run_counter = {self.run_counter}: {settings.training_episodes} episodes")
            self._agent.initialize()

            self.timestep = 0
            for self.episode_counter in range(1, settings.training_episodes + 1):
                self._agent.parameter_changes(self.episode_counter)
                # print(f"episode_counter = {self.episode_counter}")
                if self._verbose or self.episode_counter % settings.episode_print_frequency == 0:
                    print(f"episode_counter = {self.episode_counter}")

                if not settings.review_every_step and self.timestep != 0:
                    self.timestep += 1  # start next episode from the next timestep
                self._agent.do_episode()

                if self._verbose:
                    episode = self._agent.episode
                    max_t = episode.max_t
                    total_return = episode.total_return
                    if self._verbose:
                        print(f"max_t = {max_t} \ttotal_return = {total_return:.2f}")
                if not settings.review_every_step:
                    self.timestep += self._agent.episode.max_t
                    self._breakdown.review()
            self.max_timestep = max(self.max_timestep, self.timestep)

        if self._verbose:
            self._agent.print_statistics()

    def step(self) -> bool:
        if self.settings.review_every_step:
            self.review_step()
        if self.settings.display_every_step and self._model_step_callback:
            self._model_step_callback(self._agent.episode)
        return True

    def review_step(self):
        self.timestep += 1
        self._breakdown.review()
