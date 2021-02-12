from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import common
    import algorithm
    import agent
    import comparison


class Trainer:
    def __init__(self,
                 algorithm_factory: algorithm.Factory,
                 comparison_: comparison.Comparison,
                 verbose: bool = False
                 ):
        self.algorithm_factory: algorithm.Factory = algorithm_factory
        self.comparison: comparison.Comparison = comparison_
        self.verbose = verbose

        self.settings: Optional[common.Settings] = None
        self.algorithm_: Optional[algorithm.Episodic] = None
        self.run_counter: int = 0
        self.episode_counter: int = 0
        self.timestep: int = 0  # cumulative across all episodes
        self.max_timestep: int = 0  # max timestep across all runs
        self.episode: Optional[agent.Episode] = None
        self.max_t: int = 0
        self.total_return: float = 0.0
        self.review_every_step: bool = False

    def train(self, settings: common.Settings):
        self.settings = settings
        self.algorithm_ = self.algorithm_factory[self.settings]
        if self.review_every_step:
            self.algorithm_.agent.set_step_callback(self.review_step)
        settings.algorithm_title = self.algorithm_.title
        print(f"{self.algorithm_.title}: {settings.runs} runs")

        self.max_timestep = 0
        for self.run_counter in range(1, settings.runs + 1):
            if self.verbose or self.run_counter % settings.run_print_frequency == 0:
                print(f"run_counter = {self.run_counter}: {settings.training_episodes} episodes")
            self.algorithm_.initialize()

            self.timestep = 0
            for self.episode_counter in range(1, settings.training_episodes + 1):
                self.algorithm_.parameter_changes(self.episode_counter)
                # print(f"episode_counter = {self.episode_counter}")
                if self.verbose or self.episode_counter % settings.episode_print_frequency == 0:
                    print(f"episode_counter = {self.episode_counter}")

                if not self.review_every_step and self.timestep != 0:
                    self.timestep += 1  # start next episode from the next timestep
                self.algorithm_.do_episode(settings.episode_length_timeout)
                self.episode = self.algorithm_.agent.episode
                self.max_t = self.episode.max_t
                self.total_return = self.episode.total_return
                # if self.verbose:
                #     print(f"max_t = {max_t} \ttotal_return = {total_return:.2f}")
                if not self.review_every_step:
                    self.timestep += self.max_t
                    self.comparison.review()
            self.max_timestep = max(self.max_timestep, self.timestep)

        self.algorithm_.print_q_coverage_statistics()

    def review_step(self, episode: agent.Episode):
        self.episode = episode
        self.timestep += 1
        self.comparison.review()
