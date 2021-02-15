from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import common
    import algorithm
    import agent
    import comparison


class Trainer:
    def __init__(self,
                 agent_: agent.Agent,
                 comparison_: comparison.Comparison,
                 verbose: bool = False
                 ):
        self.agent: agent.Agent = agent_
        self.comparison: comparison.Comparison = comparison_
        self.verbose = verbose

        self.settings: Optional[common.Settings] = None
        self.algorithm_: Optional[algorithm.Episodic] = None
        self.run_counter: int = 0
        self.episode_counter: int = 0

        self.timestep: int = 0  # cumulative across all episodes
        self.max_timestep: int = 0  # max timestep across all runs

    @property
    def episode(self) -> agent.Episode:
        return self.agent.episode

    def train(self, settings: common.Settings):
        # process settings
        self.settings = settings
        self.agent.set_settings(self.settings)
        # self.algorithm_ = self.agent.algorithm

        # self.algorithm_ = self.algorithm_factory[self.settings.algorithm_parameters]
        # self.algorithm_.agent.new_policy(settings.policy_parameters)
        # self.algorithm_.agent.set_gamma(settings.gamma)

        if settings.review_every_step:
            self.agent.set_step_callback(self.review_step)
        settings.algorithm_title = self.agent.algorithm_title
        print(f"{settings.algorithm_title}: {settings.runs} runs")

        self.max_timestep = 0
        for self.run_counter in range(1, settings.runs + 1):
            if self.verbose or self.run_counter % settings.run_print_frequency == 0:
                print(f"run_counter = {self.run_counter}: {settings.training_episodes} episodes")
            self.agent.initialize()

            self.timestep = 0
            for self.episode_counter in range(1, settings.training_episodes + 1):
                self.agent.parameter_changes(self.episode_counter)
                # print(f"episode_counter = {self.episode_counter}")
                if self.verbose or self.episode_counter % settings.episode_print_frequency == 0:
                    print(f"episode_counter = {self.episode_counter}")

                if not settings.review_every_step and self.timestep != 0:
                    self.timestep += 1  # start next episode from the next timestep
                self.agent.do_episode()

                if self.verbose:
                    episode = self.agent.episode
                    max_t = episode.max_t
                    total_return = episode.total_return
                    if self.verbose:
                        print(f"max_t = {max_t} \ttotal_return = {total_return:.2f}")
                if not settings.review_every_step:
                    self.timestep += self.agent.episode.max_t
                    self.comparison.review()
            self.max_timestep = max(self.max_timestep, self.timestep)

        if self.verbose:
            self.agent.print_statistics()

    def review_step(self):
        self.timestep += 1
        self.comparison.review()
