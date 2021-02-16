from __future__ import annotations
from typing import Optional

import utils
import common
import agent
import view
import breakdown
import environments
import trainer
import comparisons


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose

        # self.comparison: common.Comparison = comparison.windy_timestep
        # self.comparison: common.Comparison = comparison.random_windy_timestep
        # self.comparison: common.Comparison = comparison.cliff_alpha
        self.comparison: common.Comparison = comparisons.cliff_episode

        self.settings: Optional[common.Settings] = None  # current settings

        self.environment = environments.factory(self.comparison.environment_parameters)

        # create agent (and it will create the algorithm and the policy when it is given Settings)
        self.agent = agent.Agent(self.environment)

        self.graph = view.Graph()
        self.grid_view = view.GridView(self.environment.grid_world)

        self.breakdown: breakdown.Breakdown = breakdown.factory(self.comparison, self.graph)
        self.trainer: trainer.Trainer = trainer.Trainer(
            agent_=self.agent,
            breakdown_=self.breakdown,
            verbose=False
        )
        self.breakdown.set_trainer(self.trainer)

        # self.target_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.environment)
        # self.behaviour_policy: policy.EGreedyPolicy = policy.EGreedyPolicy(self.environment,
        #                                                                    greedy_policy=self.target_policy)
        # self.behaviour_policy: policy.RandomPolicy = policy.RandomPolicy(self.environment)
        # self.target_agent = agent.Agent(self.environment, self.target_policy)
        # self.behaviour_agent = agent.Agent(self.environment, self.behaviour_policy)

    def run(self):
        timer: utils.Timer = utils.Timer()
        timer.start()
        for self.settings in self.comparison.settings_list:
            self.trainer.train(self.settings)
            timer.lap(name=str(self.settings.algorithm_title))
        timer.stop()

        self.breakdown.compile()
        self.breakdown.draw_graph()

    def demonstrate(self):
        self.grid_view.open_window()
        running_average = 0
        count = 0
        while True:
            self.agent.generate_episode(self.settings.episode_length_timeout)
            episode = self.agent.episode
            print(f"max_t: {episode.max_t} \t total_return: {episode.total_return:.0f}")
            count += 1
            running_average += (1/count) * (episode.total_return - running_average)
            print(f"count: {count} \t running_average: {running_average:.1f}")
            user_event: common.UserEvent = self.grid_view.display_episode(episode, show_trail=False)
            if user_event == common.UserEvent.QUIT:
                break
        self.grid_view.close_window()
