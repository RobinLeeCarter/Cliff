from __future__ import annotations
from typing import Optional

import utils
import common
import agent
import algorithm
import view
import comparison
import environment
import environments
import trainer
import scenarios


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose

        # self.scenario: common.Scenario = scenarios.windy_timestep
        # self.scenario: common.Scenario = scenarios.random_windy_timestep
        # self.scenario: common.Scenario = scenarios.cliff_alpha
        self.scenario: common.Scenario = scenarios.cliff_episode

        self.settings: Optional[common.Settings] = None  # current settings

        self.environment = self._create_environment()

        # create policy and agent
        # self.policy: policy.EGreedy = policy.EGreedy(
        #     environment_=self.environment,
        #     epsilon=self.scenario.scenario_settings.policy_parameters.epsilon
        # )

        # create agent (and it will create the policy)
        self.agent = agent.Agent(self.environment, self.scenario.scenario_settings.policy_parameters)

        self.algorithm_factory: algorithm.Factory = algorithm.Factory(self.environment, self.agent)

        self.graph = view.Graph()
        self.grid_view = view.GridView(self.environment.grid_world)

        self.comparison: comparison.Comparison = self._create_comparison()

        self.trainer: trainer.Trainer = trainer.Trainer(
            algorithm_factory=self.algorithm_factory,
            comparison_=self.comparison,
            verbose=False
        )
        self.comparison.set_trainer(self.trainer)

        # self.target_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.environment)
        # self.behaviour_policy: policy.EGreedyPolicy = policy.EGreedyPolicy(self.environment,
        #                                                                    greedy_policy=self.target_policy)
        # self.behaviour_policy: policy.RandomPolicy = policy.RandomPolicy(self.environment)
        # self.target_agent = agent.Agent(self.environment, self.target_policy)
        # self.behaviour_agent = agent.Agent(self.environment, self.behaviour_policy)

    def _create_environment(self) -> environment.Environment:
        environment_parameters = self.scenario.environment_parameters
        environment_type = environment_parameters.environment_type
        et = common.EnvironmentType

        if environment_type == et.CLIFF:
            environment_ = environments.Cliff(environment_parameters)
        elif environment_type == et.WINDY:
            environment_ = environments.Windy(environment_parameters)
        elif environment_type == et.RANDOM_WALK:
            environment_ = environments.RandomWalk(environment_parameters)
        else:
            raise NotImplementedError
        return environment_

    def _create_comparison(self) -> comparison.Comparison:
        c = common.ComparisonType
        comparison_type = self.scenario.comparison_parameters.comparison_type
        if comparison_type == c.EPISODE_BY_TIMESTEP:
            comparison_ = comparison.EpisodeByTimestep(self.scenario, self.graph)
        elif comparison_type == c.RETURN_BY_EPISODE:
            comparison_ = comparison.ReturnByEpisode(self.scenario, self.graph)
        elif comparison_type == c.RETURN_BY_ALPHA:
            comparison_ = comparison.ReturnByAlpha(self.scenario, self.graph)
        else:
            raise NotImplementedError

        return comparison_

    def run(self):
        timer: utils.Timer = utils.Timer()
        timer.start()
        for self.settings in self.scenario.settings_list:
            self.trainer.train(self.settings)
            timer.lap(name=str(self.settings.algorithm_title))
        timer.stop()

        self.comparison.compile()
        self.comparison.draw_graph()

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
