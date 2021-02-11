from __future__ import annotations
from typing import Optional

import utils
import common
import policy
import agent
import algorithm
import view
import comparison
import environments


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose

        self.environment = environments.Windy()
        # self.environment.verbose = True
        self.greedy_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.environment)
        self.e_greedy_policy: policy.EGreedyPolicy = policy.EGreedyPolicy(self.environment,
                                                                          greedy_policy=self.greedy_policy)
        self.agent = agent.Agent(self.environment, self.e_greedy_policy)

        self.algorithm_factory: algorithm.Factory = algorithm.Factory(self.environment, self.agent)

        self.comparison: Optional[comparison.Comparison] = None

        self.graph = view.Graph()
        self.grid_view = view.GridView(self.environment.grid_world)

        # self.target_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.environment)
        # self.behaviour_policy: policy.EGreedyPolicy = policy.EGreedyPolicy(self.environment,
        #                                                                    greedy_policy=self.target_policy)
        # self.behaviour_policy: policy.RandomPolicy = policy.RandomPolicy(self.environment)
        # self.target_agent = agent.Agent(self.environment, self.target_policy)
        # self.behaviour_agent = agent.Agent(self.environment, self.behaviour_policy)

    def setup_and_run(self, comparison_type: common.ComparisonType):
        if comparison_type == common.ComparisonType.EPISODE_BY_TIMESTEP:
            self.comparison = comparison.EpisodeByTimestep(self.algorithm_factory, self.graph, verbose=False)
        elif comparison_type == common.ComparisonType.RETURN_BY_EPISODE:
            if isinstance(self.environment, environments.Windy):
                self.comparison = environments.cliff.ReturnByEpisode(
                    self.algorithm_factory, self.graph, verbose=False)
            else:
                self.comparison = comparison.ReturnByEpisode(self.algorithm_factory, self.graph, verbose=False)
        elif comparison_type == common.ComparisonType.RETURN_BY_ALPHA:
            self.comparison = comparison.ReturnByAlpha(self.algorithm_factory, self.graph, verbose=False)
        else:
            raise NotImplementedError
        self.comparison.build()

    def run(self):
        timer: utils.Timer = utils.Timer()
        timer.start()
        for settings in self.comparison.settings_list:
            self.comparison.train(settings)
            timer.lap(name=str(settings.algorithm_title))
        timer.stop()

        self.comparison.compile()
        self.comparison.draw_graph()

    def demonstrate(self):
        self.grid_view.open_window()
        running_average = 0
        count = 0
        while True:
            self.agent.generate_episode()
            episode = self.agent.episode
            print(f"max_t: {episode.max_t} \t total_return: {episode.total_return:.0f}")
            count += 1
            running_average += (1/count) * (episode.total_return - running_average)
            print(f"count: {count} \t running_average: {running_average:.1f}")
            user_event: common.UserEvent = self.grid_view.display_episode(episode, show_trail=False)
            if user_event == common.UserEvent.QUIT:
                break
        self.grid_view.close_window()

        # self.behaviour_agent.set_policy(self.target_policy)
        # self.grid_view.open_window()
        # self.view.display_and_wait()
        # self.environment.verbose = True
        # self.agent.verbose = True

        # self.agent.verbose = True
        # self.agent.set_policy(self.greedy_policy)

        # self.grid_view.open_window()
        #
        # recorder_ = recorder.Recorder[str]()
        #
        # test: str = "test"
        #
        # running_average = 0
        # count = 0
        # while count < 100:
        #     self.agent.generate_episode()
        #     episode = self.agent.episode
        #     print(f"max_t: {episode.max_t} \t total_return: {episode.total_return:.0f}")
        #     count += 1
        #     running_average += (1/count) * (episode.total_return - running_average)
        #     print(f"count: {count} \t running_average: {running_average:.1f}")
        #     recorder_[test] = episode.total_return
        #     value = recorder_[test]
        #     print(f"recorder value: {value:.1f}")
        # user_event: common.UserEvent = self.grid_view.display_episode(episode, show_trail=False)
        # if user_event == common.UserEvent.QUIT:
        #     break
