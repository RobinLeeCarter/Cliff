from __future__ import annotations
from typing import Optional

import utils
import common
import policy
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

        # create environment
        self.environment = self._create_environment()

        # create policy and agent
        # TODO: have e-greedy_policy make it's own DetermintisticPolicy
        self.greedy_policy: policy.DeterministicPolicy = policy.DeterministicPolicy(self.environment)
        self.e_greedy_policy: policy.EGreedyPolicy = policy.EGreedyPolicy(self.environment,
                                                                          greedy_policy=self.greedy_policy)
        self.agent = agent.Agent(self.environment, self.e_greedy_policy)

        self.graph = view.Graph()
        self.grid_view = view.GridView(self.environment.grid_world)

        self.algorithm_factory: algorithm.Factory = algorithm.Factory(self.environment, self.agent)
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
        environment_type = self.scenario.environment_type

        ep: common.EnvironmentParameters = self.scenario.environment_parameters
        e = common.EnvironmentType
        if environment_type == e.CLIFF:
            environment_ = environments.Cliff(verbose=ep.verbose)
        elif environment_type == e.WINDY:
            environment_ = environments.Windy(random_wind=ep.random_wind, verbose=ep.verbose)
        elif environment_type == e.RANDOM_WALK:
            environment_ = environments.RandomWalk(verbose=ep.verbose)
        else:
            raise NotImplementedError
        return environment_

    def _create_comparison(self) -> comparison.Comparison:
        c = common.ComparisonType
        comparison_type = self.scenario.comparison_type
        if comparison_type == c.EPISODE_BY_TIMESTEP:
            comparison_ = comparison.EpisodeByTimestep(self.scenario, self.graph)
        elif comparison_type == c.RETURN_BY_EPISODE:
            comparison_ = comparison.ReturnByEpisode(self.scenario, self.graph)
        elif comparison_type == c.RETURN_BY_ALPHA:
            self.scenario: common.AlgorithmByAlpha
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

        # e = common.EnvironmentType
        # c = common.ComparisonType
        # comparison_lookup: dict[tuple[e, c], Type[comparison.Comparison]] = {
        #     (e.Windy,       c.EPISODE_BY_TIMESTEP):  comparison.EpisodeByTimestep,
        #     (e.Windy,       c.RETURN_BY_EPISODE):    comparison.ReturnByEpisode,
        #     (e.Windy,       c.RETURN_BY_ALPHA):      comparison.ReturnByAlpha,
        #     (e.Cliff,       c.EPISODE_BY_TIMESTEP):  comparison.EpisodeByTimestep,
        #     (e.Cliff,       c.RETURN_BY_EPISODE):    environments.cliff.ReturnByEpisode,
        #     (e.Cliff,       c.RETURN_BY_ALPHA):      comparison.ReturnByAlpha,
        #     (e.RandomWalk,  c.EPISODE_BY_TIMESTEP):  comparison.EpisodeByTimestep,
        #     (e.RandomWalk,  c.RETURN_BY_EPISODE):    comparison.ReturnByEpisode,
        #     (e.RandomWalk,  c.RETURN_BY_ALPHA):      comparison.ReturnByAlpha,
        # }

        # c = common.ComparisonType
        # comparison_lookup: dict[c, Type[comparison.Comparison]] = {
        #     c.EPISODE_BY_TIMESTEP:  comparison.EpisodeByTimestep,
        #     c.RETURN_BY_EPISODE:    comparison.ReturnByEpisode,
        #     c.RETURN_BY_ALPHA:      comparison.ReturnByAlpha,
        # }
        # type_for_comparison = comparison_lookup[self.scenario.comparison_type]
        # comparison_ = type_for_comparison(self.graph, verbose=False)

    # def setup_and_run(self, comparison_type: common.ComparisonType):
    #     e = common.EnvironmentType
    #     c = common.ComparisonType
    #     comparison_lookup: dict[tuple[e, c], Type[comparison.Comparison]] = {
    #         (e.Windy,       c.EPISODE_BY_TIMESTEP):  comparison.EpisodeByTimestep,
    #         (e.Windy,       c.RETURN_BY_EPISODE):    comparison.ReturnByEpisode,
    #         (e.Windy,       c.RETURN_BY_ALPHA):      comparison.ReturnByAlpha,
    #         (e.Cliff,       c.EPISODE_BY_TIMESTEP):  comparison.EpisodeByTimestep,
    #         (e.Cliff,       c.RETURN_BY_EPISODE):    environments.cliff.ReturnByEpisode,
    #         (e.Cliff,       c.RETURN_BY_ALPHA):      comparison.ReturnByAlpha,
    #         (e.RandomWalk,  c.EPISODE_BY_TIMESTEP):  comparison.EpisodeByTimestep,
    #         (e.RandomWalk,  c.RETURN_BY_EPISODE):    comparison.ReturnByEpisode,
    #         (e.RandomWalk,  c.RETURN_BY_ALPHA):      comparison.ReturnByAlpha,
    #     }
    #
    #     type_for_comparison = comparison_lookup[self.environment_type, comparison_type]
    #     self.comparison = type_for_comparison(self.graph, verbose=False)
    #
    #     self.trainer = trainer.Trainer(
    #         algorithm_factory=self.algorithm_factory,
    #         comparison_=self.comparison,
    #         verbose=False
    #     )
    #     self.comparison.set_trainer(self.trainer)
