from __future__ import annotations
from dataclasses import dataclass

from mdp import common
from mdp.scenario.racetrack.comparison.comparison_builder import ComparisonBuilder
from mdp.scenario.racetrack.comparison.comparison import Comparison
from mdp.scenario.racetrack.comparison.environment_parameters import EnvironmentParameters
from mdp.scenario.racetrack.model import grids


@dataclass
class Settings(common.Settings):
    runs: int = 1
    training_episodes: int = 10_000
    episode_print_frequency: int = 1000
    # display_every_step: bool = True
    dual_policy_relationship: common.DualPolicyRelationship = common.DualPolicyRelationship.LINKED_POLICIES


class RacetrackEpisode(ComparisonBuilder):
    def create(self):
        # TODO: Problem with the first step not learning and crashing?
        #  Try grids.TRACK_1 for example (3rd position crash)
        return Comparison(
            environment_parameters=EnvironmentParameters(
                grid=grids.TRACK_3,
                extra_reward_for_failure=-100.0,  # 0.0 in problem statement
                skid_probability=0.1,
            ),
            comparison_settings=Settings(),
            breakdown_parameters=common.BreakdownParameters(
                breakdown_type=common.BreakdownType.RETURN_BY_EPISODE
            ),
            settings_list=[
                # Settings(algorithm_parameters=common.AlgorithmParameters(
                #     algorithm_type=common.AlgorithmType.EXPECTED_SARSA,
                #     alpha=0.9
                # )),
                # Settings(algorithm_parameters=common.AlgorithmParameters(
                #     algorithm_type=common.AlgorithmType.VQ,
                #     alpha=0.2
                # )),
                # Settings(algorithm_parameters=common.AlgorithmParameters(
                #     algorithm_type=common.AlgorithmType.Q_LEARNING,
                #     alpha=0.5
                # )),
                Settings(algorithm_parameters=common.AlgorithmParameters(
                    algorithm_type=common.AlgorithmType.MC_CONTROL_OFF_POLICY,
                    initial_q_value=-40.0,
                )),
            ],
            # settings_list_multiprocessing=common.ParallelContextType.SPAWN,
            graph2d_values=common.Graph2DValues(
                has_grid=True,
                has_legend=True,
                moving_average_window_size=101,
                y_min=-200,
                y_max=0
            ),
            grid_view_parameters=common.GridViewParameters(
                grid_view_type=common.GridViewType.POSITION,
                show_demo=True,
                show_trail=True
            )
        )
