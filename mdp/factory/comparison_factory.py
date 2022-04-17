from __future__ import annotations
from typing import Type

from mdp.task.base_comparison_builder import BaseComparisonBuilder
from mdp import common

from mdp.task.blackjack.comparison.blackjack_control_es import BlackjackControlES
from mdp.task.blackjack.comparison.blackjack_evaluation_q import BlackjackEvaluationQ
from mdp.task.blackjack.comparison.blackjack_evaluation_v import BlackjackEvaluationV

from mdp.task.cliff.comparison.cliff_alpha_start import CliffAlphaStart
from mdp.task.cliff.comparison.cliff_alpha_end import CliffAlphaEnd
from mdp.task.cliff.comparison.cliff_episode import CliffEpisode

from mdp.task.gambler.comparison.gambler_value_iteration_v import GamblerValueIterationV

from mdp.task.jacks.comparison.jacks_policy_evaluation_q import JacksPolicyEvaluationQ
from mdp.task.jacks.comparison.jacks_policy_evaluation_v import JacksPolicyEvaluationV
from mdp.task.jacks.comparison.jacks_policy_improvement_q import JacksPolicyImprovementQ
from mdp.task.jacks.comparison.jacks_policy_improvement_v import JacksPolicyImprovementV
from mdp.task.jacks.comparison.jacks_policy_iteration_q import JacksPolicyIterationQ
from mdp.task.jacks.comparison.jacks_policy_iteration_v import JacksPolicyIterationV
from mdp.task.jacks.comparison.jacks_value_iteration_q import JacksValueIterationQ
from mdp.task.jacks.comparison.jacks_value_iteration_v import JacksValueIterationV

from mdp.task.mountain_car.comparison.mountain_car_standard import MountainCarStandard
from mdp.task.mountain_car.comparison.mountain_car_batch_serial import MountainCarBatchSerial
from mdp.task.mountain_car.comparison.mountain_car_parallel_weights import MountainCarParallelWeights
from mdp.task.mountain_car.comparison.mountain_car_parallel_trajectories import MountainCarParallelTrajectories

from mdp.task.racetrack.comparison.racetrack_episode import RacetrackEpisode

from mdp.task.random_walk.comparison.random_walk_episode import RandomWalkEpisode

from mdp.task.windy.comparison.windy_timestep import WindyTimestep
from mdp.task.windy.comparison.windy_timestep_random import WindyTimestepRandom

ComparisonBuilderLookup = dict[common.ComparisonType, Type[BaseComparisonBuilder]]


class ComparisonFactory:
    def create(self, comparison_type: common.ComparisonType) -> common.Comparison:
        type_of_comparison_builder: Type[BaseComparisonBuilder] = \
            BaseComparisonBuilder.type_registry[comparison_type]
        comparison_builder: BaseComparisonBuilder = type_of_comparison_builder()
        comparison: common.Comparison = comparison_builder.create()
        return comparison


def __dummy():
    """Stops Pycharm objecting to imports. The imports are needed to generate the registry."""
    return [
        BlackjackControlES,
        BlackjackEvaluationQ,
        BlackjackEvaluationV,

        CliffAlphaEnd,
        CliffAlphaStart,
        CliffEpisode,

        GamblerValueIterationV,

        JacksPolicyEvaluationQ,
        JacksPolicyEvaluationV,
        JacksPolicyImprovementQ,
        JacksPolicyImprovementV,
        JacksPolicyIterationQ,
        JacksPolicyIterationV,
        JacksValueIterationQ,
        JacksValueIterationV,

        MountainCarStandard,
        MountainCarBatchSerial,
        MountainCarParallelWeights,
        MountainCarParallelTrajectories,

        RacetrackEpisode,

        RandomWalkEpisode,

        WindyTimestep,
        WindyTimestepRandom
    ]
