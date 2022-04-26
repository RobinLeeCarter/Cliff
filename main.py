from __future__ import annotations

from mdp import common, Application

# comparison_type: common.ComparisonType = common.ComparisonType.BLACKJACK_CONTROL_ES
# comparison_type: common.ComparisonType = common.ComparisonType.CLIFF_EPISODE
# comparison_type: common.ComparisonType = common.ComparisonType.CLIFF_ALPHA_START
# comparison_type: common.ComparisonType = common.ComparisonType.GAMBLER_VALUE_ITERATION_V
# comparison_type: common.ComparisonType = common.ComparisonType.JACKS_POLICY_ITERATION_V
# comparison_type: common.ComparisonType = common.ComparisonType.MOUNTAIN_CAR_STANDARD
comparison_type: common.ComparisonType = common.ComparisonType.MOUNTAIN_CAR_SHARED_WEIGHTS_PARALLEL
# comparison_type: common.ComparisonType = common.ComparisonType.MOUNTAIN_CAR_DELTA_WEIGHTS_PARALLEL
# comparison_type: common.ComparisonType = common.ComparisonType.MOUNTAIN_CAR_TRAJECTORIES_SERIAL
# comparison_type: common.ComparisonType = common.ComparisonType.MOUNTAIN_CAR_TRAJECTORIES_PARALLEL
# comparison_type: common.ComparisonType = common.ComparisonType.MOUNTAIN_CAR_FEATURE_TRAJECTORIES_SERIAL
# comparison_type: common.ComparisonType = common.ComparisonType.MOUNTAIN_CAR_FEATURE_TRAJECTORIES_PARALLEL
# comparison_type: common.ComparisonType = common.ComparisonType.RACETRACK_EPISODE
# comparison_type: common.ComparisonType = common.ComparisonType.RANDOM_WALK_EPISODE
# comparison_type: common.ComparisonType = common.ComparisonType.WINDY_TIMESTEP


def main():
    Application(comparison_type=comparison_type, profile=False)


if __name__ == '__main__':
    main()
