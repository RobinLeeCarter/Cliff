from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
import os_environ_settings
from mdp import application, scenarios


def main():
    os_environ_settings.dummy = None    # for pycharm code inspection only

    # comparison: common.Comparison = scenarios.jacks_policy_iteration_v()
    # comparison: common.Comparison = scenarios.jacks_value_iteration_v()
    # comparison: common.Comparison = scenarios.jacks_policy_iteration_q()
    comparison: common.Comparison = scenarios.windy_timestep()
    # comparison: common.Comparison = scenarios.windy_timestep(random_wind=True)
    # comparison: common.Comparison = scenarios.cliff_alpha_start()
    # comparison: common.Comparison = scenarios.cliff_alpha_end()
    # comparison: common.Comparison = scenarios.cliff_episode()
    # comparison: common.Comparison = scenarios.random_walk_episode()
    # comparison: common.Comparison = scenarios.racetrack_episode()
    # comparison: common.Comparison = scenarios.blackjack_evaluation_v()
    # comparison: common.Comparison = scenarios.blackjack_evaluation_q()
    # comparison: common.Comparison = scenarios.blackjack_control_es()
    # comparison: common.Comparison = scenarios.gambler_value_iteration_v()

    application.Application(comparison)


if __name__ == '__main__':
    main()
