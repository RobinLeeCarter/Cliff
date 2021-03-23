from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
import os_environ_settings
from mdp import application, scenarios


def main():
    os_environ_settings.dummy = None    # for pycharm code inspection only

    # comparison: common.Comparison = scenarios.windy_timestep()
    # comparison: common.Comparison = scenarios.windy_timestep(random_wind=True)
    # comparison: common.Comparison = scenarios.cliff_alpha()
    comparison: common.Comparison = scenarios.cliff_episode()
    # comparison: common.Comparison = scenarios.random_walk_episode()
    # comparison: common.Comparison = scenarios.racetrack_episode()
    # comparison: common.Comparison = scenarios.jacks_comparison()

    application.Application(comparison)


if __name__ == '__main__':
    main()
