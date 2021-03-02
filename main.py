from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp import common
import os_environ_settings
from mdp import application
from mdp.model import scenarios


def main():
    os_environ_settings.dummy = None    # for pycharm code inspection only

    comparison: common.Comparison = scenarios.windy.comparisons.timestep()
    # comparison: common.Comparison = scenarios.windy.comparisons.timestep(random_wind=True)
    # comparison: common.Comparison = scenarios.cliff.comparisons.alpha()
    # comparison: common.Comparison = scenarios.cliff.comparisons.episode()
    # comparison: common.Comparison = scenarios.random_walk.comparisons.episode()

    application.Application(comparison)


if __name__ == '__main__':
    main()
