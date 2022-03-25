from __future__ import annotations
import os

from mdp import Application
from mdp import common

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

comparison_type: common.ComparisonType = common.ComparisonType.WINDY_TIMESTEP


def main():
    # os_environ_settings.dummy = None    # for pycharm code inspection only
    Application(comparison_type)


if __name__ == '__main__':
    main()
