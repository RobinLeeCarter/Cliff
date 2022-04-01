from __future__ import annotations

from mdp import Application
from mdp import common


comparison_type: common.ComparisonType = common.ComparisonType.WINDY_TIMESTEP_RANDOM


def main():
    Application(comparison_type)


if __name__ == '__main__':
    main()
