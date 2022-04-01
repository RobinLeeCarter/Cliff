from __future__ import annotations

from mdp import Application
from mdp import common


comparison_type: common.ComparisonType = common.ComparisonType.RANDOM_WALK_EPISODE


def main():
    Application(comparison_type)


if __name__ == '__main__':
    main()
