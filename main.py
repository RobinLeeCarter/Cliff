from __future__ import annotations

import os_environ_settings
import mdp


def main():
    os_environ_settings.dummy = None    # for pycharm code inspection only
    mdp.Application(comparison_type=mdp.ComparisonType.JACKS_POLICY_ITERATION_V)


if __name__ == '__main__':
    main()
