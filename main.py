from __future__ import annotations

import os_environ_settings
import mdp


def main():
    os_environ_settings.dummy = None    # for pycharm code inspection only
    mdp.Application(scenario_type=mdp.ScenarioType.CLIFF_EPISODE)


if __name__ == '__main__':
    main()
