from __future__ import annotations
# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#     from mdp import common
import os_environ_settings
# import cProfile

import mdp

# from mdp import application, scenarios


def main():
    os_environ_settings.dummy = None    # for pycharm code inspection only
    mdp.Application(comparison_type=mdp.ComparisonType.JACKS_POLICY_ITERATION_V_NP_JIT)


# if __name__ == '__main__':
main()
# cProfile.run('main()', 'jacks_policy_iteration_v_np.prof')
