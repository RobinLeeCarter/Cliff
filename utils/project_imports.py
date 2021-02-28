# runs standalone, don'_t import this file
from __future__ import annotations

import sys
import importlib


def project_dependencies(module_name: str):
    importlib.import_module(module_name)

    app_dir = '/home/robin/Python/Projects/'

    excluded_names = ['mdp.common', 'utils', module_name]
    included_names = []
    top_level_names = ['mdp', 'mdp.model', 'mdp.view']
    output_names = []

    for mod_name, mod in sys.modules.items():
        file = getattr(mod, '__file__', '')
        if str(file).startswith(app_dir) and mod_name != '__main__':
            add_name = True
            for excluded_name in excluded_names:
                if mod_name == excluded_name or mod_name.startswith(excluded_name + "."):
                    add_name = False
            if add_name and mod_name not in top_level_names:
                included_names.append(mod_name)
    for mod_name in sorted(included_names):
        add_name = True
        for output_name in output_names:
            if mod_name == output_name or mod_name.startswith(output_name + "."):
                add_name = False
        if add_name:
            output_names.append(mod_name)

    # for mod_name in sorted(included_names):
    #     print(mod_name)

    for mod_name in sorted(output_names):
        print(mod_name)


if __name__ == '__main__':
    # project_dependencies('mdp.model.policy')
    # import mdp.model.scenarios.random_walk.environment
    project_dependencies('mdp.model.agent')
