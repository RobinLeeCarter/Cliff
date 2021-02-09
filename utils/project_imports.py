# runs standalone, don't import this file
from __future__ import annotations

import sys
import importlib


def project_dependencies(module_name: str):
    importlib.import_module(module_name)

    app_dir = '/home/robin/Python/Projects/'

    imported_module_names = []

    for mod_name, mod in sys.modules.items():
        file = getattr(mod, '__file__', '')
        if str(file).startswith(app_dir)\
                and mod_name != '__main__'\
                and mod_name != module_name\
                and not mod_name.startswith(module_name + "."):
            imported_module_names.append(mod_name)

    for mod_name in sorted(imported_module_names):
        print(mod_name)


if __name__ == '__main__':
    project_dependencies('algorithm')
