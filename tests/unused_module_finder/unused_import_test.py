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
    project_dependencies('agent')

# import importlib

# import sys
# module_names = set(sys.modules) & set(globals())
# all_modules = [sys.modules[name] for name in module_names]
#
# for m in all_modules:
#     print(m)

# def is_module(x):
#     return str(type(x)) == "<class 'module'>"
#
#
# def show_deps(mod):
#     for name in dir(mod):
#         val = getattr(mod, name)
#         if is_module(val):
#             print(name, val.__file__)
#
#
# my_mod = importlib.import_module("agent")
#
# show_deps(my_mod)

# empty_list = []
# for name, mod in sys.modules.items():
#     if mod:
#         path_list: list = getattr(mod, 'path', empty_list)
#         print(name, path_list)
# if path_list:
#     print(path_list[0])


# mod_dict = {name: mod for name, mod in sys.modules.items()}

app_dir = '/home/robin/Python/Projects/'


    # if hasattr(mod, '__file__'):
    #     print(name, mod.__file__)

# agent_mod = mod_dict['agent']

# print(agent_mod.__file__)

# first_mod = mod_list[0]
#
# print(first_mod)
# print(type(first_mod))
# for k, v in first_mod.__dict__.items():
#     print(k, v)

# print(first_mod.path[0])

# print(first_mod.__dict__)

# modules = {name: mod for name, mod in sys.modules.items()
#            if mod.__file__ is not None}

# for k, v in sys.modules.items():
#     if v.__file__ is not None:
#         print(f"key: {k}\tvalue: {v.__file__}")
    # print(type(k))
    # print(type(v))

#
# print("dir:")
# for name in dir():
#     print(name)

# k: str
# v: types.ModuleType
# v.__file__

# print(type(k))
# print(type(v))
# print(f"file: {v.__module__.}")

# print("environment" in sys.modules.keys())


# def imports():
#     for name, val in globals().items():
#         if isinstance(val, types.ModuleType):
#             yield val.__name__
#
#
# print("globals().items()")
# for val_name in imports():
#     print(val_name)


# print(f"constants.GAMMA: {constants.GAMMA}")
