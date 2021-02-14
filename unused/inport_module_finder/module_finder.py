from modulefinder import ModuleFinder
finder = ModuleFinder()
finder.run_script("import_agent.py")
# for name, mod in finder.modules.items():
#     print(name)

app_dir = '/home/robin/Python/Projects/'

# modules = {name: mod for name, mod in finder.modules.items()}

modules = {name: mod for name, mod in finder.modules.items()
           if mod.__file__ is not None
           and mod.__file__.startswith(app_dir)}

for name in modules.keys():
    print(f"{name}")

# if mod is not None
#     and mod.__file__ is not None
# and mod.__file__.startswith(app_dir)}
# and mod.__file__ is not None
