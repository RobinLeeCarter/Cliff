import timeit
import time

import random
import numpy as np

from mdp.task._position_move.model import action
from mdp.task.cliff.model import environment_parameters
from mdp.task.cliff.model.environment import Environment

environment_parameters_ = environment_parameters.default
environment = Environment(environment_parameters_)
environment.build()
# left, right, top, bottom
moves: list[action.Action] = environment.actions
probabilities: list[float] = [0.1, 0.2, 0.3, 0.4]

move_dict: dict[action.Action, float] = {}
for move, probability in zip(moves, probabilities):
    move_dict[move] = probability

moves_np = np.array(moves, dtype=action.Action)
probabilities_np = np.array(probabilities, dtype=float)
rng: np.random.Generator = np.random.default_rng()

my_action = random.choices(moves, weights=probabilities)[0]
print(my_action)

my_action = random.choices([*move_dict.keys()], weights=[*move_dict.values()])[0]
print(my_action)

my_action = random.choices(list(move_dict.keys()), weights=list(move_dict.values()))[0]
print(my_action)

my_action = random.choices([*move_dict.keys()], weights=move_dict.values())[0]
print(my_action)

if not move_dict:
    pass

# my_action = rng.choice(moves_np, p=probabilities_np)
# print(my_action)


def get_time_ns(stmt: str) -> float:
    iterations = 100_000
    total_times = timeit.repeat(setup=SETUP_CODE, stmt=stmt, timer=time.process_time_ns, number=iterations)
    single_time_: float = min(total_times) / iterations
    return single_time_


SETUP_CODE = '''
import random
import numpy as np

from mdp import common
from mdp.scenarios.position_move.model import action
from mdp.scenarios.cliff.model import environment_parameters
from mdp.scenarios.cliff.model.environment import Environment

environment_parameters_ = environment_parameters.default
environment = Environment(environment_parameters_)
environment.build()
# left, right, top, bottom
moves: list[action.Action] = environment.actions
probabilities: list[float] = [0.1, 0.2, 0.3, 0.4]

move_dict: dict[action.Action, float] = {}
for move, probability in zip(moves, probabilities):
    move_dict[move] = probability

moves_np = np.array(moves, dtype=action.Action)
probabilities_np = np.array(probabilities, dtype=float)
rng = common.rng
'''

random_choices_from_lists_code = '''
my_action = random.choices(moves, weights=probabilities)[0]
'''

random_choices_from_unpack_code = '''
my_action = random.choices([*move_dict.keys()], weights=[*move_dict.values()])[0]
'''

random_choices_from_dict_code = '''
my_action = random.choices(list(move_dict.keys()), weights=list(move_dict.values()))[0]
'''

random_choices_from_dict_invalid_code = '''
my_action = random.choices([*move_dict.keys()], weights=move_dict.values())[0]
'''

null_test_code = '''
if not move_dict:
    pass
'''


numpy_choice_from_array_code = '''
my_action = rng.choice(moves_np, p=probabilities_np)
'''


single_time = get_time_ns(random_choices_from_lists_code)
print(f"random_choices_from_lists_code: {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

single_time = get_time_ns(random_choices_from_unpack_code)
print(f"random_choices_from_unpack_code: {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

single_time = get_time_ns(random_choices_from_dict_code)
print(f"random_choices_from_dict_code: {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

single_time = get_time_ns(random_choices_from_dict_invalid_code)
print(f"random_choices_from_dict_invalid_code: {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

single_time = get_time_ns(null_test_code)
print(f"null_test_code: {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

# single_time = get_time_ns(numpy_choice_from_array_code)
# print(f"numpy_choice_from_array_code: {single_time:.0f} ns")
# print(f"{10**9/single_time:.0f} per second")
