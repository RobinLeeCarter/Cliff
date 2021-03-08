import timeit
import time

import random

from mdp.scenarios.factory import environment_factory
from mdp.scenarios.jacks import action
from mdp.scenarios.cliff import environment_parameters


environment_parameters_ = environment_parameters.default
environment_ = environment_factory.environment_factory(environment_parameters_)
# left, right, top, bottom
moves: list[action.Action] = environment_.actions
probabilities: list[float] = [0.1, 0.2, 0.3, 0.4]

my_action = random.choices(moves, weights=probabilities)
print(my_action)

SETUP_CODE = '''
import random
import numpy as np

from mdp.scenarios.factory import environment_factory
from mdp.scenarios.position_move import state, action
from mdp.scenarios.cliff import environment_parameters


environment_parameters_ = environment_parameters.default
environment_ = environment_factory.environment_factory(environment_parameters_)
# left, right, top, bottom
moves: list[action.Action] = environment_.actions
probabilities: list[float] = [0.1, 0.2, 0.3, 0.4]
'''

TEST_CODE = '''
my_action = random.choices(moves, weights=probabilities)
'''

iterations = 100_000

print("perf_counter...")

total_time = timeit.timeit(setup=SETUP_CODE, stmt=TEST_CODE, timer=time.perf_counter_ns, number=iterations)
single_time: float = total_time / iterations
print(f"time: {single_time:.0f} ns")

total_time = timeit.timeit(setup=SETUP_CODE, stmt=TEST_CODE, number=iterations)
single_time: float = total_time*(10 ** 9 / iterations)
print(f"time: {single_time:.0f} ns")

print("repeated perf_counter...")

total_times = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, timer=time.perf_counter_ns, number=iterations)
single_time: float = min(total_times) / iterations
print(f"time: {single_time:.0f} ns")

total_times = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, number=iterations)
single_time: float = min(total_times) * (10 ** 9 / iterations)
print(f"time: {single_time:.0f} ns")

print("process_counter...")

total_time = timeit.timeit(setup=SETUP_CODE, stmt=TEST_CODE, timer=time.process_time_ns, number=iterations)
single_time: float = total_time / iterations
print(f"time: {single_time:.0f} ns")

total_time = timeit.timeit(setup=SETUP_CODE, stmt=TEST_CODE, timer=time.process_time, number=iterations)
single_time: float = total_time*(10 ** 9 / iterations)
print(f"time: {single_time:.0f} ns")

print("repeated process_counter...")

total_times = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, timer=time.process_time_ns, number=iterations)
single_time: float = min(total_times) / iterations
print(f"time: {single_time:.0f} ns")

total_times = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, timer=time.process_time, number=iterations)
single_time: float = min(total_times) * (10 ** 9 / iterations)
print(f"time: {single_time:.0f} ns")


# number = 100_000
# total_time = timeit.timeit(setup=SETUP_CODE, stmt=TEST_CODE, timer=time.perf_counter_ns, number=number)
# time: float = total_time/number
# print(f"time: {time:.0f} ns")
#
#
# total_time = timeit.timeit(setup=SETUP_CODE, stmt=TEST_CODE, number=number)
# time: float = total_time*(10**9/number)
# print(f"time: {time:.0f} ns")

# print('Binary search time: {}'.format(min(times)))

# timeit.timeit('random.choices(moves, weights=probabilities)', number=10000)


