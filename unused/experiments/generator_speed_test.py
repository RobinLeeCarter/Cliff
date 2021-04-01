import timeit
import time
from typing import Generator

from mdp import common
from mdp.scenarios import scenario_factory
from mdp.scenarios.jacks.model.state import State

scenario = scenario_factory.scenario_factory(common.ComparisonType.JACKS_POLICY_ITERATION_V)
scenario.build()
# noinspection PyProtectedMember
environment = scenario._model.environment


def non_terminal_states() -> Generator[State, None, None]:
    for state_ in environment.states:
        if not state_.is_terminal:
            yield state_


for state in environment.states:
    if not state.is_terminal:
        pass

for state in [state for state in environment.states if not state.is_terminal]:
    pass

for state in non_terminal_states():
    pass

print("start")

# x = state.ending_cars_1


def get_time_ns(stmt: str) -> float:
    iterations = 1_000
    total_times = timeit.repeat(setup=SETUP_CODE, stmt=stmt, timer=time.process_time_ns, number=iterations)
    single_time_: float = min(total_times) / iterations
    return single_time_


SETUP_CODE = '''
import timeit
import time
from typing import Generator

from mdp import common
from mdp.scenarios import scenario_factory
from mdp.scenarios.jacks.model.state import State

scenario = scenario_factory.scenario_factory(common.ComparisonType.JACKS_POLICY_ITERATION_V)
scenario.build()
# noinspection PyProtectedMember
environment = scenario._model.environment


def non_terminal_states() -> Generator[State, None, None]:
    for state_ in environment.states:
        if not state_.is_terminal:
            yield state_
'''

list_code = '''
for state in environment.states:
    if not state.is_terminal:
        pass
'''

list_comprehension_code = '''
for state in [state for state in environment.states if not state.is_terminal]:
    pass
'''

generator_code = '''
for state in non_terminal_states():
    pass
'''


single_time = get_time_ns(list_code)
print(f"list_code: {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

single_time = get_time_ns(list_comprehension_code)
print(f"list_comprehension_code: {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

single_time = get_time_ns(generator_code)
print(f"generator_code: {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")
