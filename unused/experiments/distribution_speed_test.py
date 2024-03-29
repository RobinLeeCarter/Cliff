import timeit
import time

from mdp import common
from mdp.task.jacks.model.state import State

states: list[State] = [
    State(is_terminal=False, ending_cars_1=1, ending_cars_2=2),
    State(is_terminal=False, ending_cars_1=3, ending_cars_2=2),
    State(is_terminal=False, ending_cars_1=4, ending_cars_2=2),
    State(is_terminal=False, ending_cars_1=5, ending_cars_2=2),
]
probabilities: list[float] = [0.1, 0.2, 0.3, 0.4]

distribution: common.Multinoulli[State] = common.Multinoulli()
dictionary: dict[State, float] = {}

for state, probability in zip(states, probabilities):
    distribution[state] = probability

for state, probability in zip(states, probabilities):
    dictionary[state] = probability


def get_time_ns(stmt: str) -> float:
    iterations = 100_000
    total_times = timeit.repeat(setup=SETUP_CODE, stmt=stmt, timer=time.process_time_ns, number=iterations)
    single_time_: float = min(total_times) / iterations
    return single_time_


SETUP_CODE = '''
import timeit
import time

from mdp import common
from mdp.scenarios.jacks.model.state import State

states: list[State] = [
    State(is_terminal=False, ending_cars_1=1, ending_cars_2=2),
    State(is_terminal=False, ending_cars_1=3, ending_cars_2=2),
    State(is_terminal=False, ending_cars_1=4, ending_cars_2=2),
    State(is_terminal=False, ending_cars_1=5, ending_cars_2=2),
]
probabilities: list[float] = [0.1, 0.2, 0.3, 0.4]

distribution: common.Distribution[State] = common.Distribution()
dictionary: dict[State, float] = {}
'''

distribution_code = '''
for state, probability in zip(states, probabilities):
    distribution[state] = probability
'''

dictionary_code = '''
for state, probability in zip(states, probabilities):
    dictionary[state] = probability
'''


single_time = get_time_ns(distribution_code)
print(f"distribution_code: {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

single_time = get_time_ns(dictionary_code)
print(f"dictionary_code: {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

print("Twice as slow if perform a check as items are entered so not included.")

