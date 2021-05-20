import timeit
import time

import numpy as np
import utils

n = 100

my_list = [i for i in range(n)]

prob: float = 1.0/n

p = np.array([prob for i in my_list], dtype=float)
cum_p = np.cumsum(p)

i = utils.cum_p_choice(cum_p)
print(i)

i = np.random.randint(n)
print(i)

i = utils.n_choice(n)
print(i)


def get_time_ns(stmt: str) -> float:
    iterations = 1_000_000
    total_times = timeit.repeat(setup=SETUP_CODE, stmt=stmt, timer=time.process_time_ns, number=iterations)
    single_time_: float = min(total_times) / iterations
    return single_time_


SETUP_CODE = '''
import timeit
import time

import numpy as np
import utils

n = 100_000

my_list = [i for i in range(n)]

prob: float = 1.0/n

p = np.array([prob for i in my_list], dtype=float)
cum_p = np.cumsum(p)
'''

cum_p_code = '''
i = utils.cum_p_choice(cum_p)
'''

numpy_randint = '''
i = np.random.randint(n)
'''

n_choice_code = '''
i = utils.n_choice(n)
'''


single_time = get_time_ns(cum_p_code)
print(f"cum_p_code   : {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")
cum_p_code_time = single_time

# single_time = get_time_ns(numpy_randint)
# print(f"numpy_randint: {single_time:.0f} ns")
# print(f"{10**9/single_time:.0f} per second")
# numpy_randint_time = single_time

single_time = get_time_ns(n_choice_code)
print(f"n_choice_code: {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

print(f"n_choice_code better ratio : {cum_p_code_time/single_time:.1f}")
