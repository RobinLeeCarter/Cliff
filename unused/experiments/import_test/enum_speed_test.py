import timeit
import time

import numpy as np
import utils

rng: np.random.Generator = np.random.default_rng()

v = np.random.uniform()
print(v)

v = rng.uniform()
print(v)

v = utils.uniform()
print(v)


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

rng: np.random.Generator = np.random.default_rng()
'''

numpy_code = '''
v = np.random.uniform()
'''

rng_code = '''
v = rng.uniform()
'''

numba_code = '''
v = utils.uniform()
'''


single_time = get_time_ns(numpy_code)
print(f"numpy_code   : {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")
numpy_time = single_time

single_time = get_time_ns(rng_code)
print(f"rng_code   : {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

single_time = get_time_ns(numba_code)
print(f"numba_code: {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

print(f"numba_code better ratio : {numpy_time/single_time:.1f}")
