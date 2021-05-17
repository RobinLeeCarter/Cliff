import timeit
import time

import numpy as np

n = 100000

my_list = [(i*2, i*3) for i in range(n)]
s_list = [i*2 for i in range(n)]
a_list = [i*3 for i in range(n)]

my_array = np.array(my_list, dtype=tuple)
s_array = np.array([s for s, a in my_list], dtype=int)
a_array = np.array([a for s, a in my_list], dtype=int)

for i in range(len(my_list)):
    s, a = my_list[i]

for i in range(len(my_list)):
    s = s_list[i]
    a = a_list[i]

for i in range(my_array.shape[0]):
    s, a = my_array[i]

for i in range(my_array.shape[0]):
    s = s_array[i]
    a = a_array[i]


def get_time_ns(stmt: str) -> float:
    iterations = 1
    total_times = timeit.repeat(setup=SETUP_CODE, stmt=stmt, timer=time.process_time_ns, number=iterations)
    single_time_: float = min(total_times) / iterations
    return single_time_


SETUP_CODE = '''
import timeit
import time

import numpy as np

n = 1000000

my_list = [(i*2, i*3) for i in range(n)]
s_list = [i*2 for i in range(n)]
a_list = [i*3 for i in range(n)]

my_array = np.array(my_list, dtype=tuple)
s_array = np.array([s for s, a in my_list], dtype=int)
a_array = np.array([a for s, a in my_list], dtype=int)
'''

list_code = '''
for i in range(len(my_list)):
    s, a = my_list[i]
'''

list_sep = '''
for i in range(len(my_list)):
    s = s_list[i]
    a = a_list[i]
'''

numpy_code = '''
for i in range(my_array.shape[0]):
    s, a = my_array[i]
'''

numpy_separate = '''
for i in range(my_array.shape[0]):
    s = s_array[i]
    a = a_array[i]
'''


single_time = get_time_ns(list_code)
print(f"list_code : {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")
list_single = single_time

single_time = get_time_ns(list_sep)
print(f"list_sep  : {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

single_time = get_time_ns(numpy_code)
print(f"numpy_code: {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

single_time = get_time_ns(numpy_separate)
print(f"numpy_sep : {single_time:.0f} ns")
print(f"{10**9/single_time:.0f} per second")

print(f"list better ratio : {single_time/list_single:.1f}")
