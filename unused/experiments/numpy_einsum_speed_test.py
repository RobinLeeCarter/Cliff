from numba import njit, prange
import timeit
import time
import numpy as np
import utils


@njit(cache=True, parallel=True)
def jitted_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = np.zeros(shape=a.shape[0], dtype=np.float64)
    for i in prange(a.shape[0]):
        for j in range(a.shape[1]):
            c[i] += a[i, j] * b[i, j]
    return c


@njit(cache=True, parallel=True)
def expected_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Sum_over_a( π(a|s).Q(s,a) )
    :returns np.einsum('ij,ij->i', policy_matrix, q_matrix)
    """
    out = np.zeros(shape=p.shape[0], dtype=np.float64)
    for i in prange(p.shape[0]):
        for j in range(p.shape[1]):
            out[i] += p[i, j] * q[i, j]
    return out


(S, A) = (1_000_000, 100)
P = np.random.rand(S, A)
Q = np.random.rand(S, A)

v1 = np.sum(P * Q, axis=1)
v2 = np.einsum('ij,ij->i', P, Q)
v3 = jitted_sum(P, Q)
v4 = expected_q(P, Q)


# jitted_sum.parallel_diagnostics(level=4)

print(np.sum(v1))
print(np.sum(v2))
print(np.sum(v3))
print(np.sum(v4))

timer = utils.Timer()

print("start")

timer.start()
for i in range(50):
    v3 = jitted_sum(P, Q)
timer.stop(show=False)
print(f"jitted_sum: {1000*(timer.total/50):.2f} ms")

timer.start()
for i in range(50):
    v4 = expected_q(P, Q)
timer.stop(show=False)
print(f"expected_q: {1000*(timer.total/50):.2f} ms")


def get_time_ns(stmt: str) -> float:
    iterations = 10
    total_times = timeit.repeat(setup=SETUP_CODE, stmt=stmt, timer=time.process_time_ns, number=iterations)
    single_time_: float = min(total_times) / iterations
    return single_time_


SETUP_CODE = '''
from numba import njit
import timeit
import time
import numpy as np

(S, A) = (1_000_000, 100)
P = np.random.rand(S, A)
Q = np.random.rand(S, A)
'''

sum_code = '''
V1 = np.sum(P * Q, axis=1)
'''

einsum_code = '''
v2 = np.einsum('ij,ij->i', P, Q)
'''


single_time = get_time_ns(sum_code)
print(f"sum_code: {single_time/10**6:.2f} ms")
print(f"{10**9/single_time:.0f} per second")

single_time = get_time_ns(einsum_code)
print(f"einsum_code: {single_time/10**6:.2f} ms")
print(f"{10**9/single_time:.0f} per second")

