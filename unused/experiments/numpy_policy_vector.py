import numpy as np
a = np.arange(12).reshape(2, 3, 2)
b = np.array([2, 0])
c = np.empty(shape=(2, 2), dtype=int)

for i in range(2):      # single numpy routine?
    j = b[i]
    c[i, :] = a[i, j, :]

print(c)
