import numpy as np
a = np.arange(18).reshape(3, 3, 2)
b = np.array([2, 0, 1])
# c = np.empty(shape=(3, 2), dtype=int)

print(a)
print('--------')

# for i in range(3):      # single numpy routine?
#     j = b[i]
#     c[i, :] = a[i, j, :]


c = a[np.arange(a.shape[0]), b, :]

print(c)
