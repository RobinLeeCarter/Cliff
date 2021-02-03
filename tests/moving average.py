import numpy as np


def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


data = np.arange(10)
print(data)

window_width = 3

cumsum_vec = np.cumsum(np.insert(data, 0, 0))

print(cumsum_vec)

ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

print(ma_vec)
print(ma_vec.shape)

ma = moving_average(data, 3)

print(ma)
print(ma.shape)
