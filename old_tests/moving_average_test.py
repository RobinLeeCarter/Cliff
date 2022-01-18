import numpy as np


def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def moving_average2(series: np.ndarray, window_size: int) -> np.ndarray:
    cum_sum = np.cumsum(np.insert(series, 0, 0))
    mov_av = (cum_sum[window_size:] - cum_sum[:-window_size]) / window_size
    full_ma = np.empty(shape=data.shape, dtype=float)
    full_ma.fill(np.NAN)
    nan_indent: int = int((window_width - 1) / 2)
    full_ma[nan_indent:-nan_indent] = mov_av
    return full_ma


data = np.random.randint(1, 10, 10)
print(data)

window_width = 3

# cumsum_vec = np.cumsum(np.insert(data, 0, 0))

# print(cumsum_vec)

# ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

ma = moving_average2(data, window_width)
print(ma)
print(ma.shape)

# full_ma = np.empty(shape=data.shape, dtype=float)
# full_ma.fill(np.NAN)
# nan_indent: int = int((window_width - 1) / 2)
# full_ma[nan_indent:-nan_indent] = ma

# print(full_ma)
# print(full_ma.shape)

# ma_vec = moving_average1(data, window_width)
#
# print(ma_vec)
# print(ma_vec.shape)
