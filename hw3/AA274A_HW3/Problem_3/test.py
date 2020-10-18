import numpy as np
import linear_filter

F = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])

I = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(linear_filter.mono_corr(F, I))