import numpy as np
from numpy.linalg import norm


def squared_norm_mul_along_axis(vector, arr, axis=0):
    ans = norm(np.sum(np.apply_along_axis(lambda x: np.multiply(x, vector), axis=axis, arr=arr), axis=0)) ** 2
    return ans