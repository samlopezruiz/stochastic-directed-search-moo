import numpy as np
from numpy.linalg import norm
from scipy.optimize import fmin_slsqp, minimize


def squared_norm_mul_along_axis(vector, arr, axis=0):
    ans = norm(np.sum(np.apply_along_axis(lambda x: np.multiply(x, vector), axis=axis, arr=arr), axis=0)) ** 2
    return ans


# def in_pareto_front(inv_dx, v, d):
# var0 = np.zeros((len(v) + 1))
# # var0[:-1] = v
# res = fmin_slsqp(func=lambda var: (norm(var[:-1], ord=2) ** 2) / 2 - var[-1],
#                  x0=var0,
#                  f_eqcons=lambda var: (var[:-1] - var[-1] * np.matmul(inv_dx, d).reshape(1, -1)).flatten(),
#                  f_ieqcons=lambda var: var[-1],
#                  iprint=0)
#
# v, delta = res[:-1], res[-1]
#
# v /= norm(v)
# return v, delta

def in_pareto_front(dx, d):
    var0 = np.zeros((dx.shape[1] + 1))
    # var = np.zeros((len(v) + 1))

    cons = ({'type': 'eq', 'fun': lambda var: np.matmul(dx, var[:-1]) - var[-1] * d})
    # cons = ({'type': 'eq', 'fun': lambda var: (var[:-1] - var[-1] * np.matmul(inv_dx, a).reshape(1, -1)).flatten()})

    bnds = [(None, None)] * dx.shape[1] + [(0, None)]
    res = minimize(lambda var: (norm(var[:-1], ord=2) ** 2) / 2 - var[-1],
                   x0=var0,
                   constraints=cons,
                   bounds=bnds,
                   # method='SLSQP',
                   # options={'ftol': 1e-6}
                   )
    v, delta = res['x'][:-1], res['x'][-1]

    v /= norm(v)
    return v, delta


def step_size_norm(step_eps, dx, v):
    return step_eps / norm(np.matmul(dx, v))
