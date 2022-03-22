import numpy as np
from numpy.linalg import norm, multi_dot
from scipy.optimize import minimize
import cvxpy as cp


def squared_norm_mul_along_axis(vector, arr, axis=0):
    ans = norm(np.sum(np.apply_along_axis(lambda x: np.multiply(x, vector), axis=axis, arr=arr), axis=0)) ** 2
    return ans


def sort_according_first_obj(F, X):
    ixs = np.argsort(F[:, 0])
    return F[ixs], X[ixs]


def in_pareto_front(dx, d, cvxpy=False):
    if cvxpy:
        v_cp = cp.Variable(dx.shape[1])
        delta_cp = cp.Variable()

        eq = cp.matmul(dx, v_cp) - delta_cp * d == 0.0
        loss = (cp.norm(v_cp, p=2) ** 2) / 2 - delta_cp

        constraints = [0 <= delta_cp, eq]
        objective = cp.Minimize(loss)

        prob = cp.Problem(objective, constraints)

        result = prob.solve()
        v, delta = v_cp.value, delta_cp.value

    else:
        var0 = np.zeros((dx.shape[1] + 1))  # .astype(np.float32)

        # d = d.astype(np.float32)
        cons = ({'type': 'eq', 'fun': lambda var: np.matmul(dx, var[:-1]) - var[-1] * d})
        # cons = ({'type': 'eq', 'fun': lambda var: multi_dot([dx, var[:-1]]) - var[-1] * d})

        bnds = [(None, None)] * dx.shape[1] + [(0, None)]
        res = minimize(lambda var: (norm(var[:-1], ord=2) ** 2) / 2 - var[-1],
                       x0=var0,
                       constraints=cons,
                       bounds=bnds)
        v, delta = res['x'][:-1], res['x'][-1]

    v /= norm(v)
    return v, delta


def step_size_norm(step_eps, dx, v):
    return step_eps / norm(np.matmul(dx, v))
