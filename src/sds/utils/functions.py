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


# simple moving average
class SMA:

    def __init__(self, n):
        self.n = n
        self.sma = None
        self.counter = 0

    def do(self, x):
        if self.sma is None:
            self.sma = x
        else:
            self.sma = self.sma + (x - self.sma) / self.counter
            # self.sma = (self.sma * (self.n - 1) / self.n) + x / self.n
        self.counter = min(self.counter + 1, self.n)

        return self.sma

    def reset(self):
        self.sma = None
        self.counter = 0


class MEAN:

    def __init__(self):
        self.n = 0
        self.mean = None
        self.counter = 0

    def do(self, x):
        if self.mean is None:
            self.mean = x
        else:
            self.mean = self.mean * (self.n / (self.n + 1)) + x / (self.n + 1)
        self.n += 1

        return self.mean

    def reset(self):
        self.mean = None
        self.counter = 0


# exponential moving average
class EMA:
    def __init__(self, n):
        self.ema = None
        self.counter = 0
        self.n = n
        self.c1 = 2 / (1 + self.counter)
        self.c2 = 1 - (2 / (1 + self.counter))

    def do(self, x):
        if self.ema is None:
            self.ema = x
        else:
            self.c1 = 2 / (1 + self.counter)
            self.c2 = 1 - (2 / (1 + self.counter))
            self.ema = x * self.c1 + self.c2 * self.ema

        self.counter = min(self.counter + 1, self.n)
        return self.ema

    def reset(self):
        self.ema = None
        self.counter = 0


# def ema(x, period, last_ema=None):
#     c1 = 2 / (1 + period)
#     c2 = 1 - (2 / (1 + period))
#     x = np.array(x)
#     if last_ema is None:
#         ema_x = np.array(x)
#         for i in range(1, ema_x.shape[0]):
#             ema_x[i] = x[i] * c1 + c2 * ema_x[i - 1]
#     else:
#         ema_x = np.zeros((len(x) + 1,))
#         ema_x[0] = last_ema
#         for i in range(1, ema_x.shape[0]):
#             ema_x[i] = x[i] * c1 + c2 * ema_x[i - 1]
#         ema_x = ema_x[1:]
#     return ema_x

if __name__ == '__main__':
    m = MEAN()

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    for xi in x:
        mean = m.do(xi)
        print(mean)

    print(mean)
