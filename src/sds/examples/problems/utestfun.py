import os

import numpy as np
import pandas as pd
import tensorflow as tf
from pymoo.factory import get_problem, get_reference_directions
from pymoo.problems.autodiff import AutomaticDifferentiation

funcs = {}
from autograd import grad, jacobian
import autograd.numpy as npa

# F = @(x)[1/n^alpha*(x(:,1).^2+x(:,2).^2).^alpha;; 1/n^alpha*((1-x(:,1)).^2+(1-x(:,2)).^2).^alpha]

funcs['t3'] = {'f': lambda X: [(X[0] - 1) ** 4 + (X[1] - 1) ** 2, (X[0] + 1) ** 2 + (X[1] + 1) ** 2],
               'df': lambda X: np.array([[4 * (X[0] - 1) ** 3, 2 * (X[1] - 1)],
                                         [2 * (X[0] + 1), 2 * (X[1] + 1)]]),
               'lb': [-5, -5],
               'ub': [5, 5],
               'n_var': 2,
               'n_obj': 2,
               # 'ps': pd.read_csv(os.path.join('ref', 'ex3ps.csv'), header=None).to_numpy(),
               # 'pf': pd.read_csv(os.path.join('ref', 'ex3pf.csv'), header=None).to_numpy(),
               }


@tf.function
def t3_fx(x):
    X = [tf.slice(x, begin=[i], size=[1]) for i in range(x.shape[0])]
    y1 = (X[0] - 1) ** 4 + (X[1] - 1) ** 2
    y2 = (X[0] + 1) ** 2 + (X[1] + 1) ** 2
    y = tf.concat([y1, y2], axis=0)
    return y


funcs['t3_ad'] = {'f': t3_fx,
                  'df': None,
                  'lb': [-5, -5],
                  'ub': [5, 5],
                  'n_var': 2,
                  'n_obj': 2,
                  # 'ps': pd.read_csv(os.path.join('ref', 'ex3ps.csv'), header=None).to_numpy(),
                  # 'pf': pd.read_csv(os.path.join('ref', 'ex3pf.csv'), header=None).to_numpy(),
                  }

t3_scalar_funs = [
    lambda x: (x[0] - 1) ** 4 + (x[1] - 1) ** 2,
    lambda x: (x[0] + 1) ** 2 + (x[1] + 1) ** 2,
]

f3_ag = lambda x: npa.array([f(x) for f in t3_scalar_funs])
# f3_ag = lambda x: [f(x) for f in scalar_funs]

funcs['t3_ag'] = {'f': f3_ag,
                  'df': jacobian(f3_ag),
                  'lb': [-5, -5],
                  'ub': [5, 5],
                  'n_var': 2,
                  'n_obj': 2,
                  'ps': pd.read_csv(os.path.join('problems', 'ref', 'ex3ps.csv'), header=None).to_numpy(),
                  'pf': pd.read_csv(os.path.join('problems', 'ref', 'ex3pf.csv'), header=None).to_numpy(),
                  }

# F = @(x)[1/n^alpha*(x(:,1).^2+x(:,2).^2).^alpha;; 1/n^alpha*((1-x(:,1)).^2+(1-x(:,2)).^2).^alpha]
alpha1 = 1
n = 2
t5_scalar_funs = [
    lambda X: 1 / n ** alpha1 * (X[0] ** 2 + X[1] ** 2) ** alpha1,
    lambda X: (X[0] + 1) ** 2 + (X[1] + 1) ** 2,
]

f5 = lambda x: npa.array([f(x) for f in t5_scalar_funs])

x = np.arange(-1, 0, 0.01)

funcs['t5a1'] = {'f': f5,
                 'df': jacobian(f5),
                 'lb': [-1, -1],
                 'ub': [0, 0],
                 'n_var': 2,
                 'n_obj': 2,
                 'ps': np.vstack([x, x]).T,
                 'pf': np.apply_along_axis(f5, 1, np.vstack([x, x]).T),
                 }

alpha = 0.05
n = 2
t51_scalar_funs = [
    lambda X: 1 / n ** alpha * (X[0] ** 2 + X[1] ** 2) ** alpha,
    lambda X: (X[0] + 1) ** 2 + (X[1] + 1) ** 2,
]

f505 = lambda x: npa.array([f(x) for f in t51_scalar_funs])

funcs['t5a05'] = {'f': f505,
                  'df': jacobian(f505),
                  'lb': [-1, -1],
                  'ub': [0, 0],
                  'n_var': 2,
                  'n_obj': 2,
                  'ps': np.vstack([x, x]).T,
                  'pf': np.apply_along_axis(f505, 1, np.vstack([x, x]).T),
                  }

funcs['t3_n3'] = {'f': lambda X: [(X[0] - 1) ** 4 + (X[1] - 1) ** 2 + X[2], (X[0] + 1) ** 2 + (X[1] + 1) ** 2 - X[2]],
                  'df': lambda X: np.array([[4 * (X[0] - 1) ** 3, 2 * (X[1] - 1), 1],
                                            [2 * (X[0] + 1), 2 * (X[1] + 1), 1]]),
                  'lb': [-5, -5, -5],
                  'ub': [5, 5, 5],
                  'n_var': 3,
                  'n_obj': 2
                  }

t51_scalar_funs = [
    lambda X: 1 / n ** alpha * (X[0] ** 2 + X[1] ** 2) ** alpha,
    lambda X: (X[0] + 1) ** 2 + (X[1] + 1) ** 2,
]

#
# def zdt2(x):
#     n_obj = 3
#     X_, X_M = x[:, :n_obj - 1], x[:, n_obj - 1:]
#     g = npa.sum(npa.square(X_M - 0.5), axis=1)
#     return obj_fun(X_, g, alpha=1)
#
#
# def obj_fun(X_, g, alpha=1):
#     n_obj = 3
#     f = []
#     for i in range(0, n_obj):
#         _f = (1 + g)
#         _f *= npa.prod(npa.cos(npa.power(X_[:, :X_.shape[1] - i], alpha) * npa.pi / 2.0), axis=1)
#         if i > 0:
#             _f *= npa.sin(npa.power(X_[:, X_.shape[1] - i], alpha) * npa.pi / 2.0)
#
#         f.append(_f)
#
#     return npa.column_stack(f)

zdt2 = get_problem("dtlz2", n_var=4, n_obj=3)
funcs['zdt2'] = {'f': zdt2.evaluate,
                 'df': None,
                 'lb': [0., 0., 0., 0.],
                 'ub': [1., 1., 1., 1.],
                 'n_var': 4,
                 'n_obj': 3,
                 'ps': None,
                 'pf': get_problem("dtlz2").pareto_front(get_reference_directions("das-dennis", 3, n_partitions=12)),
                 }


class TestFuncs:

    def __init__(self):
        self.funcs = funcs

    def get(self, key):
        return self.funcs[key]


if __name__ == '__main__':
    # %%


    tf = TestFuncs()
    testfun = tf.get('zdt2')
    df = testfun['df']

    x = np.array([0.] * 4).astype(np.float)
    fx = testfun['f'](x)
    # dx = testfun['df'](x)

    problem = AutomaticDifferentiation(zdt2)
    F, dF = problem.evaluate(x, return_values_of=["F", "dF"])

    print('x: {}'.format(x))
    print('fx: {}'.format(fx))
    # print('xx: {}'.format(dx))
