import numpy as np
from numpy.linalg import norm, qr
from scipy.optimize import fmin_slsqp

from src.models.d_search.core.problem import ContinuationProblem
from src.models.d_search.utils.utestfun import TestFuncs

if __name__ == '__main__':
    #%%
    tf = TestFuncs()
    testfun = tf.get('t3')
    problem = ContinuationProblem(f=testfun['f'],
                                  df=testfun['df'],
                                  n_var=testfun['n_var'],
                                  n_obj=testfun['n_var'],
                                  xl=testfun['lb'],
                                  xu=testfun['ub'])

    x0 = np.array([-1, -1])
    a0 = np.array([0.5, 0.5])
    dx = problem.gradient(x0)

    # Constraints:
    #   sum(alpha) = 1
    #   alpha_i >= 0
    alpha = fmin_slsqp(func=lambda a: norm(np.matmul(a, dx), ord=2) ** 2,
                       x0=a0,
                       f_eqcons=lambda a: np.sum(a) - 1,
                       f_ieqcons=lambda a: a)

    q, r = qr(alpha.reshape(-1, 1), mode='complete')

    res = norm(np.sum(np.apply_along_axis(lambda x: np.multiply(x, a0), axis=0, arr=q), axis=0)) ** 2

