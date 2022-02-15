import numpy as np
from numpy.linalg import pinv, norm

from src.models.d_search.utils.oper import squared_norm_mul_along_axis


class Corrector:

    def __init__(self, problem, t_fun, a_fun, maxiter, eps=1e4):
        self.problem = problem
        self.t_fun = t_fun
        self.a_fun = a_fun
        self.eps = eps
        self.maxiter = maxiter

    def bound_x(self, x):
        x = np.minimum(x, self.problem.xu) if self.problem.xu else x
        x = np.maximum(x, self.problem.xl) if self.problem.xl else x
        return x


class DsCorrector(Corrector):

    def __init__(self, problem, t_fun, a_fun, maxiter, opt=False, eps=1e4):
        super().__init__(problem=problem,
                         t_fun=t_fun,
                         a_fun=a_fun,
                         eps=eps,
                         maxiter=maxiter)

        self.opt = opt

    def do(self, x, fx, a, t):
        t0 = t
        iter = 0

        # update a
        dx = self.problem.gradient(x)
        a = self.a_fun(a, dx)

        while squared_norm_mul_along_axis(a, dx) > self.eps and iter < self.maxiter:
            t = t0
            v = np.matmul(-pinv(dx), a)

            if self.opt:
                v /= norm(v)

            # compute step
            t = self.t_fun(x, fx, v, t)
            x += t * v

            x = self.bound_x(x)

            fx = self.problem.evaluate(x)
            dx = self.problem.gradient(x)
            a = self.a_fun(a, dx)

            iter += 1

        return x, fx
