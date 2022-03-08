from copy import copy

import numpy as np
from numpy.linalg import pinv, norm

from src.models.d_search.utils.oper import squared_norm_mul_along_axis, in_pareto_front, step_size_norm


class Corrector:

    def __init__(self, problem, t_fun, a_fun, maxiter, in_pf_eps=1e4):
        self.problem = problem
        self.t_fun = t_fun
        self.a_fun = a_fun
        self.in_pf_eps = in_pf_eps
        self.maxiter = maxiter

    def bound_x(self, x):
        x = np.minimum(x, self.problem.xu) if self.problem.xu is not None else x
        x = np.maximum(x, self.problem.xl) if self.problem.xl is not None else x
        return x


class DsCorrector(Corrector):

    def __init__(self, problem, t_fun, a_fun, maxiter, opt=True, in_pf_eps=1e4):
        super().__init__(problem=problem,
                         t_fun=t_fun,
                         a_fun=a_fun,
                         in_pf_eps=in_pf_eps,
                         maxiter=maxiter)

        self.opt = opt
        self.n_f_evals = 0
        self.n_grad_evals = 0

    def do(self, x, fx, a, v, t):
        t0 = t
        iter = 0

        # update a
        dx = self.problem.gradient(x)
        self.n_grad_evals += 1
        a = self.a_fun(a, dx)

        tols = []
        tol = squared_norm_mul_along_axis(a, dx)
        while tol > self.in_pf_eps and iter < self.maxiter:
            t = t0
            v = np.matmul(-pinv(dx), a)

            if self.opt:
                v /= norm(v)

            # compute step
            x, fx, t = self.t_fun.do(x, fx, v, t)

            x = self.bound_x(x)

            dx = self.problem.gradient(x)
            self.n_grad_evals += 1
            a = self.a_fun(a, dx)

            iter += 1
            tol = squared_norm_mul_along_axis(a, dx)
            tols.append(tol)

        return {'x': x, 'fx': fx, 'tols': tols, 'dx': dx}


class DeltaCriteriaCorrector(Corrector):

    def __init__(self, problem, t_fun, a_fun, maxiter, opt=True, in_pf_eps=1e4, step_eps=None):
        super().__init__(problem=problem,
                         t_fun=t_fun,
                         a_fun=a_fun,
                         in_pf_eps=in_pf_eps,
                         maxiter=maxiter)

        self.step_eps = step_eps
        self.opt = opt
        self.fx_hist = []
        self.x_hist = []
        # self.x = None
        # self.fx = None
        self.n_f_evals = 0
        self.n_grad_evals = 0

    def do(self, x, fx, a, t):
        recalculate_t = t is None

        x = copy(x)
        self.fx_hist = []
        self.x_hist = []
        # self.x, self.fx = x, fx
        iter = 0

        while iter < self.maxiter:
            dx = self.problem.gradient(x)
            self.n_grad_evals += 1
            a = self.a_fun(a, dx)

            self.x_hist.append(x)
            self.fx_hist.append(fx)

            v, delta = in_pareto_front(dx, a)

            if delta < self.in_pf_eps:
                return {'x': x, 'fx': fx, 't': t, 'dx': dx, 'critical_point': iter > 0, 'x_hist': self.x_hist,
                        'fx_hist': self.fx_hist}
            else:
                inv_dx = pinv(dx)
                v = np.matmul(inv_dx, a)
                v /= norm(v)

            # used when corrector is called without a preceding predictor
            if recalculate_t:
                t = self.step_size(dx, v)

            x, fx, t = self.t_fun.do(a=a, v=v, x=x, t0=t, fx=fx)
            iter += 1

        return {'x': x, 'fx': fx, 't': t, 'dx': dx, 'critical_point': False, 'x_hist': self.x_hist,
                'fx_hist': self.fx_hist}

    def step_size(self, dx, v):
        return step_size_norm(self.step_eps, dx, v)
