import math
from copy import copy

import numpy as np
from numpy.linalg import pinv, norm, matrix_rank

from src.moo.utils.functions import squared_norm_mul_along_axis, in_pareto_front, step_size_norm, SMA


def calc_v(dx, a):
    inv_dx = pinv(dx)
    v = np.matmul(inv_dx, a)
    v /= norm(v)
    return v


class Corrector:

    def __init__(self,
                 problem,
                 t_fun,
                 a_fun,
                 maxiter,
                 step_eps,
                 in_pf_eps,
                 batch_gradient=False,
                 mean_grad_stop_criteria=False,
                 batch_ratio_stop_criteria=0.1
                 ):

        if hasattr(problem, 'train_n_batches'):
            self.n_batches_stop_criteria = math.ceil(problem.train_n_batches * batch_ratio_stop_criteria)
        else:
            self.n_batches_stop_criteria = 1

        self.mean_grad_stop_criteria = mean_grad_stop_criteria
        self.batch_gradient = batch_gradient
        self.problem = problem
        self.t_fun = t_fun
        self.a_fun = a_fun
        self.in_pf_eps = in_pf_eps
        self.maxiter = maxiter
        self.fx_hist = []
        self.x_hist = []
        self.n_f_evals = 0
        self.n_grad_evals = 0
        self.recalculate_t = False
        self.step_eps = step_eps

    def bound_x(self, x):
        x = np.minimum(x, self.problem.xu) if self.problem.xu is not None else x
        x = np.maximum(x, self.problem.xl) if self.problem.xl is not None else x
        return x

    def problem_dx_a(self, x, fx, a):
        dx = self.problem.grad_next_batch(x) if self.batch_gradient else self.problem.gradient(x)
        self.n_grad_evals += (1 / self.problem.train_n_batches if self.batch_gradient else 1)
        a = self.a_fun(a, dx)

        self.x_hist.append(x)
        self.fx_hist.append(fx)

        return dx, a

    def next_x(self, dx, v, x, fx, a, t):
        # used when corrector is called without a preceding predictor
        if self.recalculate_t:
            t = self.step_size(dx, v)
            self.recalculate_t = False

        return self.t_fun.do(a=a, v=v, x=x, t0=t, fx=fx)

    def reset_hist(self):
        self.fx_hist = []
        self.x_hist = []

    def dx_for_stop_criteria(self, x, dx):
        if self.mean_grad_stop_criteria and self.batch_gradient:
            dx = np.array([dx] + [self.problem.grad_next_batch(x) for _ in range(self.n_batches_stop_criteria - 1)])
            self.n_grad_evals += (self.n_batches_stop_criteria - 1) / self.problem.train_n_batches
            return np.mean(dx, axis=0)
        else:
            return dx

    def do(self, x, fx, a, t):
        self.reset_hist()
        self.recalculate_t = t is None
        iter = 0

        while iter < self.maxiter:
            dx, a = self.problem_dx_a(x, fx, a)

            dx_sc = self.dx_for_stop_criteria(x, dx)

            if self.stop_criteria(dx_sc, a):
                break

            v = calc_v(dx, a)

            x, fx, t = self.next_x(dx, v, x, fx, a, t)
            iter += 1

        return self.format_result(x, fx, t, dx)

    def format_result(self, x, fx, t, dx):
        return {'x': x, 'fx': fx, 't': t, 'dx': dx, 'x_hist': self.x_hist, 'fx_hist': self.fx_hist}

    def stop_criteria(self, dx, a):
        return False

    def step_size(self, dx, v):
        return step_size_norm(self.step_eps, dx, v)


class ProjectionCorrector(Corrector):

    def stop_criteria(self, dx, a):
        return norm(np.dot(dx.T, a.reshape(-1, 1))) ** 2 < self.in_pf_eps


class RankCorrector(Corrector):

    def stop_criteria(self, dx, a):
        return matrix_rank(dx, tol=self.in_pf_eps) < dx.shape[0]


class DeltaCorrector(Corrector):

    def __init__(self,
                 problem,
                 t_fun,
                 a_fun,
                 maxiter,
                 in_pf_eps=1e4,
                 step_eps=2e-2,
                 recalc_v=True,
                 use_cvxpy=False,
                 batch_gradient=False,
                 mean_grad_stop_criteria=False,
                 batch_ratio_stop_criteria=0.1
                 ):

        super().__init__(problem,
                         t_fun,
                         a_fun,
                         maxiter,
                         step_eps,
                         in_pf_eps,
                         batch_gradient,
                         mean_grad_stop_criteria,
                         batch_ratio_stop_criteria)

        self.use_cvxpy = use_cvxpy
        self.recalc_v = recalc_v

    def do(self, x, fx, a, t):
        self.reset_hist()
        self.recalculate_t = t is None
        iter = 0

        while iter < self.maxiter:
            dx, a = self.problem_dx_a(x, fx, a)

            dx_sc = self.dx_for_stop_criteria(x, dx)
            v, delta = in_pareto_front(dx_sc, a, cvxpy=self.use_cvxpy)

            if delta < self.in_pf_eps:
                break

            if self.recalc_v:
                v = calc_v(dx, a)

            x, fx, t = self.next_x(dx, v, x, fx, a, t)
            iter += 1

        return self.format_result(x, fx, t, dx)
