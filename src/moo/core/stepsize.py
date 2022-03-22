import numpy as np
from numpy.linalg import norm


class StepSize:
    def __init__(self, problem, eps=1e-4, maxiter=1e10):
        self.problem = problem
        self.eps = eps
        self.maxiter = maxiter
        self.x = None
        self.fx = None
        self.n_f_evals = 0
        self.n_grad_evals = 0

    def do(self, *args, **kwargs):
        return self._do(*args, **kwargs)

    def _do(self, *args, **kwargs):
        return None


class Dominance(StepSize):

    def _do(self, x, fx, v, t0, **kwargs):
        fy = self.problem.evaluate(x + t0 * v)
        self.n_f_evals += 1

        while np.any(fy > fx):
            t0 /= 5
            if t0 < 1e-10:
                t0 = 0

            fy = self.problem.evaluate(x + t0 * v)
            self.n_f_evals += 1

        return x + t0 * v, fy, t0


class WeightedDominance(StepSize):

    def _do(self, x, fx, v, t0, a, c=1, **kwargs):

        fy = self.problem.evaluate(x + t0 * v)
        self.n_f_evals += 1

        while np.any((fy - fx) * c * a < -self.eps):
            t0 /= 2
            if t0 < 1e-10:
                t0 = 0

            fy = self.problem.evaluate(x + t0 * v)
            self.n_f_evals += 1

        return x + t0 * v, fy, t0


class Armijo(StepSize):

    def _do(self, x, fx, v, t0, a, c=1, **kwargs):
        fy = self.problem.evaluate(x + t0 * v)
        self.n_f_evals += 1

        while np.any(fy > fx + c * t0 * a):
            t0 /= 2
            if t0 < 1e-10:
                t0 = 0

            fy = self.problem.evaluate(x + t0 * v)
            self.n_f_evals += 1

        return x + t0 * v, fy, t0


class AngleBisection(StepSize):

    def _do(self, a, v, x, fx, t0, **kwargs):

        if self.is_feasible(x + t0 * v, a, fx):
            return x, fx, 0

        l, h = 0, t0
        increasing_range = True

        iter = 0
        while iter < self.maxiter:
            c = (l + h) / 2
            if self.is_feasible(x + (c + t0) * v, a, fx):
                if increasing_range:
                    l, h = c, h * 2
                else:
                    l = c
                if h - l < 1e-2:
                    break
            elif c < 1e-10:
                break
            else:
                increasing_range = False
                h = c

            iter += 1

        return self.x, self.fx, (c + t0)

    def is_feasible(self, x, d, fx0):
        self.x, self.fx = x, self.problem.evaluate(x)
        self.n_f_evals += 1
        d_new = self.fx - fx0
        eps = np.dot(d, d_new) / (norm(d) * norm(d_new))
        return abs(1 - eps) <= self.eps
        # angle_eps = 1e-2
        # return angle_eps - eps <= 0
