import numpy as np


class StepSize:
    def __init__(self, problem, eps=1e-4):
        self.problem = problem
        self.eps = eps

    def do(self, *args, **kwargs):
        return self._do(*args, **kwargs)

    def _do(self, *args, **kwargs):
        return None


class Szc5(StepSize):

    def _do(self, x, fx, v, t):
        fy = self.problem.evaluate(x + t * v)
        # fx = self.evaluate(x)

        while np.any(fy > fx):
            t /= 5
            if t < self.eps:
                t = 0

            fy = self.problem.evaluate(x + t * v)

        return t, fy
