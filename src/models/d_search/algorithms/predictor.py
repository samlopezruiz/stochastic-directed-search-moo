import numpy as np


class Predictor:

    def __init__(self, problem, eps=1e-4):
        self.problem = problem
        self.eps = eps


class LeftPredictor(Predictor):

    def __init__(self, problem, eps=1e-4):
        super().__init__(problem=problem,
                         eps=eps)

    def do(self, a, x, fx, t, v):
        p = x - t * v
        fp = self.problem.evaluate(p)[0]

        while abs(np.dot(a, fp - fx)) > self.eps:
            t /= 2
            if t < self.eps:
                break

            p = x + t * v
            fp = self.problem.evaluate(p)[0]

        return p[0], fp
