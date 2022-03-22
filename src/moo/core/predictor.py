import numpy as np


class Predictor:

    def __init__(self, problem, eps=1e-4):
        self.problem = problem
        self.eps = eps
        self.n_f_evals = 0
        self.n_grad_evals = 0


class StepAdjust(Predictor):

    def __init__(self, problem, eps=1e-4):
        super().__init__(problem=problem,
                         eps=eps)

    def do(self, a, x, fx, t, v):
        P, Fp = [], []

        p = x + t * v
        fp = self.problem.evaluate(p)
        self.n_f_evals += 1
        # print('t0: {}'.format(t))
        while abs(np.dot(a, fp - fx)) > self.eps:
            t /= 2
            if t < 1e-4:
                break

            p = x + t * v
            fp = self.problem.evaluate(p)
            self.n_f_evals += 1

        # print('t1: {}'.format(t))
        P.append(p)
        Fp.append(fp)

        return {'p': P, 'fp': Fp}


class NoAdjustmentPredictors(Predictor):

    def do(self, a, x, fx, t, v, **kwargs):
        P, Fp = [], []

        p = x + t * v
        P.append(p)
        Fp.append(self.problem.evaluate(p))
        self.n_f_evals += 1

        return {'p': P, 'fp': Fp}
