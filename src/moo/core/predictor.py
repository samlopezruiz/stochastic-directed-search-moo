import numpy as np


class Predictor:

    def __init__(self, problem, eps=1e-4, limits=None, max_increment=None):
        self.problem = problem
        self.eps = eps
        self.n_f_evals = 0
        self.n_grad_evals = 0
        self.limits = limits
        self.max_increment = max_increment
        self.min_fx = np.ones(self.problem.n_obj) * np.inf

    def is_good_predictor(self, fp_i):
        return self.within_limits(fp_i) and self.within_max_increment(fp_i)

    def within_limits(self, fx):
        return self.limits is None or not np.any(fx >= self.limits)

    def within_max_increment(self, fx):
        # TODO: only works for objectives with same magnitude, change to individual increases np.all(fx <= self.min_fx * self.max_increment)
        return self.max_increment is None or (np.sum(fx) - np.sum(self.min_fx)) / np.sum(self.min_fx) < self.max_increment

    # def update_min(self, fx):
    #     self.min_fx = fx if sum(fx) < sum(self.min_fx) else self.min_fx

    def good_predictors(self, P, Fp):
        ans = {'p': [], 'fp': []}
        for p, fp in zip(P, Fp):
            if self.is_good_predictor(fp):
                ans['p'].append(p)
                ans['fp'].append(fp)
        # print('ans[fp]', ans['fp'])
        return ans


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

        return self.good_predictors(P, Fp)


class NoAdjustmentPredictors(Predictor):

    def do(self, a, x, fx, t, v, **kwargs):
        P, Fp = [], []

        p = x + t * v
        P.append(p)
        Fp.append(self.problem.evaluate(p))
        self.n_f_evals += 1

        return self.good_predictors(P, Fp)


class LimitsPredictors(Predictor):

    def do(self, a, x, fx, t, v, **kwargs):
        algorithm = kwargs['algorithm']
        P, Fp = [], []

        p = x + t * v

        fp = self.problem.evaluate(p)
        self.n_f_evals += 1
        if not self.is_good_predictor(fp):
            # print('adjusting last predictor')

            t_h, t_l = t, 0
            while True:
                t = (t_h + t_l) / 2

                p = x + t * v
                fp = self.problem.evaluate(p)
                # print({'fp': fp, 't': round(t, 4), 't_h': round(t_h, 4), 't_l': round(t_l, 4),
                #        'diff': round(t_h - t_l, 4), 'break': t_h - t_l < self.eps})

                self.n_f_evals += 1

                if t_h - t_l < self.eps:
                    algorithm.end_flag = True
                    if not self.is_good_predictor(fp):
                        t = t_l
                        p = x + t * v
                        fp = self.problem.evaluate(p)
                    P.append(p)
                    Fp.append(fp)
                    break

                if self.is_good_predictor(fp):
                    t_l = t
                else:
                    t_h = t

        else:
            P.append(p)
            Fp.append(fp)

        return self.good_predictors(P, Fp)
