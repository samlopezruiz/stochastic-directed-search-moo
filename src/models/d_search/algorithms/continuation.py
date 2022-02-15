import numpy as np
from numpy.linalg import norm, qr, pinv
from scipy.optimize import fmin_slsqp

from src.models.d_search.utils.oper import squared_norm_mul_along_axis


class Continuation:

    def __init__(self, problem, predictor, corrector, dir, termination, history=True, eps=1e-4):

        self.problem = problem
        self.termination = termination
        self.corrector = corrector
        self.predictor = predictor
        self.dir = dir
        self.history = history
        self.eps = eps
        self.end_loop = False
        self.initialize()

    def initialize(self):
        self.x = None
        self.f = None
        self.dx = None

        self.iter = 0
        self.X = []
        self.F = []

        # predictors
        self.X_p = []
        self.F_p = []

        # correctors
        self.X_c = []
        self.F_c = []

    # def evaluate(self, x):
    #     return self.f(x)

    def run(self, x0):

        # now the termination criterion should be set
        if self.termination is None:
            raise Exception("No termination criterion defined and algorithm has no default termination implemented!")

        self.X.append(x0)
        self.F.append(self.problem.evaluate(np.array(x0)))

        self.a = np.ones(self.problem.n_var) / self.problem.n_var
        self.x = np.array(x0)

        # while termination criterion not fulfilled
        while self.has_next():
            self.next()

        # create the result object to be returned
        res = self.result()

        return res

    def alpha_step(self):
        return self._alpha_func()

    def predictor_step(self, t, v):
        p, fp = self._predictor_step(t, v)
        return p, fp

    def corrector_step(self, p, fp, t):
        c, fc = self._corrector_step(p, fp, t)
        return c, fc

    def step_size(self, v):
        return self._step_size(v)

    def v_directions(self, q):
        return self._v_directions(q)

    def has_next(self):
        if self.end_loop:
            return False
        elif self.termination[0] == 'n_iter':
            print('{}/{}'.format(self.iter, self.termination[1]), end='\r')
            if self.iter < self.termination[1]:
                return True

        else:
            return False

    def next(self):

        self.fx = self.problem.evaluate(self.x)
        self.dx = self.problem.gradient(self.x)
        self.a = self.alpha_step()

        # if norm(np.matmul(self.a, self.dx), ord=2) ** 2 >= self.eps:
        if squared_norm_mul_along_axis(self.a, self.dx) >= self.eps:
            self.end_loop = True
            return

        q, r = qr(self.a.reshape(-1, 1), mode='complete')

        v = self.v_directions(q)
        t = self.step_size(v)
        p, fp = self.predictor_step(t, v)

        c, fc = self.corrector_step(p, fp, t)

        self.log_history(fp, p, fc, c)

        self.x = c
        self.iter += 1

    def log_history(self, fp, p, fc, c):
        self.X_p.append(p)
        self.F_p.append(fp)
        self.X_c.append(c)
        self.F_c.append(fc)
        self.X.append(c)
        self.F.append(fc)

    def result(self):
        return {'X_p': np.array(self.X_p),
                'F_p': np.array(self.F_p),
                'X_c': np.array(self.X_c),
                'F_c': np.array(self.F_c),
                'X': np.array(self.X),
                'F': np.array(self.F),
                }

    # implements following methods for specific implementation
    def _v_directions(self, q):
        return np.zeros((self.problem.n_vars, 1))

    def _alpha_func(self):
        return np.zeros((self.problem.n_vars, 1))

    def _step_size(self, v):
        return 0

    def _predictor_step(self, t, v):
        return np.zeros((self.problem.n_vars, 1)), np.zeros((self.problem.n_obj, 1))

    def _corrector_step(self, p, fp, t):
        return np.zeros((self.problem.n_vars, 1)), np.zeros((self.problem.n_obj, 1))


class DsContinuation(Continuation):
    def __init__(self, problem, predictor, corrector, termination, dir, eps=1e-4, history=True):
        super().__init__(problem=problem,
                         predictor=predictor,
                         corrector=corrector,
                         termination=termination,
                         dir=dir,
                         eps=eps,
                         history=history
                         )

    def _alpha_func(self):
        # quadratic optimization problem (qop)
        # for directed search

        alpha = fmin_slsqp(func=lambda a: norm(np.matmul(a, self.dx), ord=2) ** 2,
                           x0=self.a,
                           f_eqcons=lambda a: np.sum(a) - 1,
                           f_ieqcons=lambda a: a,
                           iprint=0)

        return alpha
        # return np.round_(alpha, decimals=6)

    def _v_directions(self, q):
        inv_dx = pinv(self.dx)
        v = []
        for i in range(1, q.shape[1]):
            v_i = np.matmul(-inv_dx, q[:, i])
            v_i = v_i / norm(v_i)
            v.append(self.dir * v_i)

        return np.array(v)

    def _step_size(self, v):
        e2 = 1
        t = [e2/abs(np.dot(self.dx[i, :].reshape(1, -1), v.T)[0])[0] for i in range(len(self.x))]
        return min(t)

    def _predictor_step(self, t, v):
        p, fp = self.predictor.do(self.a,
                                  self.x,
                                  self.fx,
                                  t,
                                  v)

        return p, fp

    def _corrector_step(self, p, fp, t):
        c, fc = self.corrector.do(p, fp,
                                  self.a,
                                  t)

        return c, fc
