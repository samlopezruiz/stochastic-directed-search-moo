import math
import time

import numpy as np
from numpy.linalg import norm, qr, pinv
from scipy.optimize import fmin_slsqp

from src.moo.core.boxes import Boxes
from src.moo.core.termination import NullTermination
from src.moo.utils.functions import step_size_norm


class Continuation:

    def __init__(self, problem, predictor, corrector, dir, termination, limits=None, history=True, debug=False):

        self.loop_time_start = time.time()
        self.problem = problem
        self.termination = termination if termination is not None else NullTermination()
        self.corrector = corrector
        self.predictor = predictor
        self.dir = dir
        self.history = history
        self.end_loop = False
        self.debug = debug
        self.n_f_evals = 0
        self.n_grad_evals = 0
        self.limits = limits
        self.initialize()

        self.dx_queue = []
        self.x_queue = []
        self.fx_queue = []

        # debuging vars
        self.vs = []
        self.as_ = []
        self.predictor_t = []
        self.corrector_t = []
        self.predictor_tols = []
        self.corrector_tols = []

        self.loop_f_evals = self.problem.n_f_evals
        self.loop_g_evals = self.problem.n_grad_evals

    def initialize(self):
        self.x = None
        self.f = None
        self.dx = None
        self.a = np.ones(self.problem.n_obj) / self.problem.n_obj

        self.iter = 0
        self.X = []
        self.F = []

        # predictors
        self.X_p = []
        self.F_p = []

        # correctors
        self.X_c = []
        self.F_c = []

    def run(self, x0):

        # now the termination criterion should be set
        if self.termination is None:
            raise Exception("No termination criterion defined and algorithm has no default termination implemented!")

        self.x = np.array(x0)
        self.fx = self.problem.evaluate(self.x)
        self.dx = self.problem.gradient(self.x)

        self.n_f_evals += 1
        self.n_grad_evals += 1
        self.X.append(self.x)
        self.F.append(self.fx)

        # self.a = np.ones(self.problem.n_obj) / self.problem.n_obj
        self.a = self.alpha_step()
        c, fc, dc = self.ds_x_to_pf(self.x, self.fx)
        self.add_to_queue(c, dc, fc)

        # reset n of evals
        self.loop_f_evals = self.problem.n_f_evals
        self.loop_g_evals = self.problem.n_grad_evals

        # while termination criterion not fulfilled
        while self.has_next():
            self.next()

        # create the result object to be returned
        res = self.result()

        return res

    def alpha_step(self):
        return self._alpha_func()

    def predictor_step(self, t, v):
        res = self._predictor_step(t, v)
        return res['p'], res['fp']

    def corrector_step(self, p, fp, t):
        res = self._corrector_step(p, fp, t)

        if self.debug and 'tols' in res:
            self.corrector_tols.append(res['tols'])

        if 't' in res and self.debug:
            self.corrector_t.append(t)

        history = {'c_hist': res.get('x_hist', None), 'fc_hist': res.get('fx_hist', None)}
        return res['x'], res['fx'], res['dx'], history

    def step_size(self, v):
        return self._step_size(v)

    def v_directions(self, q):
        return self._v_directions(q)

    def has_next(self):

        print('step: {} | loop time:{} s | f evals: {} | g evals: {}'.format(self.iter,
                                                                             round(time.time() - self.loop_time_start,
                                                                                   2),
                                                                             self.problem.n_f_evals - self.loop_f_evals,
                                                                             self.problem.n_grad_evals - self.loop_g_evals))

        self.loop_time_start = time.time()
        self.loop_f_evals = self.problem.n_f_evals
        self.loop_g_evals = self.problem.n_grad_evals

        if len(self.x_queue) == 0:
            print('empty queue: ended with {} it'.format(self.iter))
            return False

        return self.termination.has_next(self)

    def next(self):
        self.x = self.x_queue.pop()
        self.fx = self.fx_queue.pop()
        self.dx = self.dx_queue.pop()
        self.a = self.alpha_step()

        q, r = qr(self.a.reshape(-1, 1), mode='complete')
        v = self.v_directions(q)

        if not self.termination.has_next(self):
            return

        for v_i in v:

            t = self.step_size(v_i)
            p, fp = self.predictor_step(t, v_i)

            self.log_predictor(v_i, p, fp)

            for p_i, fp_i in zip(p, fp):

                if self.is_good_predictor(fp_i):
                    c, fc, dc, history = self.corrector_step(p_i, fp_i, t)
                    self.add_to_queue_using_boxes(c, fc, dc)
                    self.log_corrector(fc, c, history)

        self.iter += 1

    def ds_x_to_pf(self, x, fx):
        print('...getting to pareto front')
        self.log_predictor(np.zeros(self.x.shape), [x], [fx])
        c, fc, dc, history = self.corrector_step(x, fx, t=None)
        self.log_corrector(fc, c, history)
        return c, fc, dc

    def log_predictor(self, v_i, p, fp):
        self.vs.append(v_i)
        self.as_.append(self.a)
        self.X_p.append(p)
        self.F_p.append(fp)

    def log_corrector(self, fc, c, history):
        if self.limits is None or not np.any(fc >= self.limits):
            self.X_c.append(history['c_hist'])
            self.F_c.append(history['fc_hist'])
            self.X.append(c)
            self.F.append(fc)

    def format_results(self, results):
        compiled_res = {}
        for key in ['X', 'F', 'vs', 'as']:
            if key in ['X', 'F']:
                # remove last element to regularize X,F dimensions with vs,as
                compiled_res[key + "_r"] = np.vstack([res[key][:-1, :] for res in results])
            compiled_res[key] = np.vstack([res[key] for res in results])

    def add_to_queue_using_boxes(self, c, fc, dc):
        if self.limits is None or not np.any(fc >= self.limits):
            self.add_to_queue(c, dc, fc)

    def add_to_queue(self, c, dc, fc):
        self.x_queue.append(c)
        self.fx_queue.append(fc)
        self.dx_queue.append(dc)

    def is_good_predictor(self, fp_i):
        return self.limits is None or not np.any(fp_i >= self.limits)

    def result(self):
        population = {'X_p': self.X_p,
                      'F_p': self.F_p,
                      'X_c': self.X_c,
                      'F_c': self.F_c,
                      'X': np.array(self.X),
                      'F': np.array(self.F),
                      'predictor_t': np.array(self.predictor_t),
                      'corrector_t': np.array(self.corrector_t),
                      'vs': np.array(self.vs),
                      'as': np.array(self.as_),
                      'P_tols': np.array(self.predictor_tols),
                      'C_tols': self.corrector_tols,
                      }

        tot_f_evals = self.predictor.n_f_evals + self.corrector.n_f_evals + self.corrector.t_fun.n_f_evals + \
                      self.n_f_evals
        tot_grad_evals = self.predictor.n_grad_evals + self.corrector.n_grad_evals + self.corrector.t_fun.n_grad_evals + \
                         self.n_grad_evals

        evaluations = {'f': {'total': tot_f_evals,
                             'predictor': self.predictor.n_f_evals,
                             'corrector': self.corrector.n_f_evals,
                             'continuation': self.n_f_evals,
                             't_fun_corrector': self.corrector.t_fun.n_f_evals},

                       'grad': {'total': tot_grad_evals,
                                'predictor': self.predictor.n_grad_evals,
                                'corrector': self.corrector.n_grad_evals,
                                'continuation': self.n_grad_evals,
                                't_fun_corrector': self.corrector.t_fun.n_grad_evals},
                       }

        return {'population': population, 'evaluations': evaluations}

    # implements following methods for specific implementation

    def _v_directions(self, q):
        raise NotImplementedError

    def _alpha_func(self):
        # quadratic optimization core (qop)
        # for directed search
        alpha = fmin_slsqp(func=lambda a: norm(np.matmul(a, self.dx), ord=2) ** 2,
                           x0=self.a,
                           f_eqcons=lambda a: np.sum(a) - 1,
                           f_ieqcons=lambda a: a,
                           iprint=0)

        return alpha

    def _step_size(self, v):
        raise NotImplementedError

    def _predictor_step(self, t, v):
        raise NotImplementedError

    def _corrector_step(self, p, fp, t):
        raise NotImplementedError


class DsContinuation(Continuation):
    def __init__(self,
                 problem,
                 predictor,
                 corrector,
                 termination,
                 dir=1,
                 step_eps=1,
                 history=True,
                 debug=False,
                 limits=None,
                 **kwargs):
        super().__init__(problem=problem,
                         predictor=predictor,
                         corrector=corrector,
                         termination=termination,
                         dir=dir,
                         limits=limits,
                         debug=debug,
                         history=history
                         )

        self.step_eps = step_eps

    def _v_directions(self, q):
        inv_dx = pinv(self.dx)
        v = []
        for i in range(1, q.shape[1]):
            v_i = np.matmul(-inv_dx, q[:, i])
            v_i = v_i / norm(v_i)
            v.append(self.dir * v_i)

        return v

    def _step_size(self, v):
        t = step_size_norm(self.step_eps, self.dx, v)
        return t

    def _predictor_step(self, t, v):
        res = self.predictor.do(self.a, self.x, self.fx, t, v)

        return res

    def _corrector_step(self, p, fp, t):
        res = self.corrector.do(p, fp, -self.a, t)

        return res


class ContinuationBoxes(DsContinuation):
    def __init__(self,
                 problem,
                 predictor,
                 corrector,
                 f_limits,
                 h_max=None,
                 step_eps=1e-4,
                 history=True,
                 debug=False,
                 c=0.7,
                 **kwargs):

        super().__init__(problem=problem,
                         predictor=predictor,
                         corrector=corrector,
                         termination=None,
                         dir=1,
                         step_eps=step_eps,
                         debug=debug,
                         history=history,
                         )

        if h_max is None:
            h_max = problem.n_obj * math.ceil(math.log(max(f_limits[:, 1] - f_limits[:, 0]) / (step_eps * c), 2))
        print('using h_max: {}'.format(h_max))
        self.boxes = Boxes(h_max, f_limits)

    # overwrite to disable tolerance stop condition
    def tol_within_thold(self):
        return True

    def _v_directions(self, q):
        inv_dx = pinv(self.dx)

        v = []
        for i in range(1, q.shape[1]):
            v_i = np.matmul(-inv_dx, q[:, i])
            v_i = v_i / norm(v_i)
            for d in [-1, 1]:
                v.append(d * v_i)

        return v

    def add_to_queue_using_boxes(self, c, fc, dc):
        node, was_created = self.boxes.insert(fc, c, None)
        if was_created and node is not None:
            self.x_queue.append(c)
            self.fx_queue.append(fc)
            self.dx_queue.append(dc)

    def is_good_predictor(self, fp_i):
        # return True
        # case when fx is outside the limits
        if self.boxes.has_box(fp_i) is None:
            return False

        return not self.boxes.has_box(fp_i)


class BiDirectionalDsContinuation:
    def __init__(self,
                 problem,
                 predictor,
                 corrector,
                 termination,
                 step_eps=1,
                 tol_eps=1e-4,
                 history=True,
                 debug=False,
                 limits=None,
                 **kwargs):

        self.problem = problem
        self.corrector = corrector
        self.predictor = predictor

        self.continuators = []
        for d in [-1, 1]:
            self.continuators.append(DsContinuation(problem=problem,
                                                    predictor=predictor,
                                                    corrector=corrector,
                                                    termination=termination,
                                                    dir=d,
                                                    tol_eps=tol_eps,
                                                    debug=debug,
                                                    limits=limits,
                                                    history=history,
                                                    step_eps=step_eps))

    def run(self, x0):
        results = [cont.run(x0) for cont in self.continuators]
        return self.format_results(results)

    def format_results(self, results):
        compiled_res = {}
        for key in ['X', 'F', 'vs', 'as']:
            if key in ['X', 'F']:
                # remove last element to regularize X,F dimensions with vs,as
                compiled_res[key + "_r"] = np.vstack([res['population'][key][1:-1, :] for res in results])
            # remove first element which does not belong to the PF
            compiled_res[key] = np.vstack([res['population'][key][1:, :] for res in results])

        tot_f_evals = self.predictor.n_f_evals + self.corrector.n_f_evals + self.corrector.t_fun.n_f_evals + \
                      sum([c.n_f_evals for c in self.continuators])
        tot_grad_evals = self.predictor.n_grad_evals + self.corrector.n_grad_evals + self.corrector.t_fun.n_grad_evals + \
                         sum([c.n_grad_evals for c in self.continuators])

        evaluations = {'f': {'total': tot_f_evals,
                             'predictor': self.predictor.n_f_evals,
                             'corrector': self.corrector.n_f_evals,
                             'continuation': sum([c.n_f_evals for c in self.continuators]),
                             't_fun_corrector': self.corrector.t_fun.n_f_evals},

                       'grad': {'total': tot_grad_evals,
                                'predictor': self.predictor.n_grad_evals,
                                'corrector': self.corrector.n_grad_evals,
                                'continuation': sum([c.n_grad_evals for c in self.continuators]),
                                't_fun_corrector': self.corrector.t_fun.n_grad_evals},
                       }

        return {'independent': results, 'population': compiled_res, 'evaluations': evaluations}
