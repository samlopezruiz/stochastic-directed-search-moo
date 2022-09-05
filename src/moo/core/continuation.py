import math
import time
from copy import deepcopy, copy

import numpy as np
from numpy.linalg import norm, qr, pinv
from scipy.optimize import fmin_slsqp

from src.moo.core.boxes import Boxes
from src.moo.core.termination import NullTermination
from src.moo.utils.functions import step_size_norm


class Continuation:

    def __init__(self,
                 problem,
                 predictor,
                 corrector,
                 dir,
                 termination,
                 limits=None,
                 history=True,
                 debug=False,
                 verbose=True,
                 ini_descent=True,
                 # max_increment=None,
                 ):

        self.loop_time_start = time.time()
        self.verbose = verbose
        # self.max_increment = max_increment
        self.ini_descent = ini_descent
        self.problem = problem
        self.termination = termination if termination is not None else NullTermination()
        self.corrector = corrector
        self.predictor = predictor
        self.dir = dir
        self.history = history
        self.debug = debug
        self.n_f_evals = 0
        self.n_grad_evals = 0
        self.limits = limits
        self.end_flag = False
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

        self.max_lens = [6, 8, 6 * self.problem.n_obj, 8, 7, 7]
        self.headers = ["Step", "Time (s)", "F(x)", "in queue", "f evals", "g evals"]

    def initialize(self):
        self.descent_n_grad_evals = 0
        self.descent_n_f_evals = 0
        self.ini_x = None
        self.ini_fx = None
        self.x = None
        self.f = None
        self.dx = None
        self.a = np.ones(self.problem.n_obj) / self.problem.n_obj
        self.min_fx = np.ones(self.problem.n_obj) * np.inf

        self.iter = 0
        self.X = []
        self.F = []

        # predictors
        self.X_p = []
        self.F_p = []

        # correctors
        self.X_c = []
        self.F_c = []

        # initial descent history
        self.F_descent = []
        self.X_descent = []

    def reset_counters(self):
        self.corrector.n_f_evals, self.corrector.t_fun.n_f_evals = 0, 0
        self.corrector.n_grad_evals, self.corrector.t_fun.n_grad_evals = 0, 0
        self.predictor.n_f_evals, self.predictor.n_grad_evals = 0, 0

        self.loop_f_evals = self.problem.n_f_evals
        self.loop_g_evals = self.problem.n_grad_evals
        self.loop_time_start = time.time()
        self.ini_time = time.time()
        self.epochs = {'step': [], 'time': [], 'in_queue': [], 'f_evals': [], 'g_evals': []}

    def run(self, x0):

        # now the termination criterion should be set
        if self.termination is None:
            raise Exception("No termination criterion defined and algorithm has no default termination implemented!")

        self.initialize()
        self.x = np.array(x0)
        self.fx = self.problem.evaluate(self.x)
        self.dx = self.problem.gradient(self.x)

        self.n_f_evals += 1
        self.n_grad_evals += 1

        self.ini_x = self.x
        self.ini_fx = self.fx

        self.reset_counters()
        self.print_headers()
        self.a = self.alpha_step()
        if self.ini_descent:
            c, fc, dc = self.ds_x_to_pf(self.x, self.fx)
            self.add_to_queue(c, dc, fc)
            self.predictor.min_fx = fc
        else:
            self.X.append(self.x)
            self.F.append(self.fx)
            self.add_to_queue(self.x, self.dx, self.fx)
            self.predictor.min_fx = self.fx

        # while termination criterion not fulfilled
        while self.has_next():
            self.next()

        # create the result object to be returned
        res = self.result()
        self.reset()

        return res

    def print_headers(self):
        if self.verbose:
            if self.iter == 0:
                print('\n')
                print(' |  '.join('{0:{width}}'.format(x, width=y) for x, y in zip(self.headers, self.max_lens)))
                print('-' * (sum(self.max_lens) + len(self.max_lens) * 3))

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

    def print_totals(self, tot_f_evals, tot_grad_evals):
        row = {'step': self.iter,
               'time': round(time.time() - self.ini_time, 2),
               'fx': "",
               'in_queue': "",
               'f_evals': tot_f_evals,
               'g_evals': round(tot_grad_evals, 2), }

        if self.verbose:
            print('-' * (sum(self.max_lens) + len(self.max_lens) * 3))
            print(' |  '.join('{0:<{width}}'.format(x, width=y) for x, y in zip(row.values(), self.max_lens)))

    def log_epoch(self, to_pf=(False, (0, 0))):
        fx = str(np.round(self.fx_queue[-1], 2)) if len(self.fx_queue) > 0 else "" \
            if not to_pf[0] else str(np.round(to_pf[1], 2))

        row = {'step': self.iter if not to_pf[0] else 'to_pf',
               'time': round(time.time() - self.loop_time_start, 2),
               'fx': fx,
               'in_queue': len(self.fx_queue),
               'f_evals': self.problem.n_f_evals - self.loop_f_evals,
               'g_evals': round(self.problem.n_grad_evals - self.loop_g_evals, 2)}

        for key, item in row.items():
            if key not in ['fx']:
                self.epochs[key].append(item)

        if self.verbose and self.iter > 0:
            print(' |  '.join('{0:<{width}}'.format(x, width=y) for x, y in zip(row.values(), self.max_lens)))

        self.loop_time_start = time.time()
        self.loop_f_evals = self.problem.n_f_evals
        self.loop_g_evals = self.problem.n_grad_evals

    def has_next(self):
        self.log_epoch()
        if len(self.x_queue) == 0:
            self.print_totals(*self.total_evals())
            # print('empty queue: ended with {} it'.format(self.iter))
            print('empty queue')
            return False

        if self.end_flag:
            print('outside limits')
            return False

        has_next = self.termination.has_next(self)
        if not has_next:
            self.print_totals(*self.total_evals())
            print('{} exceeeded'.format(self.termination.name))
        return has_next

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
                # if self.is_good_predictor(fp_i):
                c, fc, dc, history = self.corrector_step(p_i, fp_i, t)
                self.add_to_queue_using_boxes(c, fc, dc)
                self.log_corrector(fc, c, history)

        self.iter += 1

    def ds_x_to_pf(self, x, fx):
        c, fc, dc, history = self.corrector_step(x, fx, t=None)
        self.log_corrector(fc, c, history, ini_descent=True)
        self.log_epoch(to_pf=(True, fc))

        self.descent_n_f_evals = self.corrector.n_f_evals + self.corrector.t_fun.n_f_evals
        self.descent_n_grad_evals = self.corrector.n_grad_evals + self.corrector.t_fun.n_grad_evals
        self.corrector.n_f_evals, self.corrector.t_fun.n_f_evals = 0, 0
        self.corrector.n_grad_evals, self.corrector.t_fun.n_grad_evals = 0, 0
        return c, fc, dc

    def log_predictor(self, v_i, p, fp):
        self.vs.append(v_i)
        self.as_.append(self.a)
        self.X_p.append(p)
        self.F_p.append(fp)

    def log_corrector(self, fc, c, history, ini_descent=False):
        # if self.limits is None or not np.any(fc >= self.limits):
        if ini_descent:
            self.X_descent.append(history['c_hist'])
            self.F_descent.append(history['fc_hist'])
            self.X_c.append([])
            self.F_c.append([])
        else:
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

    # def within_limits(self, fx):
    #     return self.limits is None or not np.any(fx >= self.limits)
    #
    # def within_max_increment(self, fx):
    #     return self.max_increment is None or (np.sum(fx) - np.sum(self.min_fx)) / np.sum(
    #         self.min_fx) < self.max_increment

    def add_to_queue_using_boxes(self, c, fc, dc):
        # if self.within_limits(fc) and self.within_max_increment(fc):
        self.add_to_queue(c, dc, fc)

    def add_to_queue(self, c, dc, fc):
        self.min_fx = fc if sum(fc) < sum(self.min_fx) else self.min_fx
        self.x_queue.append(c)
        self.fx_queue.append(fc)
        self.dx_queue.append(dc)

    # def is_good_predictor(self, fp_i):
    #     return self.within_limits(fp_i) and self.within_max_increment(fp_i)

    def total_evals(self):
        tot_f_evals = self.descent_n_f_evals + self.predictor.n_f_evals + self.corrector.n_f_evals + self.corrector.t_fun.n_f_evals + \
                      self.n_f_evals
        tot_grad_evals = self.descent_n_grad_evals + self.predictor.n_grad_evals + self.corrector.n_grad_evals + self.corrector.t_fun.n_grad_evals + \
                         self.n_grad_evals
        return tot_f_evals, tot_grad_evals

    def result(self):
        descent = {'ini_x': self.ini_x,
                   'ini_fx': self.ini_fx,
                   'X': self.X_descent,
                   'F': self.F_descent}

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
                      'history': self.epochs
                      }

        tot_f_evals, tot_grad_evals = self.total_evals()

        evaluations = {'f': {'total': tot_f_evals,
                             'descent': self.descent_n_f_evals,
                             'predictor': self.predictor.n_f_evals,
                             'corrector': self.corrector.n_f_evals + self.corrector.t_fun.n_f_evals,
                             'continuation': self.n_f_evals},

                       'grad': {'total': tot_grad_evals,
                                'descent': self.descent_n_grad_evals,
                                'predictor': self.predictor.n_grad_evals,
                                'corrector': self.corrector.n_grad_evals + self.corrector.t_fun.n_grad_evals,
                                'continuation': self.n_grad_evals},
                       }

        return {'population': population, 'descent': descent, 'evaluations': evaluations}

    def reset(self):
        self.end_flag = False
        self.n_f_evals = 0
        self.n_grad_evals = 0

    def _alpha_func(self):
        # quadratic optimization core (qop)
        # for directed search
        alpha = fmin_slsqp(func=lambda a: norm(np.matmul(a, self.dx), ord=2) ** 2,
                           x0=self.a,
                           f_eqcons=lambda a: np.sum(a) - 1,
                           f_ieqcons=lambda a: a,
                           iprint=0)

        return alpha

    # implements following methods for specific implementation
    def _v_directions(self, q):
        raise NotImplementedError

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
                 ini_descent=True,
                 **kwargs):
        super().__init__(problem=problem,
                         predictor=predictor,
                         corrector=corrector,
                         termination=termination,
                         ini_descent=ini_descent,
                         dir=dir,
                         limits=limits,
                         debug=debug,
                         history=history,
                         **kwargs
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
        res = self.predictor.do(self.a, self.x, self.fx, t, v, algorithm=self)

        return res

    def _corrector_step(self, p, fp, t):
        res = self.corrector.do(p, fp, -self.a, t)

        return res


class ContinuationBoxes(DsContinuation):
    def __init__(self,
                 problem,
                 predictor,
                 corrector,
                 limits,
                 tree_h_max=None,
                 step_eps=1e-4,
                 history=True,
                 verbose=True,
                 termination=None,
                 tree_h_coef=0.7,
                 ):

        super().__init__(problem=problem,
                         predictor=predictor,
                         corrector=corrector,
                         termination=termination,
                         dir=1,
                         step_eps=step_eps,
                         verbose=verbose,
                         history=history,
                         )

        # Tree height h_max
        if tree_h_max is None:
            tree_h_max = problem.n_obj * math.ceil(
                math.log(max(limits[:, 1] - limits[:, 0]) / (step_eps * tree_h_coef), 2))
        print('using h_max: {}'.format(tree_h_max))
        self.boxes = Boxes(tree_h_max, limits)

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
        # case when fx is outside the limits
        if self.boxes.has_box(fp_i) is None:
            return False

        # return False if exist a box for that point
        return not self.boxes.has_box(fp_i)


class BiDirectionalDsContinuation:
    def __init__(self,
                 problem,
                 predictor,
                 corrector,
                 termination,
                 # max_increment=None,
                 single_descent=True,
                 **kwargs):

        # self.max_increment = max_increment
        self.problem = problem
        self.corrector = corrector
        self.predictor = predictor
        self.single_descent = single_descent

        self.continuators = []
        for d in [-1, 1]:
            self.continuators.append(DsContinuation(problem=problem,
                                                    predictor=copy(predictor),
                                                    corrector=copy(corrector),
                                                    termination=termination,
                                                    # max_increment=max_increment,
                                                    dir=d,
                                                    ini_descent=(d == -1) if single_descent else True,
                                                    **kwargs))

    def run(self, x0):
        """
        run first continuation with initial descent
        second continuation starts from PF
        :param x0:
        :return:
        """
        if self.single_descent:
            results = []
            results.append(self.continuators[0].run(x0))
            if len(self.continuators[0].X_descent) > 0:
                results.append(self.continuators[1].run(self.continuators[0].X_descent[0][-1]))
            else:
                results.append(self.continuators[1].run(x0))
        else:
            results = [cont.run(x0) for cont in self.continuators]

        return self.format_results(results)

    def format_results(self, results):
        compiled_res = {}
        for key in ['X', 'F', 'vs', 'as']:
            if key in ['X', 'F']:
                # remove last element to regularize X,F dimensions with vs,as
                compiled_res[key + "_r"] = np.vstack([res['population'][key][1:-1, :] for res in results])

            if self.single_descent:
                # remove first element of second continuation because it is the same as the element of the first continuation
                compiled_res[key] = np.vstack([res['population'][key][i:, :] for i, res in enumerate(results)])
            else:
                compiled_res[key] = np.vstack([res['population'][key] for res in results])

        evaluations = {'f': {'descent': sum([c.descent_n_f_evals for c in self.continuators]),
                             'predictor': sum([c.predictor.n_f_evals for c in self.continuators]),
                             'corrector': sum([(c.corrector.n_f_evals + c.corrector.t_fun.n_f_evals) for c in
                                               self.continuators]),
                             'continuation': sum([c.n_f_evals for c in self.continuators])},

                       'grad': {'descent': sum([c.descent_n_grad_evals for c in self.continuators]),
                                'predictor': sum([c.predictor.n_grad_evals for c in self.continuators]),
                                'corrector': sum([(c.corrector.n_grad_evals + c.corrector.t_fun.n_grad_evals) for c in
                                                  self.continuators]),
                                'continuation': sum([c.n_grad_evals for c in self.continuators])},
                       }

        evaluations['f']['total'] = sum([evaluations['f'][key] for key in evaluations['f'].keys()])
        evaluations['grad']['total'] = sum([evaluations['grad'][key] for key in evaluations['grad'].keys()])
        return {'independent': results, 'population': compiled_res, 'evaluations': evaluations}

    def reset(self):
        for c in self.continuators:
            c.reset()
