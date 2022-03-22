import os
import time
from pprint import pprint

import joblib
import numpy as np
import pandas as pd
from cvxpy.constraints.constraint import Constraint
from numpy.linalg import pinv, matrix_rank
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_reference_directions
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow_constrained_optimization.python.train.constrained_optimizer import Formulation

from src.moo.core.continuation import BiDirectionalDsContinuation
from src.moo.factory import get_corrector, get_predictor, get_tfun, get_cont_termination
from src.moo.nn.problem import TsQuantileProblem, GradientTestsProblem
from src.moo.nn.utils import predict_from_batches, batch_array
from src.moo.utils.functions import in_pareto_front
from src.timeseries.utils.continuation import get_q_moo_params_for_problem
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.moo import get_hypervolume
from src.utils.plot import plot_bidir_2D_points_vectors, plot_2D_points_traces, plot_2D_points_vectors, plot_traces

if __name__ == '__main__':
    # %%
    cfg = {'save_plots': False,
           'problem_name': 'ts_q'
           }

    solve_moea = False
    project = 'snp'
    results_cfg = {'experiment_name': '60t_ema_q159',
                   'results': 'TFTModel_ES_ema_r_q159_lr01_pred'}

    results_folder = get_result_folder(results_cfg, project)
    model_results = joblib.load(os.path.join(results_folder, results_cfg['results'] + '.z'))
    model_params = get_q_moo_params_for_problem(project, model_results, shuffle_data=True)

    # %%
    t0 = time.time()
    limits = np.array([1., 1.])
    problem = GradientTestsProblem(y_train=model_params['y_train'],
                                   x_train=model_params['x_train'],
                                   y_valid=model_params['y_valid'],
                                   x_valid=model_params['x_valid'],
                                   model=model_params['model'].model,
                                   eval_fs=[model_params['loss']],
                                   n_obj=2,
                                   constraints_limits=limits,
                                   quantile_ix=0,
                                   pred_batch_size=2 ** 7,
                                   grad_batch_size=2 ** 7,
                                   moo_model_size='medium',
                                   grad_from_batches=True)

    print('init core time: {}'.format(round(time.time() - t0, 4)))

    # %%e
    t0 = time.time()
    f0 = problem.evaluate(problem.original_x)
    print('eval time: {:.4} s'.format(time.time() - t0))

    t0 = time.time()
    dx = problem.gradient(problem.original_x)
    print('grad time: {:.4} s'.format(time.time() - t0))

    t0 = time.time()
    dx_1 = pinv(dx)
    print('inverse time: {:.4} s'.format(time.time() - t0))

    # %%
    problem.constraints_limits = None
    problem.n_constr = 0
    # problem.grad_from_batches = True
    # problem.grad_batch_size = 2 ** 13

    predictor = get_predictor('no_adjustment', problem=problem)

    corrector = get_corrector('delta_criteria',
                              problem=problem,
                              t_fun=get_tfun('weighted_dominance',
                                             problem=problem,
                                             eps=1e-4,
                                             maxiter=50),
                              a_fun=lambda a, dx: a,
                              cvxpy=True,
                              step_eps=2e-2,
                              in_pf_eps=3e-4,
                              maxiter=20)

    ds_cont = BiDirectionalDsContinuation(problem=problem,
                                          predictor=predictor,
                                          corrector=corrector,
                                          limits=limits,
                                          step_eps=2e-2,
                                          termination=get_cont_termination('n_iter', maxiter=2),
                                          history=True)

    results = ds_cont.run(np.reshape(problem.original_x, (-1)))

    # %%
