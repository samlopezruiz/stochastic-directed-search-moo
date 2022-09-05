import os
import time
from copy import deepcopy

import joblib
import numpy as np
import scipy
from numpy.linalg import multi_dot, norm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_reference_directions
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from tabulate import tabulate

from src.moo.core.continuation import BiDirectionalDsContinuation
from src.moo.factory import get_corrector, get_predictor, get_tfun, get_cont_termination
from src.moo.nn.problem_old import TsQuantileProblem
from src.moo.nn.utils import batch_array
from src.timeseries.utils.continuation import get_q_moo_params_for_problem
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.moo import get_hypervolume
from src.utils.plot import plot_bidir_2D_points_vectors, plot_2D_points_vectors, plot_traces

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
    model_params = get_q_moo_params_for_problem(project, model_results)

    # %%
    t0 = time.time()
    limits = None  # np.array([1., 1.])
    problem = TsQuantileProblem(y_train=model_params['y_train'],
                                x_train=model_params['x_train'],
                                # y_valid=model_params['y_valid'],
                                # x_valid=model_params['x_valid'],
                                model=model_params['model'].model,
                                eval_fs=[model_params['loss']],
                                n_obj=2,
                                constraints_limits=limits,
                                quantile_ix=0,
                                base_batch_size=128,
                                moo_model_size='small')
    print('init core time: {}'.format(round(time.time() - t0, 4)))

    # %%
    # t0 = time.time()
    # fx = problem.evaluate(problem.original_x)
    # print('eval time: {:.4} s'.format(time.time() - t0))

    t0 = time.time()
    dx = problem.gradient(problem.original_x)
    print('grad time: {:.4} s'.format(time.time() - t0))

    # %%
    d = np.array([0.5, 0.5]).astype(np.float32)
    var0 = np.zeros((dx.shape[1] + 1)).astype(np.float32)

    # cons = ({'type': 'eq', 'fun': lambda var: np.matmul(dx, var[:-1]) - var[-1] * d})
    cons = ({'type': 'eq', 'fun': lambda var: multi_dot([dx, var[:-1]]) - var[-1] * d})

    bnds = [(None, None)] * dx.shape[1] + [(0, None)]
    res = minimize(lambda var: (norm(var[:-1], ord=2) ** 2) / 2 - var[-1],
                   x0=var0,
                   constraints=cons,
                   bounds=bnds,
                   )
    v, delta = res['x'][:-1], res['x'][-1]

    v /= norm(v)

    # %%
    import tensorflow as tf
    import tensorflow_probability as tfp

    x = tf.Variable(0.)
    loss_fn = lambda: (x - 5.) ** 2
    losses = tfp.math.minimize(loss_fn,
                               num_steps=100,
                               optimizer=tf.optimizers.Adam(learning_rate=0.1))

    # In TF2/eager mode, the optimization runs immediately.
    print("optimized value is {} with loss {}".format(x, losses[-1]))

    # %%
    losses = tfp.math.minimize(
        loss_fn, num_steps=1000, optimizer=tf.optimizers.Adam(learning_rate=0.1),
        convergence_criterion=(
            tfp.optimizer.convergence_criteria.LossNotDecreasing(atol=0.01)))
    print("optimized value is {} with loss {}".format(x, losses[-1]))

    # %%
    # d = np.array([0.5, 0.5]).astype(np.float32).reshape(-1, 1)
    # var0 = np.zeros((dx.shape[1] + 1)).astype(np.float32).reshape(-1, 1)
    #
    # d = csr_matrix(d)
    # dxs = csr_matrix(dx)
    # var0 = csr_matrix(var0)
    # a = dxs * var0[:-1] - d.multiply(var0[-1])
    #
    # # %%
    # var0 = np.zeros((dx.shape[1] + 1)).astype(np.float32).reshape(-1, 1)
    # d = np.array([0.5, 0.5]).astype(np.float32).reshape(-1, 1)
    # var0 = csr_matrix(var0)
    # d = csr_matrix(d)
    # dxs = csr_matrix(dx)
    #
    # cons = ({'type': 'eq', 'fun': lambda var: dxs * var[:-1] - d.multiply(var[-1])})
    # bnds = [(None, None)] * dx.shape[1] + [(0, None)]
    # res = minimize(lambda var: ((scipy.sparse.linalg.norm(var[:-1]) ** 2) / 2 - var[-1]).toarray()[0][0],
    #                x0=var0,
    #                constraints=cons,
    #                bounds=bnds,
    #                )
    # v, delta = res['x'][:-1], res['x'][-1]
    #
    # v /= norm(v)

    # %%
