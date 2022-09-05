import os
import time
from pprint import pprint

import joblib
import numpy as np
import pandas as pd
from numpy.linalg import pinv, matrix_rank
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_reference_directions
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer

from src.moo.core.continuation import BiDirectionalDsContinuation
from src.moo.factory import get_corrector, get_predictor, get_tfun, get_cont_termination
from src.moo.nn.problem_old import TsQuantileProblem, GradientTestsProblem
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
                                   pred_batch_size=2 ** 9,
                                   grad_batch_size=2 ** 9,
                                   moo_model_size='small',
                                   grad_from_batches=False)

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
    problem.moo_from_batches = True
    problem.moo_batch_size = 2 ** 13

    predictor = get_predictor('no_adjustment', problem=problem)

    corrector = get_corrector('delta_criteria',
                              problem=problem,
                              t_fun=get_tfun('weighted_dominance',
                                             problem=problem,
                                             eps=1e-4,
                                             maxiter=50),
                              a_fun=lambda a, dx: a,
                              step_eps=2e-2,
                              in_pf_eps=3e-4,
                              maxiter=20)

    ds_cont = BiDirectionalDsContinuation(problem=problem,
                                          predictor=predictor,
                                          corrector=corrector,
                                          limits=limits,
                                          step_eps=2e-2,
                                          termination=get_cont_termination('tol', tol=8e-4),
                                          history=True)

    t0 = time.time()
    problem.n_f_evals, problem.n_grad_evals = 0, 0
    results = ds_cont.run(np.reshape(problem.original_x, (-1)))
    print('time: {} s'.format(round(time.time() - t0, 4)))
    print('f(x) evals: {}, dx(x) evals: {}'.format(problem.n_f_evals, problem.n_grad_evals))
    print(tabulate([[name, *inner.values()] for name, inner in results['evaluations'].items()],
                   headers=list(results['evaluations'][list(results['evaluations'].keys())[0]].keys()),
                   tablefmt='psql'))

    hv = round(get_hypervolume(results['population']['F'], ref=[2., 2.]), 6)

    # %%
    file_path = os.path.join(results_folder, 'img', cfg['problem_name'])

    figs_data = []
    pareto = {'pf': None}
    plot_populations = [res['population'] for res in results['independent']]

    plot_bidir_2D_points_vectors(plot_populations,
                                 pareto,
                                 arrow_scale=0.4,
                                 markersize=5,
                                 pareto_marker_mode='markers+lines',
                                 save=cfg['save_plots'],
                                 save_png=False,
                                 file_path=file_path,
                                 size=(1980, 1080),
                                 plot_arrows=True,
                                 plot_points=True,
                                 plot_ps=False)

    # %% stopping criteria
    from numpy.linalg import norm

    res = results['population']
    u, s, v = np.linalg.svd(dx)
    rank = np.sum(s > 0.1)
    tols, deltas, ranks = [], [], []
    for x, a in zip(res['X'], res['as']):
        dx = problem.gradient(x)
        tols.append(norm(np.dot(dx.T, a.reshape(-1, 1))) ** 2)
        v, delta = in_pareto_front(dx, a)
        deltas.append(delta)
        ranks.append(matrix_rank(dx, tol=1e-4))

    df = pd.DataFrame()
    df['fx_0'] = res['F'][:, 0]
    df['fx_1'] = res['F'][:, 1]
    df['tol'] = tols
    df['delta'] = deltas
    df['rank'] = ranks

    # %%
    i = 5
    x = res['X'][i]
    fx = res['F'][i]
    a = res['as'][i]
    dx = problem.gradient(x)

    a0 = res['as'][0]
    x0 = problem.original_x
    f0 = problem.evaluate(x0)
    d0 = problem.gradient(x0)


    # %%
    def in_pareto_front(dx, d):
        var0 = np.zeros((dx.shape[1] + 1))

        cons = ({'type': 'eq', 'fun': lambda var: np.matmul(dx, var[:-1]) - var[-1] * d})

        bnds = [(None, None)] * dx.shape[1] + [(0, None)]
        res = minimize(lambda var: (norm(var[:-1], ord=2) ** 2) / 2 - var[-1],
                       x0=var0,
                       constraints=cons,
                       bounds=bnds)
        v, delta = res['x'][:-1], res['x'][-1]

        return v, delta


    # %%
    # v, delta = in_pareto_front(dx, a)
    # print(f'delta: {delta}')
    # print(f'norm(v): {(norm(v, ord=2) ** 2) / 2}')
    # print(f'f_min: {((norm(v, ord=2) ** 2) / 2) - delta}')
    #
    # tol = norm(np.dot(dx.T, a.reshape(-1, 1))) ** 2
    # print(f'tol: {tol}')
    #
    # d = np.matmul(dx, v) / delta
    # print('{}, {}'.format(a, d))

    print('-' * 20)
    v, delta = in_pareto_front(d0, a0)
    print(f'delta: {delta}')
    print(f'norm(v): {(norm(v, ord=2) ** 2) / 2}')
    print(f'f_min: {((norm(v, ord=2) ** 2) / 2) - delta}')

    tol = norm(np.dot(d0.T, a0.reshape(-1, 1))) ** 2
    print(f'tol: {tol}')

    d = np.matmul(d0, v) / delta
    print('{}, {}'.format(a0, d))

    # %%
    dx, a = d0, a0
    v_plus = np.matmul(pinv(dx), a)
    ans = np.matmul(dx, v_plus)

    v_plus /= norm(v_plus)
    delta = np.mean(np.matmul(dx, v_plus) / a)

    print(f'norm(v): {(norm(v_plus, ord=2) ** 2) / 2}, delta: {delta}')
    print(f'f_min: {((norm(v_plus, ord=2) ** 2) / 2) - delta}')

    # %%
    alpha = 0.5
    v_plus *= alpha
    delta_1 = np.matmul(dx, v_plus) / a
    print('{} {}'.format(delta_1, delta_1[0] - delta_1[1]))
    delta = np.mean(delta_1)
    print(f'norm(v): {(norm(v_plus, ord=2) ** 2) / 2}, delta: {delta}')
    print(f'f_min: {((norm(v_plus, ord=2) ** 2) / 2) - delta}')

    # %%
    d = np.matmul(dx, v_plus) / delta
    print('{}, {}'.format(a, d))
# %%
# # %%
# import tensorflow as tf
# import tensorflow_probability as tfp
#
# x = tf.Variable(0.)
# loss_fn = lambda: (x - 5.) ** 2
# losses = tfp.math.minimize(loss_fn,
#                            num_steps=100,
#                            optimizer=tf.optimizers.Adam(learning_rate=0.1))
#
# # In TF2/eager mode, the optimization runs immediately.
# print("optimized value is {} with loss {}".format(x, losses[-1]))
#
# # %%
# d = tf.constant(a, dtype=tf.float32)
# var = tf.zeros((1, dx.shape[1] + 1))
# dx_tf = tf.Variable(dx)
# loss_fn = lambda var: (tf.norm(var[0, :-1], ord=2, axis=0) ** 2) / 2 - var[0, -1]
# eq_const = tf.linalg.matmul(dx, tf.reshape(var[:, :-1], [-1, 1])) - \
#            tf.reshape(tf.math.scalar_mul(var[0, -1], d), [-1, 1])
#
# # %%
# import tensorflow as tf
#
# # Use the GitHub version of TFCO
# # !pip install git+https://github.com/google-research/tensorflow_constrained_optimization
# import tensorflow_constrained_optimization as tfco
#
#
# class SampleProblem(tfco.ConstrainedMinimizationProblem):
#     def __init__(self, dx, d):
#         self.v = tf.Variable(tf.zeros((1, dx.shape[1])), dtype=tf.float32, name='v')
#         self.delta = tf.Variable(0.0, dtype=tf.float32, name='delta')
#         self.dx = dx
#         self.d = tf.constant(d, dtype=tf.float32)
#
#     @property
#     def num_constraints(self):
#         return 4
#
#     def objective(self):
#         return (tf.norm(self.v, ord=2, axis=0) ** 2) / 2 - self.delta
#
#     def constraints(self):
#         eq_const = tf.linalg.matmul(self.dx, tf.reshape(self.v, [-1, 1])) - \
#                    tf.reshape(tf.math.scalar_mul(self.delta, self.d), [-1, 1])
#
#         constraints = tf.reshape(tf.concat([eq_const, -eq_const], axis=0), [-1])
#         return constraints
#
#
# problem = SampleProblem(dx, a)
#
# optimizer = tfco.LagrangianOptimizer(
#     optimizer=tf.optimizers.Adagrad(learning_rate=0.1),
#     num_constraints=problem.num_constraints
# )
#
# const = problem.constraints()
#
# var_list = list(problem.trainable_variables) + optimizer.trainable_variables()
#
# for i in range(100):
#     optimizer.minimize(problem, var_list=var_list)
#     if i % 10 == 0:
#         print(f'step = {i}')
#         print(f'loss = {loss_fn()}')
#         print(f'constraint = {(x + y).numpy()}')
#         print(f'x = {x.numpy()}, y = {y.numpy()}')

# %%

# %%
