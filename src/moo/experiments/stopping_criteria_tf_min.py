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
    problem.grad_from_batches = True
    problem.grad_batch_size = 2 ** 13

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
    res = results['population']

    i = 1
    x = res['X'][i]
    fx = res['F'][i]
    a = res['as'][i]
    dx = problem.gradient(x)

    a0 = res['as'][0]
    x0 = problem.original_x
    f0 = problem.evaluate(x0)
    d0 = problem.gradient(x0)

# %%
import tensorflow as tf
#
# d = tf.constant(a, dtype=tf.float32)
# var = tf.zeros((1, dx.shape[1] + 1))
# dx_tf = tf.Variable(dx)
# loss_fn = lambda var: (tf.norm(var[0, :-1], ord=2, axis=0) ** 2) / 2 - var[0, -1]
# eq_const = tf.linalg.matmul(dx, tf.reshape(var[:, :-1], [-1, 1])) - \
#            tf.reshape(tf.math.scalar_mul(var[0, -1], d), [-1, 1])
#
# v_tf = tf.Variable(tf.zeros((1, dx.shape[1])), dtype=tf.float32, name='v')
# delta_tf = tf.Variable(0.0, dtype=tf.float32, name='delta')
# d_tf = tf.constant(d, dtype=tf.float32)
# dx_tf = tf.constant(dx, dtype=tf.float32)
#
# vars = [delta_tf] + [v_tf]
# with tf.GradientTape(persistent=True) as tape:
#     loss = (tf.norm(v_tf, ord=2, axis=1) ** 2) / 2 - delta_tf
#
# grads = tape.gradient(loss, vars)

# %%

# Use the GitHub version of TFCO
# !pip install git+https://github.com/google-research/tensorflow_constrained_optimization
# import tensorflow_constrained_optimization as tfco
#
#
# class SampleProblem(tfco.ConstrainedMinimizationProblem):
#     def __init__(self, dx, d):
#         self.v = tf.Variable(tf.zeros((1, dx.shape[1])), dtype=tf.float32, name='v')
#         self.delta = tf.Variable(0.0, dtype=tf.float32, name='delta')
#         self.dx = tf.constant(dx, dtype=tf.float32)
#         self.d = tf.constant(d, dtype=tf.float32)
#
#     @property
#     def num_constraints(self):
#         return 4
#
#     def objective(self):
#         return (tf.norm(self.v, ord=2, axis=1) ** 2) / 2 - self.delta
#
#     def constraints(self):
#         eq_const = tf.linalg.matmul(self.dx, tf.reshape(self.v, [-1, 1])) - \
#                    tf.reshape(tf.math.scalar_mul(self.delta, self.d), [-1, 1])
#
#         # constraints = tf.concat([eq_const, -eq_const], axis=0)
#         constraints = tf.reshape(tf.concat([eq_const, -eq_const], axis=0), [-1])
#         return constraints
#
#
# min_problem = SampleProblem(dx, a)
#
# optimizer = tfco.LagrangianOptimizer(
#     optimizer=tf.optimizers.Adagrad(learning_rate=0.05),
#     num_constraints=min_problem.num_constraints
# )
#
# optimizer = tfco.ConstrainedOptimizer(
#     formulation=Formulation(),
#     optimizer=tf.optimizers.Adagrad(learning_rate=0.05),
#     num_constraints=min_problem.num_constraints
# )
#
# const = min_problem.constraints()
# loss = min_problem.objective()
#
# var_list = list(min_problem.trainable_variables) + optimizer.trainable_variables()
# var_list = list(min_problem.trainable_variables)
# var_list = [min_problem.delta, min_problem.v] + list(min_problem.trainable_variables) + optimizer.trainable_variables()


# %%
# for i in range(10000):
#     optimizer.minimize(min_problem, var_list=var_list)
#     if i % 1000 == 0:
#         print(f'step = {i}')
#         print(f'loss = {min_problem.objective().numpy()[0]}')
#         print(f'constraint = {min_problem.constraints().numpy().T}')
#
# # %%
# delta_res, v_res = [var.numpy() for var in min_problem.trainable_variables]
#
# d = np.matmul(dx, v_res.reshape(-1, 1)) / delta_res
# print('{}, {}'.format(a, d))

# %%
from numpy.linalg import norm


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


v_res0, delta_res0 = in_pareto_front(dx, a)
d = np.matmul(dx, v_res0) / delta_res0
print('{}, {}'.format(a, d))
print('delta: {}'.format(delta_res0))

# %%
import cvxpy as cp


def in_pareto_front(dx, d):
    v_cp = cp.Variable(dx.shape[1])
    delta_cp = cp.Variable()

    eq = cp.matmul(dx, v_cp) - delta_cp * d == 0.0
    loss = (cp.norm(v_cp, p=2) ** 2) / 2 - delta_cp

    constraints = [0 <= delta_cp, eq]
    objective = cp.Minimize(loss)

    prob = cp.Problem(objective, constraints)

    result = prob.solve()
    v = v_cp.value
    return v_cp.value, delta_cp.value


v_res0, delta_res0 = in_pareto_front(dx, a)
d = np.matmul(dx, v_res0) / delta_res0
print('{}, {}'.format(a, d))
print('delta: {}'.format(delta_res0))
