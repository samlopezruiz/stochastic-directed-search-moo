import os
import time
from copy import deepcopy

import joblib
import numpy as np
from numpy.linalg import pinv
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_reference_directions
from pymoo.optimize import minimize
from tabulate import tabulate

from src.moo.core.continuation import BiDirectionalDsContinuation, ContinuationBoxes
from src.moo.factory import get_corrector, get_predictor, get_tfun, get_cont_termination
from src.timeseries.moo.cont.core.problem import TsQuantileProblem
from src.moo.nn.problem_old import GradientTestsProblem
from src.moo.nn.utils import batch_array
from src.moo.utils.functions import SMA
from src.timeseries.utils.continuation import get_q_moo_params_for_problem
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.moo import get_hypervolume
from src.utils.plot import plot_bidir_2D_points_vectors, plot_2D_points_vectors, plot_traces, plot_points_4d, \
    plot_points_centers_2d, plot_boxes_2d

if __name__ == '__main__':
    # %%
    cfg = {'save_plots': False,
           'problem_name': 'ts_q',
           'split_model': "small",
           'quantile_ix': 0,
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
    f_max_limit = 1.
    # limits = np.array([f_max_limit] * 3)
    limits = np.array([f_max_limit] * 2)
    problem = TsQuantileProblem(y_train=model_params['y_train'],
                                x_train=model_params['x_train'],
                                y_valid=model_params['y_valid'],
                                x_valid=model_params['x_valid'],
                                model=model_params['model'].model,
                                eval_fs=[model_params['loss']],
                                constraints_limits=limits,
                                quantile_ix=cfg['quantile_ix'],
                                base_batch_size=2 ** 11,
                                moo_batch_size=2 ** 9,
                                moo_model_size=cfg['split_model'])

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
    if solve_moea:
        pop_size, n_gen = 100, 100
        algorithm = NSGA2(
            pop_size=pop_size,
            n_offsprings=pop_size,
            ref_dirs=get_reference_directions("energy", problem.n_obj, pop_size),
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", n_gen)

        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=False,
                       verbose=True)

        X = res.X
        F = res.F

        ixs = np.argsort(F[:, 0])
        pf_moea = F[ixs]
        ps_moea = X[ixs]
        plot_points_4d(pf_moea,
                       markersize=5,
                       title='MOEA method Pareto Front')

    # %%
    problem.constraints_limits = None
    problem.n_constr = 0
    predictor = get_predictor('no_adjustment', problem=problem)

    in_pf_eps_cfg = {'small': {'delta': 1e-4, 'rank': 1e-2, 'projection': 5e-4},
                     'medium': {'delta': 1e-2, 'rank': 8e-3, 'projection': 5e-4}}

    corrector_type = 'delta'
    corrector = get_corrector(corrector_type,
                              problem=problem,
                              t_fun=get_tfun('weighted_dominance',
                                             problem=problem,
                                             eps=1e-4,
                                             maxiter=30),
                              a_fun=lambda a, dx: a,
                              # use_cvxpy=True,
                              # recalc_v=True,
                              batch_gradient=False,
                              mean_grad_stop_criteria=True,
                              batch_ratio_stop_criteria=0.1,
                              step_eps=2e-2,
                              in_pf_eps=in_pf_eps_cfg[cfg['split_model']][corrector_type],  # in_pf_eps=5e-3,
                              maxiter=10,
                              )

    ds_cont = ContinuationBoxes(problem,
                                predictor,
                                corrector,
                                limits=np.array([[0., f_max_limit]] * problem.n_obj),
                                termination=get_cont_termination('n_iter', 1000),
                                tree_h_coef=0.7,
                                tree_h_max=None,
                                step_eps=2e-2,
                                verbose=True
                                )

    # %%
    t0 = time.time()
    problem.n_f_evals, problem.n_grad_evals = 0, 0
    results = ds_cont.run(np.reshape(problem.original_x, (-1)))
    print('time: {} s'.format(round(time.time() - t0, 4)))
    print('f(x) evals: {}, dx(x) evals: {}'.format(problem.n_f_evals, problem.n_grad_evals))

    print(tabulate([[name, *inner.values()] for name, inner in results['evaluations'].items()],
                   tablefmt='psql',
                   headers=list(results['evaluations'][list(results['evaluations'].keys())[0]].keys())))

    # %%
    points = ds_cont.boxes.get_points()
    print('{} solutions found, {} best solutions found'.format(len(points['fx']),
                                                               len(points['fx'][points['best_ix'], :])))
    boxes_edges = ds_cont.boxes.get_boxes()
    box_fig = plot_boxes_2d(boxes_edges, return_fig=True)
    pts_fig = plot_points_centers_2d(points['fx'],
                                     centers=points['c'],
                                     best=points['best_ix'],
                                     return_fig=True,
                                     markersize=6)
    fig_data = [box_fig.data, pts_fig.data]
    plot_traces([box_fig.data, pts_fig.data])

    # %%
    hv_moea_train = get_hypervolume(pf_moea, ref=[2.] * problem.n_obj) if solve_moea else np.nan
    hv_train = get_hypervolume(results['population']['F'], ref=[2.] * problem.n_obj)

    hvs = {}
    for hv, key in zip([hv_train, hv_moea_train], ['train/cont', 'train/moea']):
        hvs[key] = hv

    print(tabulate(hvs.items(),
                   headers=['subset/method', 'hypervolume'],
                   tablefmt='psql',
                   floatfmt=(None, ",.6f"),
                   stralign="right"))
