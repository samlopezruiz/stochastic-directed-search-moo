import os
import time
from copy import deepcopy

import joblib
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination, get_reference_directions
from pymoo.optimize import minimize
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
                                y_valid=model_params['y_valid'],
                                x_valid=model_params['x_valid'],
                                model=model_params['model'].model,
                                eval_fs=[model_params['loss']],
                                n_obj=2,
                                constraints_limits=limits,
                                quantile_ix=0,
                                base_batch_size=128,
                                moo_model_size='small')
    print('init core time: {}'.format(round(time.time() - t0, 4)))

    # %%
    t0 = time.time()
    f0 = problem.evaluate(problem.original_x)
    print('eval time: {:.4} s'.format(time.time() - t0))

    t0 = time.time()
    d0 = problem.gradient(problem.original_x)
    print('grad time: {:.4} s'.format(time.time() - t0))

    # %%

    # %%
    if solve_moea:
        pop_size, n_gen = 10, 10
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
        pf = F[ixs]
        ps = X[ixs]

    # %%
    problem.constraints_limits = None
    problem.n_constr = 0
    predictor = get_predictor('no_adjustment', problem=problem)

    corrector = get_corrector('delta_criteria',
                              problem=problem,
                              t_fun=get_tfun('weighted_dominance',
                                             problem=problem,
                                             eps=1e-5,
                                             maxiter=50),
                              a_fun=lambda a, dx: a,
                              step_eps=2e-2,
                              in_pf_eps=1e-5,  # in_pf_eps=5e-3,
                              maxiter=20)

    ds_cont = BiDirectionalDsContinuation(problem=problem,
                                          predictor=predictor,
                                          corrector=corrector,
                                          limits=limits,
                                          step_eps=2e-2,
                                          termination=get_cont_termination('tol', tol=8e-4),
                                          history=True)

    # %%
    t0 = time.time()
    problem.n_f_evals, problem.n_grad_evals = 0, 0
    results = ds_cont.run(np.reshape(problem.original_x, (-1)))
    print('time: {} s'.format(round(time.time() - t0, 4)))
    print('f(x) evals: {}, dx(x) evals: {}'.format(problem.n_f_evals, problem.n_grad_evals))

    print(tabulate([[name, *inner.values()] for name, inner in results['evaluations'].items()],
                   headers=list(results['evaluations'][list(results['evaluations'].keys())[0]].keys())))

    # %%
    print('...evaluating with valid data')
    pred_corr_keys = {'X_p': 'F_p', 'X_c': 'F_c'}
    valid_population = [deepcopy(res['population']) for res in results['independent']]

    for pop in valid_population:
        pop['F'] = np.array([problem.eval_valid(ind) for ind in pop['X']])
        for x, f in pred_corr_keys.items():
            ans = []
            for epoch in pop[x]:
                ans.append([problem.eval_valid(ind) for ind in epoch])
            pop[f] = ans

    # %%
    F_valid = np.array([problem.eval_valid(x) for x in results['population']['X']])
    plot_2D_points_vectors(results['population']['F'],
                           vectors=F_valid - results['population']['F'],
                           pareto=None,
                           scale=0.5,
                           arrow_scale=0.4,
                           point_name='train',
                           vector_name='valid',
                           markersize=5,
                           save=False,
                           save_png=False,
                           file_path=None,
                           title=None,
                           size=(1980, 1080),
                           )

    # %%
    file_path = os.path.join(results_folder, 'img', cfg['problem_name'])
    train_population = [res['population'] for res in results['independent']]

    F_valid_sorted = F_valid[np.argsort(F_valid[:, 0])]
    pareto = {'pf': None}
    train_fig = plot_bidir_2D_points_vectors(train_population,
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
                                             plot_ps=False,
                                             return_fig=True)

    valid_fig = plot_bidir_2D_points_vectors(valid_population,
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
                                             plot_ps=False,
                                             return_fig=True)

    plot_traces([train_fig.data, valid_fig.data])

    if solve_moea:
        pareto = {'pf': pf if solve_moea else None, 'ps': ps if solve_moea else None}
        plot_bidir_2D_points_vectors(train_population,
                                     pareto,
                                     arrow_scale=0.4,
                                     markersize=5,
                                     save=cfg['save_plots'],
                                     save_png=False,
                                     file_path=file_path,
                                     size=(1980, 1080),
                                     plot_arrows=True,
                                     plot_points=True,
                                     plot_ps=False)

    # %%
    hv_moea_train = get_hypervolume(pf, ref=[2., 2.]) if solve_moea else np.nan
    hv_train = get_hypervolume(results['population']['F'], ref=[2., 2.])
    hv_valid = get_hypervolume(F_valid, ref=[2., 2.])
    hv_moea_valid = get_hypervolume(np.array([problem.eval_valid(x) for x in ps]),
                                    ref=[2., 2.]) if solve_moea else np.nan

    hvs = {}
    for hv, key in zip([hv_train, hv_valid, hv_moea_train, hv_moea_valid],
                       ['train/cont', 'valid/cont', 'train/moea', 'valid/moea']):
        hvs[key] = hv

    print(tabulate(hvs.items(),
                   headers=['subset/method', 'hypervolume'],
                   tablefmt='psql',
                   floatfmt=(None, ",.6f"),
                   stralign="right"))
