import os
import time

import joblib
import numpy as np
from numpy.linalg import pinv
from tabulate import tabulate

from src.moo.core.continuation import BiDirectionalDsContinuation
from src.moo.factory import get_corrector, get_predictor, get_tfun, get_cont_termination
from src.timeseries.moo.cont.core.problem import TsQuantileProblem
from src.timeseries.utils.continuation import get_q_moo_params_for_problem
from src.timeseries.utils.filename import get_result_folder
from src.timeseries.utils.moo import get_hypervolume
from src.utils.plot import plot_bidir_2D_points_vectors

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
    problem = TsQuantileProblem(y_train=model_params['y_train'],
                                x_train=model_params['x_train'],
                                y_valid=model_params['y_valid'],
                                x_valid=model_params['x_valid'],
                                model=model_params['model'].model,
                                eval_fs=[model_params['loss']],
                                n_obj=2,
                                # constraints_limits=limits,
                                quantile_ix=0,
                                base_batch_size=2 ** 11,
                                moo_batch_size=2 ** 11,
                                moo_model_size='medium')

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
    predictor = get_predictor('no_adjustment', problem=problem)

    corrector = get_corrector('rank',
                              problem=problem,
                              t_fun=get_tfun('weighted_dominance',
                                             problem=problem,
                                             eps=1e-4,
                                             maxiter=50),
                              a_fun=lambda a, dx: a,
                              # use_cvxpy=True,
                              # recalc_v=True,
                              step_eps=2e-2,
                              in_pf_eps=1e-2,  # in_pf_eps=1e-4,
                              maxiter=10)

    ds_cont = BiDirectionalDsContinuation(problem=problem,
                                          predictor=predictor,
                                          corrector=corrector,
                                          limits=limits,
                                          verbose=True,
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
                   tablefmt='psql',
                   headers=list(results['evaluations'][list(results['evaluations'].keys())[0]].keys())))

    # %%
    file_path = os.path.join(results_folder, 'img', cfg['problem_name'])
    train_population = [res['population'] for res in results['independent']]
    # train_population = [results['independent'][0]['population']]

    pareto = {'pf': None}
    plot_bidir_2D_points_vectors(train_population,
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
                                 return_fig=False)

    # %%
    hv_train = get_hypervolume(results['population']['F'], ref=[2., 2.])
    print(f'hv: {hv_train}')
